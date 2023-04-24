from dataio_piezo import LowResDataset
import torch
from torch.utils.data import DataLoader, ConcatDataset, RandomSampler
from soundsig.signal import bandpass_filter
from soundsig.sound import spectrogram, fundEstimator, spec_colormap
from tqdm import tqdm
import sklearn
import matplotlib.pyplot as plt
import os
import csv
import math
import numpy as np


PATH = "./"
TRAIN_PATH = "./data/train/"
TEST_PATH = "./data/test/"
FMIN = 100 # Min frequency in Hz
FMAX = 5000 # Max frequency in Hz
SR = 50000 # Sampling rate in Hz. Actual logger sampling rates can vary by +-2 Hz
TH = 0.27 # Prediction threshold
SPEC_SR = 1000 # Spectogram param, 1000 (default) for Protonet, CRNN & ResNet, 330 for GoogLeNet
FREQ_SPACING = 100 # Spectogram param, 73 for CRNN, 100 (default) for Protonet & ResNet, 146 for GoogLeNet
VOC_BS = 3 # Vocalization batch size
NOISE_BS = 15 # Noise batch size
PENALTY = 10 # Penalty for missing a vocalization, should be greater than (NOISE_BS/VOC_BS)
NUM_EPOCHS = 20 # Number of epochs to train
BEST_MODEL_NAME = "./checkpoints/protonetmp1.pt"
TEST_ONLY = False # Whether to only test the model
USE_GPU = False
VAL = 0.2 # Percentage of data used for cross-validation
MODEL = "ProtonetMP()"
PREDICTIONS = "./predictions/protonetmp1.csv"
FIGURE = "./figures/protonetmp1.png"

def color_spec(t, freq, spec, ax=None, ticks=True, fmin=None, fmax=None, colormap=None, colorbar=True, log = True, dBNoise = 50):

    colormap = plt.get_cmap('SpectroColorMap')

    ex = (t.min(), t.max(), freq.min(), freq.max())
    plotSpect = np.abs(spec)

    if log == True and dBNoise is not None:
        plotSpect = 20*np.log10(plotSpect)
        maxB = plotSpect.max()
        minB = maxB-dBNoise
    else:
        if dBNoise is not None:
            maxB = 20*np.log10(plotSpect.max())
            minB = ((maxB-dBNoise)/20.0)**10
        else:
            maxB = plotSpect.max()
            minB = plotSpect.min()

    plotSpect[plotSpect < minB] = minB
    plt.imsave("temp.png", plotSpect[...,::-1,:], cmap=colormap, vmin=minB, vmax=maxB)
    im = plt.imread("temp.png")
    return im[:, :, :3]

def preprocess(wave):
    """
    Preprocess WAVE. WAVE is bandpassed and its spectrogram is calculated
    """
    if USE_GPU:
        filtered = bandpass_filter(wave.cpu(), SR, FMIN, FMAX)
    else:
        filtered = bandpass_filter(wave, SR, FMIN, FMAX)
    t, f, spec, spec_rms = spectrogram(s=filtered,
     sample_rate=SR,
     spec_sample_rate=SPEC_SR,
     freq_spacing=FREQ_SPACING,
     min_freq=FMIN,
     max_freq=FMAX,
     nstd=6,
     cmplx=False,
    )
    return color_spec(t, f, spec)


def get_spec_tensor(x):
    """
    Returns a tensor of batched spectrograms from X, a tensor of batched audio waves
    """
    spec_arr = None
    idx = 0
    if USE_GPU:
        spec_arr = torch.cuda.FloatTensor(preprocess(x[0]))
    else:
        spec_arr = torch.FloatTensor(preprocess(x[0]))
    spec_arr = torch.unsqueeze(spec_arr, 0)
    for i in range(1, len(x)):
        if USE_GPU:
            new_entry = torch.cuda.FloatTensor(preprocess(x[i]))
        else:
            new_entry = torch.FloatTensor(preprocess(x[i]))
        new_entry = torch.unsqueeze(new_entry, 0)
        spec_arr = torch.cat((spec_arr, new_entry), dim=0)
    spec_arr = spec_arr.permute(0, 3, 2, 1)
    return spec_arr


def shuffle(x, y):
    """
    Horizontally stacks x and y, shuffles the combined tensor along its rows and
    splits up the tensor into shuffled_x and shuffled_y
    """
    hstacked = torch.hstack((x, y.view(len(y), 1)))
    shuffled = hstacked[torch.randperm(hstacked.size()[0])]
    shuffled_x, shuffled_y = torch.split(shuffled, [shuffled.shape[1]-1, 1], dim=1)
    return shuffled_x, shuffled_y.view(shuffled_y.shape[0])


def validate_nums(num_voc, num_noise, tp, tn):
    if num_voc == 0:
        raise ValueError("At least one vocalization sample is required.")
    if num_noise == 0:
        raise ValueError("At least one noise sample is required.")
    print("Voc: {} / {} = {:.3f} Correct, Noise: {} / {} = {:.3f} Correct".format(tp, num_voc, tp/num_voc, tn, num_noise, tn/num_noise))


def plot_losses(epoch_ind, train_losses, val_losses, tps, tns, num_vocs, num_noises):
    fig, (ax1, ax2) = plt.subplots(2)
    train_arr = np.array(train_losses)
    val_arr = np.array(val_losses)
    normalized_train = (train_arr - np.min(train_arr)) / (np.max(train_arr) - np.min(train_arr))
    normalized_val = (val_arr - np.min(val_arr)) / (np.max(val_arr) - np.min(val_arr))
    ax1.plot(epoch_ind, normalized_train, label='Train Loss')
    ax1.plot(epoch_ind, normalized_val, label='Validation Loss')
    ax1.legend()
    for i, j in zip(epoch_ind, normalized_train):
        ax1.annotate(str(round(j, 2)), xy=(i, j))
    for i, j in zip(epoch_ind, normalized_val):
        ax1.annotate(str(round(j, 2)), xy=(i, j))
    tp_arr = np.array(tps) / np.array(num_vocs)
    tn_arr = np.array(tns) / np.array(num_noises)
    ax2.plot(epoch_ind, tp_arr, label='TP')
    ax2.plot(epoch_ind, tn_arr, label='TN')
    ax2.legend()
    for i, j in zip(epoch_ind, tp_arr):
        ax2.annotate(str(round(j, 2)), xy=(i, j))
    for i, j in zip(epoch_ind, tn_arr):
        ax2.annotate(str(round(j, 2)), xy=(i, j))
    plt.savefig(FIGURE)
    print("Saved figure to", FIGURE)


def train(model, train_data, voc_bs, noise_bs, epochs, optim, penalty):
    """
    Train the model
    """
    print("\nTraining...")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of trainable parameters:", total_params, "\n")
    # For each batch, load a fixed number of voc samples and a fixed number of noise samples.

    lowest_loss = float('inf')
    epoch_ind, train_losses, val_losses, tps, tns, num_vocs, num_noises = [], [], [], [], [], [], []
    for epoch in range(epochs):
        train_loss, val_loss = 0, 0
        # for the next line, 0 = undersample noise, 1 = oversample voc
        num_voc, num_noise, tp, tn = 0, 0, 0, 0
        num_batches = train_data.get_train_len(voc_bs, noise_bs)[0]
        for train_batch in tqdm(range(num_batches)):
            # x are wave arrays and y are labels
            x, y = train_data.get_train(voc_bs, noise_bs)
            # Shuffle x and y. We don't want the model to memorize that the first few samples are
            # always vocalizations
            shuffled_x, shuffled_y = shuffle(x, y)
            # Penalty vector. If the label is voc, weight = penalty. If the label is noise, weight = 1
            if USE_GPU:
                penalties = torch.cuda.FloatTensor([penalty if e == 1 else 1 for e in shuffled_y.tolist()])
            else:
                penalties = torch.FloatTensor([penalty if e == 1 else 1 for e in shuffled_y.tolist()])
            # Calculate spectrogram for each wave in the batch and stack them into a tensor
            # Shape: (batch size, 1 (channels), 100 (#cols, or time), 64 (#rows, or freq))
            spec_tensor = get_spec_tensor(shuffled_x)
            # The model generates its prediction
            logits = model(spec_tensor)
            # Calculate the loss. A custom loss function might be faster than creating an instance
            # of a PyTorch loss function every batch?
            loss_func = torch.nn.BCELoss(weight=penalties)
            loss = loss_func(logits, shuffled_y.float())
            if train_batch < num_batches * (1-VAL):
                train_loss += loss.item()
                # Clear old gradients
                optim.zero_grad()
                # Backpropagation
                loss.backward()
                # Step based on the new gradients
                optim.step()
            else:
                # cross validation
                with torch.no_grad():
                    val_loss += loss.item()
                    for i in range(len(logits)):
                        expected = shuffled_y[i]
                        predicted = logits[i]
                        if expected == 1:
                            if predicted >= TH:
                                tp += 1
                            num_voc += 1
                        else:
                            if predicted < TH:
                                tn += 1
                            num_noise += 1
        validate_nums(num_voc, num_noise, tp, tn)
        print("Epoch {} train loss: {:.3f} validation loss: {:.3f}".format(epoch, train_loss, val_loss))

        # Save the model if it has lower loss
        if val_loss < lowest_loss:
            print("Saving model to", os.path.join(PATH, BEST_MODEL_NAME))
            torch.save(model.state_dict(), BEST_MODEL_NAME)
            lowest_loss = val_loss
        epoch_ind.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        tps.append(tp)
        tns.append(tn)
        num_vocs.append(num_voc)
        num_noises.append(num_noise)
    plot_losses(epoch_ind, train_losses, val_losses, tps, tns, num_vocs, num_noises)


def test(model, test_dataset):
    """
    Test the model
    """
    print("\nTesting...\n")
    # Expected values, i.e. labels
    y_true = []
    # Predicted values, i.e. model logits
    y_score = []
    with torch.no_grad():
        num_voc, num_noise, tp, tn = 0, 0, 0, 0
        tuples = []
        has_printed = False
        for test_sample in tqdm(range(test_dataset.get_test_len() // 50)):
            # No need for shuffling or backward pass
            x, y, name, section = test_dataset.get_test()
            spec_tensor = get_spec_tensor(x)
            logits = model(spec_tensor)
            expected = y.item()
            predicted = logits.item()

            if expected == 1:
                if predicted >= TH:
                    tp += 1
                num_voc += 1
            elif expected == 0:
                if predicted < TH:
                    tn += 1
                num_noise += 1
            else:
                raise ValueError("label error, should be 1 or 0")
            tuples.append([name, section, predicted, expected])
        validate_nums(num_voc, num_noise, tp, tn)
        with open(PREDICTIONS, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerows(tuples)


if __name__ == "__main__":
    # Generate the colormap we want
    spec_colormap()
    if not TEST_ONLY:
        # Train dataset
        train_dataset = LowResDataset(is_test=False, use_gpu=USE_GPU)
        # Vocalization and noise batch sizes
        # Model
        if USE_GPU:
            model = eval(MODEL).cuda()
        else:
            model = eval(MODEL)
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        # Train the model
        train(model, train_dataset, VOC_BS, NOISE_BS, NUM_EPOCHS, optimizer, PENALTY)
    # Test dataset
    if not os.path.exists(TEST_PATH):
        raise ValueError("Test folder not found!\nCreate test folder at " + TEST_PATH + " and place test files inside.\nThe model you just trained has been saved.\nTo test the model, set TEST_ONLY to true and rerun this script.")
    test_dataset = LowResDataset(is_test=True, use_gpu=USE_GPU)
    # Load the best model
    if USE_GPU:
        best_model = eval(MODEL).cuda()
    else:
        best_model = eval(MODEL)
    best_model.load_state_dict(torch.load(os.path.join(PATH, BEST_MODEL_NAME)))
    # Test the best model
    best_model.eval()
    test(best_model, test_dataset)

