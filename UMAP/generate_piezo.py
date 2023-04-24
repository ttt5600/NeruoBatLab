import scipy.io
import numpy as np
import math
import os
import csv
from soundsig.sound import WavFile, BioSound

WAVE_KEY = "Piezo_wave"
FS_KEY = "Piezo_FS"
START_KEY = "IndVocStartPiezo"
STOP_KEY = "IndVocStopPiezo"
SR = 50000 # Sampling rate in Hz. Actual logger sampling rates can vary by +-2 Hz
TEST_PATH = "./data/test/"
TRAIN_PATH = "./data/train/"
WIN = 100 # Window length in ms
INPUT_FOLDER = "./data/VocExtracts"
FMIN = 100 # Min frequency in Hz
FMAX = 5000 # Max frequency in Hz
FREQ_SPACING = 100 # Spectogram param, 73 for CRNN, 100 (default) for Protonet & ResNet, 146 for GoogLeNet

# Accessor functions
def load_mat(filename):
    """
    This function should be called instead of scipy.io.loadmat
    as it can recover python dictionaries from mat files.
    """

    def _check_keys(d):
        """
        Checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        """
        for key in d:
            if isinstance(d[key], scipy.io.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        """
        A recursive function which constructs from matobjects nested dictionaries
        """
        nested_d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
                nested_d[strg] = _todict(elem)
            else:
                nested_d[strg] = elem
        return nested_d

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def get_num_sections(data_file, logger_id):
    """
    Gets the number of sections in piezo LOGGER_ID from ndarray DATA_FILE.
    """
    return len(data_file[WAVE_KEY][logger_id])


def get_piezo_wave_at_section(data_file, logger_id, section):
    """
    Gets Piezo_wave of piezo LOGGER_ID from ndarray DATA_FILE at section SECTION.
    """
    return data_file[WAVE_KEY][logger_id][section]


def get_label_at_section(label_file, key, piezo_ind, section):
    """
    Gets labels of piezo indexed PIEZO_IND from ndarray LABEL_FILE at section SECTION.
    KEY should be in LABEL_KEYS.
    """
    result = label_file[key][section]
    if result.size == 0:
        return result
    else:
        return result[piezo_ind]


def get_label_start_stop_at_section(label_file, piezo_ind, section):
    """
    Gets the start and stop indices of the labels of the piezo indexed PIEZO_IND
    from LABEL_FILE.
    """
    start_labels = get_label_at_section(label_file, START_KEY, piezo_ind, section)
    if type(start_labels) == int:
        return np.array([[start_labels, get_label_at_section(label_file, STOP_KEY, piezo_ind, section)]])
    s = start_labels.size
    if s == 0:
        return start_labels
    else:
        stop_labels = get_label_at_section(label_file, STOP_KEY, piezo_ind, section)
        result = np.zeros((s, 2))
        for i in range(s):
            result[i][0] = start_labels[i]
            result[i][1] = stop_labels[i]
        return result


def get_mapping(data_file):
    """
    Gets two-way mapping between piezo indices and logger ID's for DATA_FILE.
    """
    result = {}
    i = 0
    for key in data_file["Piezo_wave"]:
        result[i] = key
        result[key] = i
        i += 1
    return result


"""
Each file has 10 piezos, each piezo has 100 sections, each section has ~10 100ms windows.
These functions break VocExtractData.mat files into these small windows, and stores each
window with its corresponding labels in a new .mat file.
"""

def get_hard_bounds(labels, prev_end, new_end):
    """
    Given section LABELS and window from PREV_END to NEW_END, determines how labels should
    be split. High resolution
    """
    win_label = []
    for label in labels:
        if label[0] > prev_end and label[1] < new_end:
            win_label.append([label[0] - prev_end, label[1] - prev_end])
        elif label[0] < prev_end and label[1] < new_end and label[1] > prev_end:
            win_label.append([0, label[1] - prev_end])
        elif label[0] > prev_end and label[1] > new_end and label[0] < new_end:
            win_label.append([label[0] - prev_end, new_end - prev_end])
        elif label[0] < prev_end and label[1] > new_end:
            win_label.append([0, new_end - prev_end])
    return win_label


def get_soft_bounds(labels, prev_end, new_end):
    """
    Returns 1 if there are labels in the window, else returns 0. Low resolution
    """
    for label in labels:
        if label[0] > prev_end and label[1] < new_end:
            return 1
        elif label[0] < prev_end and label[1] < new_end and label[1] > prev_end:
            return 1
        elif label[0] > prev_end and label[1] > new_end and label[0] < new_end:
            return 1
        elif label[0] < prev_end and label[1] > new_end:
            return 1
    return 0


def write_mat_piezo(data_name, data_file, label_file, logger_id, piezo_ind, is_test):
    """
    Processes a single section in a .mat file. Wrapper for write_mat_section
    """
    piezo_vocs = []
    piezo_noises = []
    for i in range(get_num_sections(data_file, logger_id)):
        wave = get_piezo_wave_at_section(data_file, logger_id, i)
        if type(wave) != np.ndarray or np.isnan(wave).any():
            print("Malformed input at {0}.mat, {1} section {2}. Skipping".format(data_name, logger_id, i))
            continue
        labels = get_label_start_stop_at_section(label_file, piezo_ind, i)
        window_len = SR*WIN/1000
        prev_end = 0
        num_windows = np.floor(len(wave)/window_len).astype(int)
        section_wins = []
        section_labels = []
        name = "{0}_{1}_{2}.mat".format(data_name, logger_id, i)
        for j in range(num_windows):
            new_end = math.floor(prev_end+window_len)
            win = wave[prev_end:new_end]
            win_label = get_soft_bounds(labels, prev_end, new_end)
            section_wins.append(win)
            if win_label == 1:
                piezo_vocs.append([name, j])
            else:
                piezo_noises.append([name, j])
            section_labels.append(win_label)
            prev_end = new_end
        # Pad the tail case
        pad_len = int(window_len - (len(wave) - prev_end))
        last = np.pad(wave[prev_end:], (0, int(pad_len)), 'constant')
        last_label = get_soft_bounds(labels, prev_end, new_end)
        if last_label == 1:
            piezo_vocs.append([name, num_windows])
        else:
            piezo_noises.append([name, num_windows])
        section_wins.append(last)
        section_labels.append(last_label)
        data = {"waves":section_wins, "labels":section_labels}
        if is_test:
            full_name = os.path.join(TEST_PATH, name)
        else:
            full_name = os.path.join(TRAIN_PATH, name)
        scipy.io.savemat(full_name, data, do_compression=True)
    return piezo_vocs, piezo_noises


def write_mat_entire(data_name, data_file, label_file, mapping, is_test):
    """
    Proccesses an entire .mat file. Wrapper for write_mat_piezo
    """
    mat_vocs = []
    mat_noises = []
    for key, value in mapping.items():
        if type(key) == int:
            vocs, noises = write_mat_piezo(data_name, data_file, label_file, mapping[key], key, is_test)
            mat_vocs.extend(vocs)
            mat_noises.extend(noises)
    return mat_vocs, mat_noises


def create_zip(paths, idx):
    data_buffer, label_buffer = [], []
    train_data_files, train_label_files = [], []
    test_data_files, test_label_files = [], []
    for path in paths:
        for file in os.listdir(path):
            if file.endswith("200.mat"):
                label_buffer.append(os.path.join(path, file))
        label_buffer = sorted(label_buffer)
        data_buffer = [e[:-8] + ".mat" for e in label_buffer]
        for e in data_buffer:
            if not os.path.exists(e):
                raise ValueError("file", e, "not found, but its label exists!")
        train_data_files.extend(data_buffer[:idx])
        test_data_files.extend(data_buffer[idx:])
        train_label_files.extend(label_buffer[:idx])
        test_label_files.extend(label_buffer[idx:])
        data_buffer, label_buffer = [], []
    train_zip = list(zip(train_data_files, train_label_files))
    test_zip = list(zip(test_data_files, test_label_files))
    return train_zip, test_zip


def process_files(files_zip, is_test):
    all_vocs = []
    all_noises = []
    for i in range(len(files_zip)):
        # Load files and their sample rates
        sample = files_zip[i]
        data_file = load_mat(sample[0])
        label_file = load_mat(sample[1])
        if START_KEY not in label_file:
            print("{0} doesn't have {1}. Skipping".format(sample[1], START_KEY))
            continue
        # Create two-way mapping between indices and logger ID's
        mapping = get_mapping(data_file)
        # Short name for saving to new format
        chars_to_keep = "0123456789_"
        short_name = ''.join(c for c in sample[0] if c in chars_to_keep)
        # Write to disk
        print("Generating", i, "out of", len(files_zip))
        mat_vocs, mat_noises = write_mat_entire(short_name, data_file, label_file, mapping, is_test)
        all_vocs.extend(mat_vocs)
        all_noises.extend(mat_noises)
    if is_test:
        vocs_csv = "test_vocs.csv"
        noises_csv = "test_noises.csv"
    else:
        vocs_csv = "train_vocs.csv"
        noises_csv = "train_noises.csv"
    with open(vocs_csv, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows(all_vocs)
    with open(noises_csv, 'w', newline='') as f:
        write = csv.writer(f)
        write.writerows(all_noises)

if __name__ == "__main__":
#    # The files that we want to read
#    if not os.path.exists(TEST_PATH):
#        os.makedirs(TEST_PATH)
#    if not os.path.exists(TRAIN_PATH):
#        os.mkdir(TRAIN_PATH)
#    data_filename = "200123_0958_VocExtractData1.mat"
#    label_filename = "200123_0958_VocExtractData1_200.mat"
#    data_file = load_mat(data_filename)
#    label_file = load_mat(label_filename)
#    # Create two-way mapping between indices and logger ID's
#    mapping = get_mapping(data_file)
#    # Short name for saving to new format
#    chars_to_keep = "0123456789_"
#    short_name = ''.join(c for c in data_filename if c in chars_to_keep)
#    # Write to disk
#    print("Generating")
#    write_mat_entire(short_name, data_file, label_file, mapping, is_test=False)
#    write_mat_piezo(short_name, data_file, label_file, "Logger3", 3)
#    write_mat_section(short_name, data_file, label_file, "Logger3", 3, 13)
    # paths = ["Z:/users/JulieE/DeafSalineGroup151/20200123/audiologgers", "Z:/users/JulieE/DeafSalineGroup151/20200124/audiologgers", "Z:/users/JulieE/DeafSalineGroup151/20200127/audiologgers"]
    root = os.getcwd()
    paths = [root + "/20190207"]
    idx = -4
    train_zip, test_zip = create_zip(paths, idx)
    if not os.path.exists(TRAIN_PATH):
        os.makedirs(TRAIN_PATH)
    if not os.path.exists(TEST_PATH):
        os.mkdir(TEST_PATH)
    process_files(train_zip, False)
    process_files(test_zip, True)
