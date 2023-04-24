import scipy.io
import numpy as np
import math
import os
import csv
import umap
from soundsig.sound import WavFile, BioSound

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

def open_files(paths):
    mat_objs = []
    for path in paths:
        for file in os.listdir(path):
            if file.endswith("200.mat"):
                mat_objs.append(load_mat(path + "/" + str(file)))
    return mat_objs


if __name__ == "__main__":
    root = os.getcwd()
    paths = [root + "/20200127"]
    mat_objs = open_files(paths)
    # print(len(mat_objs), mat_objs[0].keys(), mat_objs[1]['BioSoundCalls'])
    flag = True
    specs = mat_objs[1]['BioSoundCalls']
    # print(specs[0])
    funct = lambda row: len(row[0].spectro[0])
    # print(funct(specs[0]))
    # print(specs[:, 0])
    min_time = len(min(specs, key = funct)[0].spectro[0])
    # print(min_time)
    vocalizations_data = []
    for i, row in enumerate(specs):
        spectrogram = row[0].spectro
        start = len(spectrogram[0]) // 2 - min_time // 2
        end = start + min_time
        vocalizations_data.append(spectrogram[start:end, :].flatten())
        print(end)
        if flag:
            print(spectrogram[start:end, :])
            flag = False
        # specs[i][0] = spectrogram[:, ]
        # print(len(spectrogram[0]))
        # min_time = min(min_time, len(spectrogram[0]))
    vocalizations_data = vocalizations_data
    print(vocalizations_data.shape)
    # reducer = umap.UMAP()
    # reducer.fit(digits.data)
    # print(min_time)
    # embedding = reducer.transform(digits.data)
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
    # assert(np.all(embedding == reducer.embedding_))
    # print(embedding.shape)
    # train_zip, test_zip = create_zip(paths, -4)
    # print(train_zip)
