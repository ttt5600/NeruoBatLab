from torch.utils.data import Dataset, Sampler
import os
import scipy.io
import numpy as np
import csv
from random import randrange
import torch
import math


class LowResDataset(Dataset):
    """
    Wave: array representing soundwave
    Label: 1 if vocalization is present, 0 if noise
    """

    def __init__(self, is_test, use_gpu):
        if is_test:
            vocs_csv = "test_vocs.csv"
            noises_csv = "test_noises.csv"
            self.path = "./data/test/"
            self.is_test = True
        else:
            vocs_csv = "train_vocs.csv"
            noises_csv = "train_noises.csv"
            self.path = "./data/train/"
            self.is_test = False
        with open(vocs_csv, newline='') as v:
            voc_reader = csv.reader(v)
            vocs = list(voc_reader)
            self.vocs = vocs
        with open(noises_csv, newline='') as n:
            noise_reader = csv.reader(n)
            noises = list(noise_reader)
            self.noises = noises
        if is_test:
            self.test_len = len(vocs) + len(noises)
            self.test_data = []
            self.test_data.extend(self.vocs)
            self.test_data.extend(self.noises)
            self.curr_index = 0
        else:
            self.noises_len = len(noises)
            self.vocs_len = len(vocs)
            self.avail_voc_indices = [i for i in range(self.vocs_len)]
            self.avail_voc_range = self.vocs_len
            self.avail_noise_indices = [i for i in range(self.noises_len)]
            self.avail_noise_range = self.noises_len
        self.use_gpu = use_gpu

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None

    def get_train_len(self, voc_bs, noise_bs):
        voc_len = math.floor(self.vocs_len / voc_bs)
        noise_len = math.floor(self.noises_len / noise_bs)
        return voc_len, noise_len

    def get_test_len(self):
        return self.test_len

    def get_voc(self):
        if self.avail_voc_range == 0:
            self.avail_voc_range = self.vocs_len
            self.avail_voc_indices = [i for i in range(self.vocs_len)]
        voc_ind = randrange(self.avail_voc_range)
        ind = self.avail_voc_indices.pop(voc_ind)
        self.avail_voc_range -= 1
        return self.vocs[ind]

    def get_noise(self):
        if self.avail_noise_range == 0:
            self.avail_noise_range = self.noises_len
            self.avail_noise_indices = [i for i in range(self.noises_len)]
        noises_ind = randrange(self.avail_noise_range)
        ind = self.avail_noise_indices.pop(noises_ind)
        self.avail_noise_range -= 1
        return self.noises[ind]

    def get_train(self, num_vocs, num_noises):
        indices = []
        for _ in range(num_vocs):
            indices.append(self.get_voc())
        for _ in range(num_noises):
            indices.append(self.get_noise())
        result_list = []
        label_list = []
        for ind in indices:
            name = os.path.join(self.path, ind[0])
            loaded = scipy.io.loadmat(name)
            wave = loaded["waves"][int(ind[1])]
            label = loaded["labels"][0][int(ind[1])]
            result_list.append(wave)
            label_list.append(label)
        if not self.use_gpu:
            return torch.FloatTensor(np.array(result_list)), torch.FloatTensor(np.array(label_list))
        else:
            return torch.cuda.FloatTensor(np.array(result_list)), torch.cuda.FloatTensor(np.array(label_list))

    def get_test(self):
        ind = self.test_data[self.curr_index]
        name = os.path.join(self.path, ind[0])
        # added for analysis
        section = ind[1]
        # section = name[: -4] + '_' + ind[1] +'.mat'
        loaded = scipy.io.loadmat(name)
        wave = loaded["waves"][int(ind[1])]
        label = loaded["labels"][0][int(ind[1])]
        self.curr_index += 1
        if not self.use_gpu:
            return torch.FloatTensor([wave]), torch.FloatTensor([label]), name, section
        else:
            return torch.cuda.FloatTensor([wave]), torch.cuda.FloatTensor([label]), name, section

