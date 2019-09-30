from __future__ import print_function
import os
import json
import numpy as np

import torch
import torchvision.models as models
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class SignalDataset(Dataset):
    def __init__(self, name, data_path, version=None):
        super(SignalDataset, self).__init__()
        # using in several times
        self.key = ['vital', 'gt', 'demo']
        self.name = name
        self.data_path = data_path
        self.load_data(name, data_path)
        self.tensorize()

    def load_data(self, name, data_path):
        self.dic = {}
        for key in self.key:
            print(os.path.join(data_path, name+'_'+key+'.npy'))
            self.dic[key] = np.load(os.path.join(
                data_path, name+'_'+key+'.npy'))
            if key == 'gt':
                self.dic[key] = np.expand_dims(self.dic[key], axis=-1)

    # def split_data(self):
    #     self.new_dic ={}
    #     data = self.load_data(self.name, self.data_path)
    #     for v, g, d in data['vital']:
    #         np.concatenate(((v[:3, 2000:], v[3:, :1000]), 0))
    #         np.concatenate(((d[:3, 2000:], d[3:, :1000]), 0))
    def tensorize(self):
        for key in self.key:
            self.dic[key] = torch.from_numpy(self.dic[key])

    def __getitem__(self, idx):
        vital = self.dic['vital'][idx]
        gt = self.dic['gt'][idx]
        demo = self.dic['demo'][idx]
        return vital, gt, demo

    def __len__(self):
        return len(self.dic['gt'])
