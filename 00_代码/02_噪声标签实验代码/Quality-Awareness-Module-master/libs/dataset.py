from __future__ import print_function, division
import os, sys
import numpy as np
import pickle, h5py, time, argparse, itertools, datetime

import torch
import torch.nn as nn
import torch.utils.data

import torchvision.transforms as transforms
from transforms3d.affines import compose
from transforms3d.euler import euler2mat
from skimage.transform import warp, AffineTransform
from scipy.ndimage.morphology import distance_transform_edt as distrans

class CTDataset(torch.utils.data.Dataset):
    # assume for test, no warping [hassle to warp it back..]
    def __init__(self, input, label=None, mode = 'train', noise_level=0.5, noise_range=[5, 13]):
        
        self.mode = mode
        self.input = input
        self.label = label
        self.nlevel = noise_level
        self.nrange = noise_range
        
    def __getitem__(self, index):
        index = index % self.input.shape[0]
        data_ret = self.input[index]
        label_ret = self.label[index]
        if self.mode == 'train':

            out_input, out_label, out_qua, noi_level = noise_generator(data_ret, label_ret,
                                                            nlevel=self.nlevel, nrange=self.nrange)
            weight = np.array([(out_label == 0).sum(),(out_label == 1).sum(),
                               (out_label == 2).sum(),(out_label == 3).sum()])
            weight = weight / (out_label == 0).sum()
            weight = torch.clamp(torch.Tensor(weight), min=0.001)
            weight = 1 / weight
            #print(weight)
            #weight = weight / torch.sum(weight)
            #print(weight)
            out_input = torch.Tensor(out_input)
            out_label = torch.Tensor(out_label)
            out_qua = torch.Tensor(out_qua)
            noi_level = torch.Tensor(noi_level)
            out_input = out_input.unsqueeze(0)
            return out_input, out_label, out_qua, noi_level, weight
        else:
            out_label = np.zeros(data_ret.shape)
            for i in range(label_ret.shape[0]):
                out_label[label_ret[i] == 1] = i+1
            out_input = torch.Tensor(data_ret)
            out_label = torch.Tensor(out_label)
            out_input = out_input.unsqueeze(0)
            return out_input, out_label


    def __len__(self): # number of possible position
        if self.mode == 'train':
            return self.input.shape[0] * 10000
        else:
            return self.input.shape[0]
    

def noise_generator(out_input, out_label, nlevel, nrange):
    input_size = np.shape(out_input)
    tmp_label = np.zeros(out_label.shape) #########################
    npix = 0
    if np.random.rand() < nlevel:
        ntype = np.random.randint(2)  # 0 for dilation, 1 for erosion
        npix = nrange[0] + (nrange[1] - nrange[0]) * np.random.rand()
        for t in range(tmp_label.shape[0]):
            if ntype:
                tmp_label[t] = (distrans(1-out_label[t]) < npix).astype(np.float)
            else:
                tmp_label[t] = 1-(distrans(out_label[t]) < npix).astype(np.float)
    else:
        tmp_label = out_label.copy()
    n_label = np.zeros([out_label.shape[1],out_label.shape[2]])
    q_input = np.zeros([4, out_label.shape[1],out_label.shape[2]])
    q_input[0] = out_input.copy()

    for i in range(1,4):
        n_label[tmp_label[i-1] == 1] = i
    for i in range(1,4):
        q_input[i] = (tmp_label[i-1] == 1).astype(np.float)
    return out_input, n_label, q_input, np.array([npix])

