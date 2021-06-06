#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:43:33 2020

@author: chuang.lin
"""
import torch
import torch.utils.data as data
import cv2
import PIL
import numpy as np
import csv
import os
import torchvision.transforms as tr
import copy as cp
import random

file_name = '/home/chuang.lin/Desktop/graduation/dataset/Road/'

with open(os.path.join(file_name, 'img_select_label.txt')) as f:
        reader = f.readlines()
        file_paths = [row[:-1] for row in reader]
        
final_list = cp.deepcopy(file_paths)
fileNum = len(final_list)
rate = 0.2
pickNum = int(rate * fileNum)
print(pickNum)
sample = random.sample(final_list, pickNum)

with open(os.path.join(file_name,'test.txt'), 'w') as f:
    for file_path in sample:
        f.writelines(file_path.split('.')[0]+'.tiff'+'\n')
