#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:43:52 2019

@author: chuang.lin
"""

from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import scipy.misc 


from tqdm import tqdm
import random
import copy as cp

'''
作用：将标签图进行编码，比如标签原本是rgb,两个黑白两类，将黑白变成0和1
流程：首先遍历img_select_label.txt中的图像，得到所有图像集合，之后再对所有图像进行处理
输入：
    file_name：也就是img_select_label.txt所在的文件夹
    split_path：目标标签的文件夹
    save_path:处理之后的标签存放文件夹
    
'''

file_name = '/home/chuang.lin/Desktop/graduation/dataset/Road/'
split_path = '/home/chuang.lin/Desktop/graduation/dataset/Road/label/'
save_path = '/home/chuang.lin/Desktop/graduation/dataset/Road/encoder_label/'
#files = os.listdir(split_path)
with open(os.path.join(file_name, 'img_select_label.txt')) as f:
    reader = f.readlines()
    file_paths = [row[:-1] for row in reader]
files = cp.deepcopy(file_paths)
for m in tqdm(range(len(files))):
    img = Image.open(split_path+files[m])
    img = np.array(img)
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i][j] == 255):
                img[i][j] = 1 
    img = Image.fromarray(img)
    img.save(save_path+files[m])