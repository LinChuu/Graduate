#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:50:25 2019

@author: chuang.lin
"""


import cv2
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
import scipy.misc
'''
cla_list = ['cheng_shi_zhu_zhai','cun_zhen_zhu_zhai',
            'he_liu','hu_po','keng_tang']

cla_list = ['shui_tian','shui_jiao_di','jiao_tong_yun_shu','han_geng_di','yuan_di','qiao_mu_lin_di','guan_mu_lin_di','tian_ran_cao_di',
            'ren_gong_cao_di','gong_ye_yong_di','cheng_shi_zhu_zhai','cun_zhen_zhu_zhai',
            'he_liu','hu_po','keng_tang']

test_list = ['1_test','2_test','3_test','4_test','5_test','6_test','7_test','8_test','9_test','10_test']
'''


def merge_picture(merge_file_path,cla_name,num_rows,num_cols,test_path):
    filename = file_name(merge_file_path,cla_name, '.png')
    #print(filename[0])
    shape = cv2.imread(filename[0]).shape
    
    rows = shape[0]
    cols = shape[1]
    #channels = shape[2]
    #dst = np.zeros((num_rows*500, num_cols*500, channels),np.uint8)
    dst = np.zeros((num_rows*500, num_cols*500,3),np.uint8)
    cols_stride = 250
    rows_stride = 250
    for i in range(len(filename)):
        img = scipy.misc.imread(filename[i])
        rows_th = int(filename[i].split('_')[-2])
        cols_th = int(filename[i].split('_')[-1].split('.')[0])
        
        #roi = img[0:rows, 0:cols, :]
        #print(rows_th, cols_th)
        dst[rows_th*rows_stride:rows_th*rows_stride+rows,cols_th*cols_stride:cols_th*cols_stride+cols,:] = img[:]
    #test_file = test_path + cla_th + '_test/'
    if not os.path.exists(test_path):
            os.makedirs(test_path)
    dst = Image.fromarray(dst)
    dst.save(test_path + filename[0].split('/')[-1].split('_')[0] + '_merge.tif')


def file_name(root_path, cla_name,picturetype):
    filename = []
    for root, dirs, files in os.walk(root_path+cla_name+'/'):
        for file in files:
            if os.path.splitext(file)[1] == picturetype:
                filename.append(os.path.join(root,file))
    return filename




if __name__ == '__main__':
    
    test_path = '/home/chuang.lin/Desktop/graduation/code/dilate/QAM/result53%(12000)/saved_test_images/'   
    #merge_file_path = '/home/chuang.lin/Desktop/graduation/code/erosion/6000/method/APD_kernel9_50%/result/saved_test_images/'  
    cla_list = os.listdir(test_path)
    
    for i in tqdm(range(len(cla_list))):
        if os.path.isdir(os.path.join(test_path,cla_list[i])):
            num_rows = 10
            num_cols = 10
            merge_picture(test_path,cla_list[i],num_rows,num_cols,test_path)