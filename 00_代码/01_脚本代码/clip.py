#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:10:19 2019

@author: chuang.lin
"""

import cv2
import os
from scipy import misc
from PIL import Image
from libtiff import TIFF

'''
作用：对一张大的图像切成小图
输入：
    clip_file_path:裁剪图像的文件夹
    filename:要裁剪图像的名称
    rows, cols:小图的大小
结果：
    将每一张大图切的小图保存在一个文件夹
'''


def clip_one_picture(clip_file_path, filename, rows, cols):
    img = Image.open(clip_file_path+filename)
    #img = tif.read_image()
    print(img)
    sum_rows = img.size[0]   
    sum_cols = img.size[1]
    print(sum_rows,sum_cols)
    save_path = clip_file_path + filename.split('.')[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cols_stride = 250
    rows_stride = 250
    for i in range(int(sum_cols/cols_stride)-1):
        for j in range(int(sum_rows/rows_stride)-1):
            img_path = save_path+'/'+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+'.png'
            '''
            image = img[j*rows:(j+1)*rows,i*cols:(i+1)*cols,:]
            
            image = Image.fromarray(image)
            image.save(img_path)
            '''
            img_roi = img.crop((i*cols_stride,j*rows_stride,i*cols_stride+cols,j*rows_stride+rows))
            img_roi.save(img_path)
            #print(str(j)+'_'+str(i)+'.png')
            






if __name__ == '__main__':
    clip_file_path = '/home/chuang.lin/Desktop/graduation/dataset/IAI/test_encoder_label/'
    file = os.listdir(clip_file_path)
    for filename in file:
        #filename = 'austin5.tif'
        print(filename)
        rows = 500
        cols = 500
        clip_one_picture(clip_file_path, filename, rows, cols)
    