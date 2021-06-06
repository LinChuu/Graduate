#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 05:01:25 2019

@author: chuang.lin
"""

'''
作用：对数据集进行扩充，首先将大图裁剪成小图，之后根据mode参数判断是否对小图进行变换，
    mode=original表示不变换只切图，mode=augment表示进行变换，旋转，镜面对称，颜色变化，加噪声等、
'''

from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import random
import os
import numpy as np
import scipy.misc as sc
import time
from tqdm import tqdm
import tifffile as tiff




img_file_name = '/home/chuang.lin/Desktop/graduation/dataset/original/img_label/'

nrgb_img_file_save = '/home/chuang.lin/Desktop/graduation/dataset/original/nrgb/'
rgb_img_file_save = '/home/chuang.lin/Desktop/graduation/dataset/original/rgb/'
lbl_file_save = '/home/chuang.lin/Desktop/graduation/dataset/original/label/'

img_w = 256
img_h = 256

'''
作用：遍历文件夹的所有文件
输入：
    file_dir:目标文件夹
输出：
    文件名称集合
'''

def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            img_name = os.path.split(file)[1]
            L.append(img_name)
    return L


image_sets = file_name(img_file_name)

'''
作用：对图像加入白噪声
输入：目标图像
输出：返回处理之后的图像
'''

def add_noise(img):
    drawObject = ImageDraw.Draw(img)
    for i in range(250):
        temp_x = np.random.randint(0, img.size[0])
        temp_y = np.random.randint(0, img.size[1])
        drawObject.point((temp_x, temp_y), fill="white")
    return img

'''
作用：对图像进行颜色变换
输入：目标图像
输出：返回处理之后的图像
'''

def random_color(img):
    img = ImageEnhance.Color(img)
    img = img.enhance(2)
    return img

'''
作用：对图像进行数据增强，旋转，对称，颜色变换，白噪声
输入：目标图像， 目标图像标签
输出：返回处理之后的图像及标签
'''

def data_augment(src_roi, label_roi):
    if np.random.random() < 0.25:
        src_roi = src_roi.rotate(90)
        label_roi = label_roi.rotate(90)
    if np.random.random() < 0.25:
        src_roi = src_roi.rotate(180)
        label_roi = label_roi.rotate(180)
    if np.random.random() < 0.25:
        src_roi = src_roi.rotate(270)
        label_roi = label_roi.rotate(270)

    if np.random.random() < 0.25:
        src_roi = src_roi.transpose(Image.FLIP_LEFT_RIGHT)
        label_roi = label_roi.transpose(Image.FLIP_LEFT_RIGHT)

    if np.random.random() < 0.25:
        src_roi = src_roi.transpose(Image.FLIP_TOP_BOTTOM)
        label_roi = label_roi.transpose(Image.FLIP_TOP_BOTTOM)

    if np.random.random() < 0.25:
        src_roi = src_roi.filter(ImageFilter.GaussianBlur)

    if np.random.random() < 0.25:
        src_roi = random_color(src_roi)

    if np.random.random() < 0.2:
        src_roi = add_noise(src_roi)
    return src_roi, label_roi

'''
作用：对文件夹中的所有大图进行随机裁剪，之后根据mode模式进行数据变换扩充
输入：
    image_num:设定数据集的数据量
    mode:original只进行裁剪不变换，augment进行裁剪同时对小图变换
输出：返回处理之后的图像及标签
'''


def creat_dataset(image_num=2000, mode='original'):
    print('creating dataset...')
    image_each = image_num / len(image_sets)
    g_count = 0
    for i in tqdm(range(len(image_sets))):
        count = 0
        
        '''
        作用：这段代码是读取多波段图像中的几个波段生产nrgb和rgb图像，
        并将标签变化成0和1
        src_img = tiff.imread(img_file_name + image_sets[i])
        src_img = np.uint8(src_img)
        
        nrgb_img = src_img[:,:,0:4][:,:,(3,2,1,0)]
        
        rgb_img = src_img[:,:,0:3][:,:,(2,1,0)]
        
        nrgb_img = Image.fromarray(nrgb_img)
        nrgb_img.save(nrgb_img_file_save + image_sets[i])
        
        rgb_img = Image.fromarray(rgb_img)
        rgb_img.save(rgb_img_file_save + image_sets[i])
        
        label_img = src_img[:,:,-1]
        
        building_num = image_sets[i].split('.')[0].split('_')[-1]
               
        time_start = time.time()
        print('begin')
        for m in range(label_img.shape[0]):
            for n in range(label_img.shape[1]):
                if label_img[m][n] != int(building_num):
                    label_img[m][n] = 0
                else :
                    label_img[m][n] = 1

        time_end = time.time()
        print(str((time_end-time_start)/60) + 'min')
        
        
        label_img = Image.fromarray(label_img)
        label_img.save(lbl_file_save + image_sets[i].split('_')[0] + '_lbl.tif')
             
        
        
        #matplotlib.image.imsave(rgb_img_file_save + '1.tif', rgb_img)
        
        
        
        for i in range(label_img.shape[0]):
            for j in range(label_img.shape[1]):
                if label_img[i][j] != 6:
                    label_img[i][j] = 0
                    
        '''
        #对图像进行裁剪并做变换
        while count < image_each:
            width1 = random.randint(0, src_img.size[0] - img_w)
            height1 = random.randint(0, src_img.size[1] - img_h)
            width2 = width1 + img_w
            height2 = height1 + img_h

            src_roi = src_img.crop((width1, height1, width2, height2))
            label_roi = label_img.crop((width1, height1, width2, height2))
            
            
            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)
            src_roi.save(img_file_save + '%d.tif' % g_count)
            label_roi.save(lbl_file_save + '%d.png' % g_count)
            count += 1
            g_count += 1
        
        
        
        
        #对标签进行裁剪并做变换
        label_img = Image.open(lbl_file_name + image_sets[i].split('.')[0] + '.png')
        while count < image_each:
            width1 = random.randint(0, src_img.size[0] - img_w)
            height1 = random.randint(0, src_img.size[1] - img_h)
            width2 = width1 + img_w
            height2 = height1 + img_h

            src_roi = src_img.crop((width1, height1, width2, height2))
            label_roi = label_img.crop((width1, height1, width2, height2))

            if mode == 'augment':
                src_roi, label_roi = data_augment(src_roi, label_roi)
            src_roi.save(img_file_save + '%d.tif' % g_count)
            label_roi.save(lbl_file_save + '%d.png' % g_count)
            count += 1
            g_count += 1
        

if __name__ == '__main__':
    creat_dataset(mode='original')