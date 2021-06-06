#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 04:50:38 2019

@author: chuang.lin
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import scipy.misc as m

'''
作用：对标签进行译码，也就是赋予颜色，原本是像素值为0,1，可以有颜色的rgb图像
输入：
    lbl_file_name:要处理的标签存放文件夹
    lbl_file_save:处理后的标签存放文件夹
'''
                                                                               # xiugai  1
lbl_file_name = '/nas/chuang.lin/road/dataset/gf1/test_label/'
lbl_file_save = '/nas/chuang.lin/road/dataset/gf1/test_label_vis/'

'''
作用：遍历文件夹文件
'''
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            img_name = os.path.split(file)[1]
            
            L.append(img_name)
    return L

image_sets = file_name(lbl_file_name)
print(image_sets)

'''
作用：相应像素点对应的rgb颜色
'''

def get_labels():
    '''
    return np.asarray([[0,0,0],[0,200,0],[150,250,0],[150,200,150],
                       [200,0,200],[150,0,250],[150,150,250],
                       [250,200,0],[200,200,0],
                       [200,0,0],[250,0,150],[200,150,150],[250,150,150],
                       [0,0,200],[0,150,200],[0,200,250]])
    '''
    return np.asarray([[0,0,0],[255,255,255]])                  #xiugai 2

'''
作用：进行颜色转换
输入：
    temp:输入图像
    plot:是否画图
输出：
    返回变换后的rgb图像
'''

def decode_segmap(temp, plot=False):
        label_colours = get_labels()
        #print(label_colours)
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(2):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        #print(rgb)
        
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
  
'''
作用：主函数，首先获取目标文件夹的所有文件，之后对每一个文件进行颜色变换
'''
def setup():

    for i in tqdm(range(len(image_sets))):
        
        lbl_path = lbl_file_name +'/'+image_sets[i]
        lbl = decode_segmap(m.imread(lbl_path))
        '''
        if not os.path.exists(lbl_file_save + image_sets[i].split('_')[0]):
            os.makedirs(lbl_file_save + image_sets[i].split('_')[0])
        '''
        lbl_decode_name = lbl_file_save +image_sets[i]
        print(lbl_decode_name)
        m.imsave(lbl_decode_name,lbl)
        
        
if __name__ == '__main__':
    setup()