#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 04:53:06 2019

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

'''
作用：筛选掉背景类占大部分的标签图
'''


file_name = '/nas/chuang.lin/xiong/dataset/'


'''
作用：将文件夹下的文件写入txt中
输入：
    file_name：文件夹路径
    cla_list:文件夹的拼接字符
输出：
    文件名称集合
'''
def write_to_txt(cla_list='train'):
    with open(file_name+cla_list+ '.txt', 'w') as f:
        files = os.listdir(file_name + cla_list)
        for img in files:
            if img.split('.')[-1] == 'tif':
                f.writelines(img+'\n')

'''
作用：筛选背景类占大部分的标签
流程：首先打开txt文件获取标签集合，之后判断标签中背景类像素数与总像素数的比值，
     超过一定阈值就认为是坏的标签
输入：文件夹路径
输出：bad_sample_list坏标签的集合
     final_list最终标签集合
'''

def find_bad_sample(cla_list='val'):
    
    file_paths = None
    bad_sample_list = []
    
    
    
    with open(os.path.join(file_name, 'train.txt')) as f:
        reader = f.readlines()
        file_paths = [row[:-1] for row in reader]
        
    final_list = cp.deepcopy(file_paths)
    
    
   
    for i in file_paths:
        #test_path = file_name + '/test_label/' + i.split('_',1)[0] +'/'+ i
        print(i)
        if i.split('.')[-1] == 'png':
                
            train_path = file_name + 'train_label_vis' + '/' + i.split('.')[0] + '.png'
            
            ratio = judge_bad_sample(train_path)
            
        if ratio> 0.95:
            bad_sample_list.append(i)
            final_list.remove(i)

    print(len(final_list))
    print(len(bad_sample_list))
    
    with open(os.path.join(file_name,'img_select_label.txt'), 'w') as f:
        for file_path in final_list:
            
            f.writelines(file_path.split('.')[0]+'.png'+'\n')
            
    with open(os.path.join(file_name,'img_bad_label.txt'), 'w') as f:
        for file_path in bad_sample_list:
            f.writelines(file_path.split('.')[0]+'.png'+'\n')
            
    
'''
作用：判断图像背景类的像素数与总像素数的比值
输入：图像
输出：比值
'''
        

def judge_bad_sample(img_name):
    #print(img_name)
    Img = cv2.imread(img_name)
    '''
    
    Gray = tr.Compose([
                       tr.ToPILImage(),
                       #tr.Grayscale(),
                       tr.ToTensor()
                       ])

    Img_Gray = Gray(Img)
    ratio = float(torch.eq(Img_Gray, 0.0).sum())/(256.0*256.0)
    '''
    im_gray = cv2.cvtColor(Img, cv2.COLOR_BGR2GRAY) 
    retval, dst = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
    #背景类占总像素点数的比值
    ratio = float((dst == 0.0).sum())/(256.0*256.0)
    #print(ratio) 
    #print(img_name, float(torch.eq(Img_Gray, 0.0).sum() + torch.eq(Img_Gray, 1.0).sum())/10000.0)
    return ratio
'''
def judge_bad_sample_test(img_name):
    Img = cv2.imread(os.path.join('/home/zxw/2019BaiduXJTU/data/', img_name))
    Gray = tr.Compose([
                       tr.ToPILImage(),
                       tr.Grayscale(),
                       tr.ToTensor()
                       ])

    Img_Gray = Gray(Img)
    ratio = float(torch.eq(Img_Gray, 0.0).sum())/10000.0

    #print(ratio) 
    #print(img_name, float(torch.eq(Img_Gray, 0.0).sum() + torch.eq(Img_Gray, 1.0).sum())/10000.0)
    return ratio
'''
if __name__ == '__main__':
    write_to_txt(cla_list='test')
    #find_bad_sample(cla_list='train')












