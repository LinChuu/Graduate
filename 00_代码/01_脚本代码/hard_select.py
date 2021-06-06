#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 14:26:07 2020

@author: chuang.lin
"""

import cv2
import numpy as np
import random
import os
import copy as cp

'''
作用：对图像进行随机挑选
输入：
    nlevel:表示要挑选出多少样本
    file_name:train_select_label.txt所在的文件夹，txt文件中存放图片名称
输出：
    挑选结果存放在_cg后缀的txt文件中
'''


file_name = '/home/chuang.lin/Desktop/graduation/dataset/IAI/'

def select_label(nlevel):
    
    with open(os.path.join(file_name, 'train_select_label.txt')) as f:
        reader = f.readlines()
        file_paths = [row[:-1] for row in reader]
        
    final_list = cp.deepcopy(file_paths)
    
    select_list = random.sample(final_list, int((len(final_list))*nlevel))
    print(len(select_list))
    slt_final_list = []
    for label in final_list:
        if label in select_list:
            tmp_label = label.split('.')[0] + '_cg.' + label.split('.')[1]
            slt_final_list.append(tmp_label)
        else:
            slt_final_list.append(label)
    
    with open(file_name+"train_cg"+str(nlevel).split('.')[1]+'.txt', 'w') as f:
        
        for img in slt_final_list:
            
            f.writelines(img+'\n')
    

def noise_generator(out_input, out_label, nlevel, nrange, nosiyType, ker_iter):
    
    
    ntype = nosiyType  # 0 for dilation, 1 for erosion
    npix = 0
    
    
    
    npix = nrange
    
    kernel = np.ones((nrange,nrange),np.uint8)
    if ntype:
        tmp_label = cv2.erode(out_label, kernel,iterations = ker_iter)
    else:
        tmp_label = cv2.dilate(out_label, kernel,iterations = ker_iter)

    tmp_label = out_label.copy()
       
    


if __name__ == '__main__':
    select_list = select_label(nlevel = 0.50)
    
    