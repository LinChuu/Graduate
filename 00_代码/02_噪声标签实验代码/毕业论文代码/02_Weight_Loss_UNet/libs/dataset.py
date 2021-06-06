from __future__ import print_function, division
import os
import numpy as np


import torch
import torch.nn as nn
import torch.utils.data

from scipy.ndimage.morphology import distance_transform_edt as distrans

from PIL import Image
import cv2
import collections
from torchvision.transforms import Compose, ToTensor

'''
作用:读取数据
输入:
    root：数据集存放路径
    mode:要读取哪个数据集数据，train,val,test
    noise_level:表示数据集样本噪声率，有百分之多少是噪声数据，如果为0，表示无噪声数据
    noise_range:表示产生噪声膨胀或者腐蚀的卷积核大小
    noiseType:表示噪声类型，0表示膨胀噪声，1表示腐蚀噪声
    ker_iter：表示膨胀和腐蚀操作迭代的次数，kernel=9的卷积核迭代1,2,3与kernel=9,17,25迭代1次的作用是相同的
流程：首先根据数据集路径及对应的txt文件，读取数据，存放在一个list中
     在训练过程中会遍历list，读取每一张图像，执行 __getitem__函数
     
'''


class CTDataset(torch.utils.data.Dataset):
    # assume for test, no warping [hassle to warp it back..]
    def __init__(self, root, mode = 'train', noise_level=0.5, noise_range=7, nosiyType=1, ker_iter=1):
        
        self.root = root
        self.mode = mode
        
        self.nosiyType = nosiyType
        self.nlevel = noise_level
        self.nrange = noise_range
        self.ker_iter = ker_iter
        
        
        self.files = collections.defaultdict(list)
        
        with open(os.path.join(root, 'train_select_label.txt')) as f:
            reader = f.readlines()
            self.train_list = [row[:-1] for row in reader]
            
        with open(os.path.join(root, 'val_select_label.txt')) as f:
            reader = f.readlines()
            self.val_list = [row[:-1] for row in reader]
            
        with open(os.path.join(root, 'test_select_label.txt')) as f:
            reader = f.readlines()
            self.test_list = [row[:-1] for row in reader]
            
        self.files['train'] = self.train_list
        self.files['val'] = self.val_list
        self.files['test'] = self.test_list
        
        self.files = self.files[self.mode]
        print('total train images: ', len(self.train_list))   
        print('total value images: ', len(self.val_list))   
        print('total testing images: ', len(self.test_list))   
        
    '''
    输入：图像的index
    流程：首先判断是否是训练数据，是的话要对图像进行噪声处理
         之后读取图像和标签，转化为数组
         执行noise_generator()函数对标签加噪声
         计算类别平衡系数
    输出：如果是训练数据
         out_input, out_label, out_qua, noi_level, weight
         out_input：图像数组
         out_label:噪声处理之后的标签
         out_qua：将out_input和out_label叠加，作为注意力子网络的输入
         noi_level：标记数组，如果标签被噪声处理，则标记为kernel值，
                    若是没有噪声处理即干净标签，则标记为0
    '''
        
    def __getitem__(self, index):

        if self.mode == 'train':
            
            img_name = self.files[index] 
            
            data_ret, label_ret = self.readfile(img_name,img_name)           

            out_input, out_label, out_qua, noi_level = noise_generator(data_ret, label_ret,
                                                            nlevel=self.nlevel, nrange=self.nrange, 
                                                            nosiyType=self.nosiyType, ker_iter=self.ker_iter)
            '''
            作用：计算类别平衡系数
            方式：前景类与背景类的比值
            **注意**：torch.clamp()中min,max作为超参数需调节
            '''
            weight = np.array([(out_label == 0).sum(),(out_label == 1).sum()
                               ])
            
            if (weight[0] != 0 and weight[1] != 0):
                weight = weight / (out_label == 0).sum()
            
                weight = torch.clamp(torch.Tensor(weight), min=0.1,max=10)
            
                weight = 1 / weight
            else:
                weight = torch.Tensor(np.array([1,1]))
            #print(weight)
            #weight = weight / torch.sum(weight)
            #print(weight)
            out_input = torch.Tensor(out_input)
            out_label = torch.Tensor(out_label)
            out_qua = torch.Tensor(out_qua)
            noi_level = torch.Tensor(noi_level)
            out_label = out_label
            return out_input, out_label, out_qua, noi_level, weight
        else:
            img_name = self.files[index] 
            data_ret, label_ret = self.readfile(img_name,img_name) 
            
            o_input = np.zeros([3, data_ret.shape[0], data_ret.shape[1]])
            for i in range(0,3):
                o_input[i] = data_ret[:,:,i].copy()

            out_input = torch.Tensor(o_input)
            out_label = torch.Tensor(label_ret)
            out_label = out_label
            return out_input, out_label,img_name


    def __len__(self): # number of possible position
        return len(self.files)
        
    def readfile(self, img_name,lbl_name):
        
        if self.mode == 'test':
            img_path = self.root + self.mode + '/' + img_name.split('_',1)[0]+'/'+img_name 
            lbl_path = self.root + self.mode + '_label'+'/' + img_name.split('_',1)[0]+'/'+img_name
        else: 
            img_path = self.root + self.mode + '/' + img_name.split('.')[0] + '.tif' 
            lbl_path = self.root + self.mode + '_label/' + lbl_name

        img = Image.open(img_path).convert('RGB')
        
        img = np.array(img)
        lbl = Image.open(lbl_path)
        lbl = np.array(lbl)
        
        return img, lbl
    

'''
作用：对标签进行噪声处理
输入：
    out_input:输入图像
    out_label:标签
    nlevel:样本噪声率
    nrange:膨胀腐蚀操作kernel大小
    noiseType:噪声类型
    ker_iter:迭代次数
输出：
     o_input:输入图像
     tmp_label：处理之后的标签
     q_input,：输入图像与处理之后标签的叠加
     np.array([npix])：是否被噪声操作的标记
'''    

def noise_generator(out_input, out_label, nlevel, nrange, nosiyType, ker_iter):
    input_size = np.shape(out_input)
    tmp_label = np.zeros(out_label.shape) #########################
    ntype = nosiyType  # 0 for dilation, 1 for erosion
    npix = 0
    
    
    if np.random.rand() < nlevel:
        npix = nrange
        
        kernel = np.ones((nrange,nrange),np.uint8)
        if ntype:
            tmp_label = cv2.erode(out_label, kernel,iterations = ker_iter)
        else:
            tmp_label = cv2.dilate(out_label, kernel,iterations = ker_iter)
    else:
        tmp_label = out_label.copy()
       
    q_input = np.zeros([4,out_label.shape[0],out_label.shape[1]])
    q_input[0] = tmp_label.copy()
    
    
    o_input = np.zeros([3,out_input.shape[0],out_input.shape[1]])

    for i in range(0,3):
        o_input[i] = out_input[:,:,i].copy()
    for i in range(1,4):
        q_input[i] = out_input[:,:,i-1].copy()
    return o_input, tmp_label, q_input, np.array([npix])

