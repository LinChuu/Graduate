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

class CTDataset(torch.utils.data.Dataset):
    # assume for test, no warping [hassle to warp it back..]
    def __init__(self, root, mode = 'train', noise_level=0.5, noise_range=[5, 13]):
        
        self.root = root
        self.mode = mode
        
        self.nlevel = noise_level
        self.nrange = noise_range
        
        self.kernel = 5
        self.files = collections.defaultdict(list)
        
        with open(os.path.join(root, 'train_select_label.txt')) as f:
            reader = f.readlines()
            self.train_list = [row[:-1] for row in reader]
            
        with open(os.path.join(root, 'val_select_label.txt')) as f:
            reader = f.readlines()
            self.val_list = [row[:-1] for row in reader]
            
        self.test_path = os.path.join(root,'ori_test')
        self.test_list = os.listdir(self.test_path)
            
        self.files['train'] = self.train_list
        self.files['val'] = self.val_list
        self.files['test'] = self.test_list
        
        self.files = self.files[self.mode]
        
        
        self.test = []
        for idx in self.test_list:
            self.test.append(idx)
        print('total testing images: ', len(self.test))   
        
        
        
    def __getitem__(self, index):

        if self.mode == 'train':
            
            img_name = self.files[index] 
            data_ret, label_ret = self.readfile(img_name,img_name)           

            out_input, out_label, out_qua, noi_level = noise_generator(data_ret, label_ret,
                                                            nlevel=self.nlevel, nrange=self.nrange)
            weight = np.array([(out_label == 0).sum(),(out_label == 1).sum()
                               ])
            if(weight[0] == 0):
                weight = np.array([0, 1])
            else:
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
            out_label = out_label.unsqueeze(0)
            return out_input, out_label, out_qua, noi_level, weight
        else:
            img_name = self.files[index] 
            data_ret, label_ret = self.readfile(img_name,img_name) 
            
            o_input = np.zeros([3, data_ret.shape[0], data_ret.shape[1]])
            for i in range(0,3):
                o_input[i] = data_ret[:,:,i].copy()

            out_input = torch.Tensor(o_input)
            out_label = torch.Tensor(label_ret)
            out_label = out_label.unsqueeze(0)
            return out_input, out_label,img_name


    def __len__(self): # number of possible position
        return len(self.files)
        
    def readfile(self, img_name,lbl_name):
                  
        img_path = self.root + self.mode + '/' + img_name.split('.')[0] + '.tif' 
        lbl_path = self.root + self.mode + '_label/' + lbl_name

        img = Image.open(img_path).convert('RGB')
        
        img = np.array(img)
        lbl = Image.open(lbl_path)
        lbl = np.array(lbl)
        
        return img, lbl
    

def noise_generator(out_input, out_label, nlevel, nrange):
    input_size = np.shape(out_input)
    
    tmp_label = np.zeros(out_label.shape) #########################
    
    npix = 0
    if np.random.rand() < nlevel:
        ntype = np.random.randint(2)  # 0 for dilation, 1 for erosion
        npix = nrange[0] + (nrange[1] - nrange[0]) * np.random.rand()
        for t in range(1):
            if ntype:
                tmp_label = (distrans(1-out_label) < npix).astype(np.float)
                
            else:
                tmp_label = 1-(distrans(out_label) < npix).astype(np.float)
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

