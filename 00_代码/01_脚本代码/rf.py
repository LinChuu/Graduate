#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:04:06 2021

@author: chuang.lin
"""

#method bag
import pickle
from PIL import Image,ImageMath
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
import time
#import PIL as plt

from sklearn.externals import joblib

from sklearn.ensemble import RandomForestClassifier


import method
import os

import sklearn
import tifffile


def get_pascal_labels():
        return np.asarray([[0,0,0], [144,18,176], [0,128,0], [209,161,17], [0,0,128], [128,0,128],
                              [59,167,196], [25,237,14], [84,21,24], [192,0,0], [64,128,0], [192,128,0],
                              [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
                              [0,192,0], [128,192,0], [0,64,128]])
def decode_segmap(temp, plot=False):
        label_colours = get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0,9):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]
        print(r.shape)
        rgb = np.zeros((1489,1538,3))
        print(rgb[:, :, 0].shape)
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

# RandomForest
def randomforest(imagelist,lablelist,test1_imagelist, test_imagelist):
    # model
    
    
    clf = RandomForestClassifier(n_estimators=160, max_depth=None, min_samples_split=10)
    
    
    start_time = time.time()
    print("model fit")
    rf = clf.fit(imagelist, lablelist)
    
    joblib.dump(rf,'/nas/chuang.lin/label_trans/model/rf3.model')
    diff = time.time() - start_time
    print("single_time need:%f"%(diff*1000))
    #clf = joblib.load('/data1/wenna.xu/TM_ori/farmlandall/RFvis/model/rf.model')
    #
    #    predict image
    #pretrain = clf.predict(imagelist)
    
    # predict test1
    pretest1 = clf.predict(test_imagelist)
    
 
    print("get image")
    methodname = 'Random Forest'
    #pretrainvis = pretrain.reshape(4050,256,256)
    #lavis = lablelist.reshape(4050,256,256)
    pretest1vis = pretest1.reshape(1,1489,1538)
    #latest= test1_lablelist.reshape(868,256,256)
    
#    for i in range(1):
#        print(i,np.array(pretrainvis).shape,np.array(pretrainvis)[i,:,:].shape)
#        pretrainvistrain=decode_segmap(np.array(pretrainvis)[i,:,:])
#        lavistrain=decode_segmap(np.array(lavis)[i,:,:])
#        pretrainvistrain = Image.fromarray(pretrainvistrain.astype('uint8'))
#        lavistrain = Image.fromarray(lavistrain.astype('uint8'))
#        pretrainvistrain.save("/data1/wenna.xu/TM/farmlandall/RFvis/trainpre"+str(i)+'.jpg')
#        lavistrain.save("/data1/wenna.xu/TM/farmlandall/RFvis/trainlable"+str(i)+".jpg" )
    
    for i in range(1):
        
        #pretest1vistest=(np.array(pretest1vis)[i,:,:])
        pretest1vistest=decode_segmap(np.array(pretest1vis)[i,:,:])
        #latestvis=decode_segmap(np.array(latest)[i,:,:])
    
    
        pretest1vistest = Image.fromarray(pretest1vistest.astype('uint8'))
        #latestvis = Image.fromarray(latestvis.astype('uint8'))
 
        #latestvis.save("/data1/wenna.xu/TM/farmlandall/RFvis/testlable"+str(i)+".jpg" )
        pretest1vistest.save("/nas/chuang.lin/label_trans/result3.tif")
        

    
    # print
    diff = time.time() - start_time
    print("single_time need:%f"%(diff*1000))

def sipleimagerun():
    # path of data
    '''
    max_train_everynum = 1000+1
    begin_train_everynum= 80
    step_train_everynum= 80
    randonm_times = 3
    '''
    # read image
   
    imagepath = "/nas/chuang.lin/label_trans/dataset/sb_band4.tif"
    #labelpath = "/nas/chuang.lin/label_trans/dataset/resample_label.tif"
    
    imagelist = []
    lablelist = []
    
    test1_imagelist = []
    test_imagelist = []
    
    
    readimage = tifffile.imread(imagepath)
    #readimagelabel = Image.open(labelpath)
    readimage = np.array(readimage)
    #readimagelabel = np.array(readimagelabel)
    print(readimage.shape)
    
    row = readimage.shape[0]
    col = readimage.shape[1]
  
    
    for i in range(row):
        for j in range(col):
            if (readimage[i][j][4] == readimage[0][0][4]):       
                test1_imagelist.append(readimage[i][j][0:4])
            else:
                imagelist .append(readimage[i][j][0:4])
                lablelist.append(readimage[i][j][4])
            test_imagelist.append(readimage[i][j][0:4])
            
            
    imagelist = np.array(imagelist)
    lablelist = np.array(lablelist)
    test1_imagelist = np.array(test1_imagelist)
    test_imagelist = np.array(test_imagelist)
    print(imagelist.shape)
    print(lablelist.shape)

    
       
    
    
    randomforest(imagelist,lablelist,test1_imagelist, test_imagelist)
    
    
   




  
if __name__ == '__main__':
    sipleimagerun()
#    imagepath = "/nas/chuang.lin/label_trans/result1.tif"
#    readimage = tifffile.imread(imagepath)
#    
#    readimage = np.array(readimage)
#    print(readimage.shape)
#    decode_segmap(readimage)