#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:42:33 2020

@author: chuang.lin
"""

import os, sys
import numpy as np
import pickle, h5py, time, argparse, itertools, datetime

import torch
import torch.nn as nn
import torch.utils.data
from libs.loss import WeightedCELoss

from libs.dataset import CTDataset
from libs.decoder import decode_segmap

from libs.se_model import seg_module
from libs.unet_model import UNet
from libs.qua_model import se_res_extracter as qua_module

from tensorboardX import SummaryWriter
from PIL import Image

import scipy.io as sio
from sklearn.metrics import confusion_matrix

np.set_printoptions(precision=4,suppress=True)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["CUDA_VISIBLE_DEVICES"]="3"

def get_args():
    parser = argparse.ArgumentParser(description='Quality Awareness Model')
    parser.add_argument('-o','--output', default='result/',
                        help='Output path')
    parser.add_argument('-n','--noise',type=float, default=0,help='Noise percentage')
    parser.add_argument('-ns',type=float, default=5,help='Noise range (min)')
    parser.add_argument('-nm',type=float, default=13,help='Noise range (max)')
    parser.add_argument('-lr', type=float, default=.0001,
                        help='Learning rate')
    parser.add_argument('-mi','--max-iter', type=int, default=10002,
                        help='Total number of iteration')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=60,
                        help='# of the epochs')
    parser.add_argument('--n_classes', nargs='?', type=int, default=2,
                        help='# of the classes')
    parser.add_argument('--iter-save', type=int, default=100,
                        help='Number of iteration to save')
    parser.add_argument('--epoch-save', type=int, default=10,
                        help='Number of iteration to save')
    parser.add_argument('-g','--num-gpu', type=int,  default=2,
                        help='Number of gpu')
    parser.add_argument('-c','--num-cpu', type=int,  default=4,
                        help='Number of cpu')
    parser.add_argument('-b','--batch-size', type=int,  default=16,
                        help='Batch size')
    parser.add_argument('--model_path', nargs='?', type=str, default='result/saveModule/250_seg_module.pth',
                        help='Path to the saved model')
    args = parser.parse_args()
    return args

def init(args):
    
    sn = args.output+'/'
    if not os.path.isdir(sn):
        os.makedirs(sn)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    return device

def dice(pred, gt, label):
    intersection = ((gt == label) * (pred == label)).sum().astype(np.float)
    
    return 2 * intersection / ((gt == label).sum() + (pred == label).sum()).astype(np.float)

def evalue(pred, labels, n_classes):
    
    lbl0 = (labels == 0).astype(np.int)
    lbl1 = (labels == 1).astype(np.int)
    pred0 = (pred == 0).astype(np.int)
    pred1 = (pred == 1).astype(np.int)
    #print((pred == 1).astype(np.int).sum())
    TP = (lbl1 * pred1).sum().astype(np.float)
    TN = (lbl0 * pred0).sum().astype(np.float)
    FN = lbl1.sum() - TP
    FP = pred1.sum() - TP
    F1 = 2*TP/(2*TP+FP+FN)
    ACC = (TP+TN)/(TP+TN+FN+FP)
    MIOU = ((TP/(TP+FP+FN))+(TN/(TN+FN+FP)))/2
    
    return ACC, MIOU, F1


def eval_valid(args,loader, model, device, logger, writer):
    model.eval()
    
    ACC = np.zeros(1)
    acc = np.zeros(1)
    MIOU = np.zeros(1)
    miou = np.zeros(1)
    F1 = np.zeros(1)
    f1 = np.zeros(1)
    dcnt = len(loader)
    cm = np.zeros((2,2), dtype=np.float64)
    for i, (data, label,img_name) in enumerate(loader):
        data = data.to(device)
        pred = model(data)
        pred = pred.data.cpu().numpy()
        label = label.data.numpy()
        pred0 = np.argmax(pred, axis=1)
        label = label.reshape([-1])
        pred = pred0.reshape([-1])
        
        ACC, MIOU, F1 = evalue(pred, label, 2)  
        acc += ACC
        miou += MIOU
        f1 += F1
        
        print('accuracy:', ACC)
        print('MIOU:', MIOU)
        print('F1 score:', F1)
        logger.write("acc=%.5f miou=%.5f f1 score=%.5f" % ( \
                        ACC,MIOU,F1))   
        print(img_name[0])
        
        file_name = args.output+'/saved_test_images/'+img_name[0].split('_',1)[0]+'/'+img_name[0]
        if not os.path.exists(args.output+'/saved_test_images/'+img_name[0].split('_',1)[0]+'/'):
            os.makedirs(args.output+'/saved_test_images/'+img_name[0].split('_',1)[0]+'/')
         
        pred0 = pred0.squeeze()    
        decoded = decode_segmap(pred0)
        image = Image.fromarray(np.uint8(decoded))
        image.save(file_name)
        
        cm += confusion_matrix(label.ravel(), pred.ravel(), labels=range(2))
    
    
    
    file = open('result.txt',mode='w')
    print('Over Accuracy:%f'%(acc / dcnt),file=file)
    print('Over MIOU:%f'%(miou / dcnt),file=file)
    print('Over F1 score:%f'%(f1 / dcnt),file=file)
    
    print('Over Accuracy:%f'%(acc / dcnt))
    print('Over MIOU:%f'%(miou / dcnt))
    print('Over F1 score:%f'%(f1 / dcnt))
    
    
    sio.savemat(args.output+"/cm.mat",{'cm':cm})
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    TN = cm[0,0]
    FP = cm[0,1]
    FN = cm[1,0]
    TP = cm[1,1]
    accuracy = (TP + TN) / (TP + FP + TN + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    dice = 2 * TP / (2 * TP + FP + FN)
    p0 = accuracy
    pe = ((TP + FN) * (TP + FP) + (FP + TN) * (FN + TN)) / ((TP + FP + TN + FN) * (TP + FP + TN + FN))
    kappa = (p0 -pe) / (1 - pe)
    
    
    
    print('准确率:%f'%(accuracy))
    print('精确率:%f'%(precision))
    print('召回率:%f'%(recall))
    print('F1-score:%f'%(f1))
    print('Dice:%f'%(dice))
    print('Kappa:%f'%(kappa))
    
    print('准确率:%f'%(accuracy),file=file)
    print('精确率:%f'%(precision),file=file)
    print('召回率:%f'%(recall),file=file)
    print('F1-score:%f'%(f1),file=file)
    print('Dice:%f'%(dice),file=file)
    print('Kappa:%f'%(kappa),file=file)
    
    
    file.close()
        
    


def get_input(args):
    
    data_path = "/home/chuang.lin/Desktop/graduation/dataset/IAI/"
    
    
    test_dataset = CTDataset(data_path,mode='test')
    
    test_loader =  torch.utils.data.DataLoader(test_dataset,
            batch_size=1, shuffle=False,
            num_workers=args.num_cpu, pin_memory=True)
    return test_loader

def get_logger(args):
    log_name = args.output+'/log'
    date = str(datetime.datetime.now()).split(' ')[0]
    time = str(datetime.datetime.now()).split(' ')[1].split('.')[0]
    log_name += '_approx_'+date+'_'+time
    logger = open(log_name+'.txt','w') # unbuffered, write instantly

    # tensorboardX
    writer = SummaryWriter('runs/'+log_name)
    return logger, writer


def main():
    
    
    
    
    args = get_args()
    assert args.ns < args.nm

    print('0. initial setup')
    device = init(args) 
    logger, writer = get_logger(args)

    print('1. setup data')
    test_loader = get_input(args)
    
    print('2. setup model')
    model = UNet(n_classes=2,n_channels=3)
    model = nn.DataParallel(model)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    
    print('4. start training')
    eval_valid(args,test_loader, model, device, logger, writer)
    
    print('5. finish training')
    logger.close()
    writer.close()

if __name__ == "__main__":
    main()
