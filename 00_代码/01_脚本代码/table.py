#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 04:35:33 2020

@author: chuang.lin
"""
import pandas as pd 
import matplotlib.pyplot as plt 
df = pd.read_excel("/home/chuang.lin/Desktop/graduation/dataset/img_process/WLUTABEL/miou_erode.xlsx") 

print(df["WLU_9_1"]) 
plt.plot(df["noisy_rate"],df["UNET_9_1"],label='UNET_9_1',linewidth=2,color='r',linestyle='-') 
plt.plot(df["noisy_rate"],df["WLU_9_1"],label='WLU_9_1',linewidth=2,color='r',linestyle='-.') 
plt.plot(df["noisy_rate"],df["UNET_9_2"],label='UNET_9_2',linewidth=2,color='g',linestyle='-') 
plt.plot(df["noisy_rate"],df["WLU_9_2"],label='WLU_9_2',linewidth=2,color='g',linestyle='-.') 
plt.plot(df["noisy_rate"],df["UNET_9_3"],label='UNET_9_3',linewidth=2,color='b',linestyle='-') 
plt.plot(df["noisy_rate"],df["WLU_9_3"],label='WLU_9_3',linewidth=2,color='b',linestyle='-.') 
plt.xlabel("Noisy Rate") 
plt.ylabel('MIOU') 
plt.title("Erode MIOU") 
plt.legend() 
plt.grid() 
plt.savefig("/home/chuang.lin/Desktop/graduation/dataset/img_process/WLUTABEL/WLU2_erode_MIOU.png")
plt.show()


