#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 03:46:35 2020

@author: chuang.lin
"""

import matplotlib.pyplot as plt
import numpy as np


def get_pascal_labels(self):
    return np.asarray([[0,0,0], [255,255,255]])
    
def decode_segmap(temp, plot=False):
    label_colours = np.asarray([[0,0,0], [255,255,255]])
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 2):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb