#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 15:17:13 2020

@author: chuang.lin
"""

'''
作用：对tif图像进行裁剪，合成，赋予地理坐标
'''

import cv2
import os
import numpy as np
from scipy import misc
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tifffile as tiff
from osgeo import gdal
from osgeo import gdal,osr
from osgeo.gdalconst import *
import scipy.misc
import math


def projTransform(filename):
    source=osr.SpatialReference()
    source.ImportFromEPSG(32650)
    #目标图像投影
    target=osr.SpatialReference()
    target.ImportFromEPSG(3857)
    coordTrans=osr.CoordinateTransformation(source,target)
    #打开源图像文件
    ds=gdal.Open(filename)
    #仿射矩阵六参数
    mat=ds.GetGeoTransform()
    #源图像的左上角与右下角像素，在目标图像中的坐标
    (ulx, uly, ulz)=coordTrans.TransformPoint(mat[0],mat[3])
    (lrx, lry, lrz ) = coordTrans.TransformPoint(mat[0] + mat[1]*ds.RasterXSize, mat[3] + mat[5]* ds.RasterYSize )
    #创建目标图像文件（空白图像），行列数、波段数以及数值类型仍等同原图像
    driver=gdal.GetDriverByName("GTiff")
    ts=driver.Create("fdem_lonlat.tif",ds.RasterXSize,ds.RasterYSize,1,GDT_UInt16)
    #转换后图像的分辨率
    resolution=(int)((lrx-ulx)/ds.RasterXSize)
    #转换后图像的六个放射变换参数
    mat2=[ulx, resolution,0,uly,0, -resolution]
    ts.SetGeoTransform(mat2)
    ts.SetProjection(target.ExportToWkt())
    #投影转换后需要做重采样
    gdal.ReprojectImage(ds, ts, source.ExportToWkt(), target.ExportToWkt(), gdal.GRA_Bilinear)
    #关闭
    ds = None
    ts= None


'''
作用：读取图像
输入：
    clip_file_path：要裁剪图像存放的文件夹
    filename:图像的文件名
    mode:生成哪种类型的图像 rgb, nrg, allband
输出：
    返回图像的地图投影信息，仿射矩阵，图像数组，图像行数，列数
'''
# 读图像文件
def read_img(clip_file_path, filename, mode="rgb"):
    
    filename = clip_file_path + filename

    dataset = gdal.Open(filename)  # 打开文件
    
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    band=dataset.RasterCount
    
    im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
    im_proj = dataset.GetProjection()  # 地图投影信息
    data = np.zeros([im_height,im_width,band])
    
    
    for i in range(band):    # 将数据写成数组，对应栅格矩阵
        dt=dataset.GetRasterBand(i+1)
        temp = dt.ReadAsArray(0,0,im_width,im_height)
        temp = np.uint8(np.double(temp)*255.0/1024)
        data[:,:,i] = temp
    
    
    if mode == "rgb":
        img = np.zeros([im_height,im_width,band])
        img[:,:,0] = data[:,:,2]
        img[:,:,1] = data[:,:,1]
        img[:,:,2] = data[:,:,0]
    elif mode == "nrg":
        img = np.zeros([im_height,im_width,band])
        img[:,:,0] = data[:,:,3]
        img[:,:,1] = data[:,:,2]
        img[:,:,2] = data[:,:,1]
    elif mode == "allbands":
        img = np.zeros([math.ceil(im_height/256.0)*256,math.ceil(im_width/256.0)*256,band])
        print(img.shape)
        for i in range(band):
            img[0:im_height,0:im_width,i] = data[0:im_height,0:im_width,i]
        
    
    img = np.uint8(img)
    
    del dataset  # 关闭对象，文件dataset
    return im_proj, im_geotrans, img, img.shape[0], img.shape[1]

'''
作用：对图像进行裁剪
输入：
    img:图像数组
    clip_file_path:要裁剪图像存放的文件夹
    filename:图像名称
    
'''

def clip_one_picture(img, clip_file_path, filename):
    
    rows, cols = 256, 256
    #print(img.shape)
    #img = Image.fromarray(img)
    clip_file = clip_file_path + filename
    img = tiff.open(clip_file)
    sum_rows = img.shape[0]   
    sum_cols = img.shape[1]
    print(sum_rows,sum_cols)
    save_path = clip_file_path + filename.split('.')[0]
    
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cols_stride = 256
    rows_stride = 256
    for i in range(int(sum_cols/cols_stride)):
        for j in range(int(sum_rows/rows_stride)):
            img_path = save_path+'/'+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+'.tif'
            img_roi = img.crop((i*cols_stride,j*rows_stride,i*cols_stride+cols,j*rows_stride+rows))
            img_roi.save(img_path)
            print(str(j)+'_'+str(i)+'.tif')

'''
作用：将文件夹中的小图合成为一张大图
输入：
    merge_file_path + cla_name：表示小图存放的文件夹路径
    num_rows：表示行的张数， num_rows*小图的行大小=大图的行大小
    num_cols：表示列的张数，num_cols*小图的行大小=大图的列大小
    res_path:表示大图存放的文件夹

'''

def merge_picture(merge_file_path,cla_name,num_rows,num_cols,res_path):
    filename, _ = file_name(merge_file_path,cla_name, '.tif')
   
    shape = tiff.imread(filename[0]).shape
    
    rows = shape[0]
    cols = shape[1]
    
    dst = np.zeros((shape[2], num_rows*rows, num_cols*cols),np.uint8)
    print(dst.shape)
    cols_stride = 256
    rows_stride = 256
    for b in range(shape[2]):
        for i in range(len(filename)):
            img = tiff.imread(filename[i])
            #print(img.shape)
            rows_th = int(filename[i].split('_')[-2])
            cols_th = int(filename[i].split('_')[-1].split('.')[0])
            #print(rows_th, cols_th)

            dst[b, rows_th*rows_stride:rows_th*rows_stride+rows,cols_th*cols_stride:cols_th*cols_stride+cols] = img[:, :, b]
        
    if not os.path.exists(res_path):
            os.makedirs(res_path)
    merge_img_name = res_path + filename[0].split('/')[-1].split('_')[0] + '_merge.tif'
    print(dst.shape)
    dst = tiff.imwrite(merge_img_name, dst)
    #dst.save()

'''
作用：读取文件夹中文件
输入：
    root_path + cla_name:表示要读取的文件夹路径
    picturetype:表示图片类型
输出：
    filename：文件路径集合
    allFile: 文件名称集合
'''

def file_name(root_path, cla_name,picturetype):
    filename = []
    allFile = []
    for root, dirs, files in os.walk(root_path+cla_name+'/'):
        for file in files:
            if os.path.splitext(file)[1] == picturetype:
                
                filename.append(os.path.join(root,file))
                allFile.append(file)
    return filename, allFile

'''
作用：将地理坐标写出图像中
输入：
    filename：图像名称
    im_proj:地图投影信息
    im_geotrans:仿射矩阵
    im_data:图像数据
'''

def write_img(filename, im_proj, im_geotrans, im_data):
    # gdal数据类型包括
    # gdal.GDT_Byte,
    # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
    # gdal.GDT_Float32, gdal.GDT_Float64

    # 判断栅格数据的数据类型
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    # 判读数组维数
    if len(im_data.shape) == 3:
        im_height, im_width, im_bands = im_data.shape
    else:
        (im_height, im_width), im_bands = 1, im_data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
    dataset.SetProjection(im_proj)  # 写入投影

    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i + 1).WriteArray(im_data[:,:,i])
    #print(dataset.GetGeoTransform())
    del dataset
if __name__=='__main__':
    
    '''
    流程：读取文件夹的所有文件，遍历每一张图像，进行裁剪，之后对小图写入地理坐标，
         再把小图合成大图，判断是否合成正确
    '''
    
    clip_file_path = "/nas/chuang.lin/0602/res_x/"  #要裁减的文件夹
    fileName, allFile = file_name(clip_file_path, "", '.tif')
    print(allFile)
    
    for img_name in allFile:
    #filename = "Jiangmen_LT5_2010_res.tif"  #要裁减的图片
        cla_name = img_name.split('.')[0]
        save_path = clip_file_path + img_name.split('.')[0]
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        cols_stride = 256
        rows_stride = 256
        rows, cols = 256, 256
        proj, geotrans, values, row, column = read_img(clip_file_path, img_name, "allbands")  # 读数据
        
        for i in range(int(column/cols_stride)):
            for j in range(int(row/rows_stride)):
                img_path = save_path+'/'+os.path.splitext(img_name)[0]+'_'+str(j)+'_'+str(i)+'.tif'
                
                img_roi = values[j*rows_stride:j*rows_stride+rows, i*cols_stride:i*cols_stride+cols, :]
                write_img(img_path, proj, geotrans, img_roi)#写数据
                #print(str(j)+'_'+str(i)+'.tif')
        merge_picture(clip_file_path, cla_name, (int)(row/256), (int)(column/256),clip_file_path)
        #write_img(resfile, proj, geotrans, values)#写数据
    '''
    test_path = '/nas/chuang.lin/0602/res/'   
    cla_name = "Jiangmen_LT5_2010_res"
    merge_picture(test_path, cla_name, (int)(row/256), (int)(column/256),test_path)
    #clip_one_picture(values, clip_file_path, filename)
    
    test_path = '/nas/chuang.lin/road/dataset/ori/'   
    cla_name = "GF1_PMS1_E86"
    merge_picture(test_path, cla_name, (int)(row/256), (int)(column/256),test_path)
    write_img(resfile, proj, geotrans, values)#写数据
    '''
