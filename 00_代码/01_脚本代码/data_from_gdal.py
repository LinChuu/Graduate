from osgeo import gdal
import os
def date_read_from_it_name(datapath,imagename):
    # 说明，根据图片的名字和路径提取图片和标签
    # 图片文件的路径 datapath
    # imagename 图片的名字
    # 存储路径如下
    # image/x_image.tif
    # label/x_label.tif
    imagepath = os.path.join(datapath, 'image')
    labelpath = os.path.join(datapath, 'label')
    labelname = imagename.replace("_image", "_label")
    imageobject = load_TIF_NoGEO(imagepath, imagename)
    labelobject = load_TIF_NoGEO(labelpath, labelname)
    # image重塑
    imageobject = imageobject.swapaxes(0, 2)
    imageobject = imageobject.swapaxes(0, 1)
    # 判断image和label是否对应，如果不对应则报错，并停止循环
    rows_I, cols_I, dims_I = imageobject.shape
    rows_L, cols_L = labelobject.shape
    if rows_I != rows_L or cols_I != cols_L:
        print('The image must correspond to the label, please check the image!')
        exit(1)
    return imageobject,labelobject
# 读入TIF，并接收地理信息
def read_image(path ,name):
    imagepath = os.path.join(path,name)
    dataset = gdal.Open(imagepath)
    # imformation read
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    # geo_imformation read
    im_geotrans = dataset.GetGeoTransform()
    im_proj = dataset.GetProjection()
    # numpy write
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    #print(im_data.shape)
    return im_data,im_geotrans,im_proj
# 同时写入地理信息，保存tif
def write_image(image,path ,name,im_geotrans,im_proj):
    if 'int8' in image.dtype.name:
        datatype=gdal.GDT_Byte
    elif 'int16' in image.dtype.name:
        datatype = gdal.GDT_Byte
    else:
        datatype=gdal.GDT_Float32

    if len(image.shape)==3:
        im_bands,im_height,im_width=image.shape
    else:
        im_bands,(im_height, im_width)= 1,image.shape
    #(3652,3590)
    print('(im_height, im_width):',(im_height, im_width))
    #writeimage
    save_dir=path
    #
    driver = gdal.GetDriverByName("GTiff")
    pathway = os.path.join(save_dir, name)
    dataset = driver.Create(pathway,im_width, im_height, im_bands, datatype)
    print('im_width:',dataset.RasterXSize)
    print('im_height:',dataset.RasterYSize)
    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(image)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(image[i])
# 读入文件，但是不拾取其地理坐标，读入文件为tif
def load_TIF_NoGEO(path,name):
    imagepath = os.path.join(path,name)
    dataset = gdal.Open(imagepath)
    print(imagepath)
    # imformation read
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    # numpy write
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    #print(im_data.shape)
    return im_data
# 写入文件，但是不拾取其地理坐标，写入文件为tif
def write_image_NoGeo(image,path ,name):
    if 'int8' in image.dtype.name:
        datatype=gdal.GDT_Byte
    elif 'int16' in image.dtype.name:
        datatype = gdal.GDT_Byte
    else:
        datatype=gdal.GDT_Float32
    if len(image.shape)==3:
        im_bands,im_height,im_width=image.shape
    else:
        im_bands,(im_height, im_width)= 1,image.shape
    #(3652,3590)
    print('(im_height, im_width):',(im_height, im_width))
    #writeimage
    save_dir=path
    #
    driver = gdal.GetDriverByName("GTiff")
    pathway = os.path.join(save_dir, name)
    dataset = driver.Create(pathway,im_width, im_height, im_bands, datatype)
    print('im_width:',dataset.RasterXSize)
    print('im_height:',dataset.RasterYSize)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(image)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(image[i])