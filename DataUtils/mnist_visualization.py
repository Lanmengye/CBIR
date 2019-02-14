#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 09:32:42 2019

@author: lanmengye
"""

import os
import  struct
import numpy as np
import matplotlib.pyplot as plt



# struct.unpack: 用于将字节流转换成python数据类型。
# 它的函数原型为: struct.unpack(fmt, string)，该函数返回一个元组。
if __name__ =='__main__':
    labels_path = os.path.join(os.path.abspath('../DataSets/mnist/'),'%s-labels.idx1-ubyte' % 'train')
    images_path = os.path.join(os.path.abspath('../DataSets/mnist/'),'%s-images.idx3-ubyte' % 'train')
    with open(labels_path,'rb') as labelsStream:
        struct.unpack('>II',labelsStream.read(8))
        labels = np.fromfile(labelsStream,dtype=np.uint8)
    with open(images_path, 'rb') as imgsStream:
        struct.unpack('>IIII',imgsStream.read(16))
        images = np.fromfile(imgsStream, dtype=np.uint8).reshape(len(labels), 784)
    
    fig, ax = plt.subplots(nrows=2,ncols=5,sharex=True,sharey=True, )
    ax = ax.flatten()
    for i in range(10):
        img = images[labels == i][1].reshape(28,28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
