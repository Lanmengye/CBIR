# -*- coding: utf-8 -*-

import numpy as np
import pickle as p
import os
from matplotlib import image as imglib
import random
import struct


def load_cifar10(path):
    filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']
    imgdata = []
    labels = []
    for i in range(len(filenames)):
        filedir = os.path.join(path, filenames[i])
        with open(filedir, 'rb') as f:
            d = p.load(f, encoding='latin1')  # 四维数组 dict_keys(['batch_label', 'labels', 'data', 'filenames'])
            data = d['data'].reshape(10000, 3, 32, 32)  # 图像信息
            if i == 0:
                imgdata = data
                labels = np.array(d['labels'])
            else:
                imgdata = np.concatenate((imgdata, data), 0)
                labels = np.concatenate((labels, np.array(d['labels'])), 0)
    return imgdata, labels

def load_mnist(path):
    imgfiles = ['t10k-images.idx3-ubyte', 'train-images.idx3-ubyte']
    labelfiles = ['t10k-labels.idx1-ubyte', 'train-labels.idx1-ubyte']

    for i in range(2):
        images_path = os.path.join(path, imgfiles[i])
        labels_path = os.path.join(path, labelfiles[i])
        with open(labels_path, 'rb') as labelsStream:
            struct.unpack('>II', labelsStream.read(8))
            labels = np.fromfile(labelsStream, dtype=np.uint8)
        with open(images_path, 'rb') as imgsStream:
            struct.unpack('>IIII', imgsStream.read(16))
            images = np.fromfile(imgsStream, dtype=np.uint8).reshape(len(labels), 28, 28)
        if i == 0:
            imgdata = images
            labeldata = np.array(labels)
        else:
            imgdata = np.concatenate((imgdata, images), 0)
            labeldata = np.concatenate((labeldata, np.array(labels)), 0)
    return imgdata, labeldata



# 将CIFAR-10分为test\train\database三部分
def data_prepare_cifar10(imgs, labels):
    test_data = []
    test_label = []
    train_data = []
    train_label = []
    dataset_data = []
    dataset_label = []
    for label in range(10):
        indexs = (i for i, x in enumerate(labels) if x == label)
        indexs = list(indexs)
        random.shuffle(indexs)
        test_indexs = indexs[0:10]
        train_indexs = indexs[10:20]
        dataset_index = indexs[20:30]
        for i1 in test_indexs:
            test_label.append(labels[i1])
            test_data.append(imgs[i1])

        for i2 in train_indexs:
            train_label.append(labels[i2])
            train_data.append(imgs[i2])

        for i3 in dataset_index:
            dataset_label.append(labels[i3])
            dataset_data.append(imgs[i3])
    test = {'data': test_data, 'labels': test_label}
    train = {'data': train_data, 'labels': train_label}
    dataset = {'data': dataset_data, 'labels': dataset_label}
    return test, train, dataset


def data_prepare_mnist(imgs, labels):
    test_data = []
    test_label = []
    train_data = []
    train_label = []
    for label in range(10):
        indexs = (i for i, x in enumerate(labels) if x == label)
        indexs = list(indexs)
        random.shuffle(indexs)
        test_indexs = indexs[0:100]
        train_indexs = indexs[100:600]
        for i1 in test_indexs:
            test_label.append(labels[i1])
            test_data.append(imgs[i1])

        for i2 in train_indexs:
            train_label.append(labels[i2])
            train_data.append(imgs[i2])
    test = {'data': test_data, 'labels': test_label}
    train = {'data': train_data, 'labels': train_label}
    return test, train

# 保存图像，同时保存图像位置、标签到txt文件中
def save_img_label2txt_cifar10(data, kinds, path):
    filenametxt = ''
    labeltxt = ''

    imgdir, img_filename, label_filename = (kinds + '_img/',
                                            kinds + '_img' + '.txt',
                                            kinds + '_label' + '.txt')
    # 创建图像存放目录
    isExists = os.path.exists(os.path.join(path, imgdir))
    if not isExists:
        os.makedirs(os.path.join(path, imgdir))

    print(len(data['data']))
    # 保存图像
    for i in range(len(data['data'])):
        img_item = data['data'][i]
        img = np.transpose(img_item, (1, 2, 0))
        img_name = str(i + 1) + '.png'
        imglib.imsave(os.path.join(path, imgdir, img_name), img)
        filenametxt += imgdir + str(i + 1) + '.png' + '\n'
        labeltxt += str(data['labels'][i]) + '\n'

    # 图像文件名写入txt文件
    with open(os.path.join(path, img_filename), 'a+') as f:
        f.writelines(filenametxt)

        # 图像标签值写入txt文件
    with open(os.path.join(path, label_filename), 'a+') as f:
        f.writelines(labeltxt)


def save_img_label2txt_mnist(data, kinds, path):
    filenametxt = ''
    labeltxt = ''

    imgdir, img_filename, label_filename = (kinds + '_img/',
                                            kinds + '_img' + '.txt',
                                            kinds + '_label' + '.txt')
    # 创建图像存放目录
    isExists = os.path.exists(os.path.join(path, imgdir))
    if not isExists:
        os.makedirs(os.path.join(path, imgdir))

    print(len(data['data']))
    # 保存图像
    for i in range(len(data['data'])):
        img_item = data['data'][i]
        #img = np.transpose(img_item, (1, 2, 0))
        img_name = str(i + 1) + '.png'
        imglib.imsave(os.path.join(path, imgdir, img_name), img_item)
        filenametxt += imgdir + str(i + 1) + '.png' + '\n'
        labeltxt += str(data['labels'][i]) + '\n'

    # 图像文件名写入txt文件
    with open(os.path.join(path, img_filename), 'a+') as f:
        f.writelines(filenametxt)

        # 图像标签值写入txt文件
    with open(os.path.join(path, label_filename), 'a+') as f:
        f.writelines(labeltxt)