# -*- coding: utf-8 -*-

import os
import sys
from DataUtils import DataProcessor as dp

if __name__ == '__main__':
    path = os.path.abspath('../DataSets/mnist/')
    imgdata, labels = dp.load_mnist(path)

    test, train = dp.data_prepare_mnist(imgdata, labels)

    dp.save_img_label2txt_mnist(test, 'test', os.path.abspath('../Data/mnist/test'))
    dp.save_img_label2txt_mnist(train, 'train', os.path.abspath('../Data/mnist/train'))
