# -*- coding: utf-8 -*-
import DataUtils.DataSetUtil as DS
import Preparing as Pre
from torch.utils.data import DataLoader


class MNIST_Helper():
    def __init__(self, src, batch_size, nums_of_classes):
        self.src = src
        self.batch_size = batch_size
        self.nums_of_classes = nums_of_classes

    def getTrainDataAndTestData(self):
        ds = {
            'train': DS.MNIST_DS(self.src + '/train', 'train_img.txt', 'train_label.txt'),
            'test': DS.MNIST_DS(self.src + '/test', 'test_img.txt', 'test_label.txt')
        }
        labels = {
            'train': Pre.loadLabels('/train_label.txt', self.src + '/train'),
            'test': Pre.loadLabels('/test_label.txt', self.src + '/test'),
        }
        trainData = {
            'num': len(ds['train']),
            'dataloader': DataLoader(ds['train'], batch_size=self.batch_size, shuffle=True, num_workers=4),
            'label': labels['train'],
            'one_hots': Pre.getOnehotCode(labels['train'], self.nums_of_classes)
        }

        testData = {
            'num': len(ds['test']),
            'dataloader': DataLoader(ds['test'], batch_size=self.batch_size, shuffle=True, num_workers=4),
            'label': labels['test'],
            'one_hots': Pre.getOnehotCode(labels['test'], self.nums_of_classes)
        }

        return trainData, testData
