# -*- coding: utf-8 -*-

import os
import torch
from torchvision import *
from torch.autograd import Variable

import CNN_model

import numpy as np


# 加载类别标签信息
def loadLabels(filename, DATA_DIR):
    path = DATA_DIR + filename
    print(path)
    fp = open(path, 'r')
    labels = [x.strip() for x in fp]
    fp.close()
    return torch.LongTensor(list(map(int, labels)))


# 获得one-hot编码
def getOnehotCode(target, nclasses):
    target_onehot = torch.FloatTensor(target.size(0), nclasses)
    target_onehot.zero_()
    target_onehot.scatter_(1, target.view(-1, 1), 1)
    return target_onehot


# 计算相似矩阵
def calcSim(batch_label, train_label):
    s = (batch_label.mm(train_label.t()) > 0).type(torch.FloatTensor)
    return s


# 获取深层卷积网络模型
def get_cnn_model(model_name, bits):
    if model_name == 'vgg11':
        vgg11 = models.vgg11(pretrained=True)
        cnn_model = CNN_model.cnn_model(vgg11, model_name, bits)
    if model_name == 'alexnet':
        alexnet = models.alexnet(pretrained=True)
        cnn_model = CNN_model.cnn_model(alexnet, model_name, bits)
    if model_name == 'resnet':
        cnn_model = CNN_model.getResnetModel(bits)
    if torch.cuda.is_available():
        cnn_model = cnn_model.cuda()
    return cnn_model


def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.]).cuda()))
    else:
        lt = torch.log(1 + torch.exp(-torch.abs(x))) + torch.max(x, Variable(torch.FloatTensor([0.])))
    return lt


# 计算损失函数
def calc_loss(U, B, S, eta, nums):
    theta = U.mm(U.t()) / 2
    if torch.cuda.is_available():
        lt = torch.log(1 + torch.exp(-torch.abs(theta))) + torch.max(theta, Variable(torch.FloatTensor([0.]).cuda()))
    else:
        lt = torch.log(1 + torch.exp(-torch.abs(theta))) + torch.max(theta, Variable(torch.FloatTensor([0.])))
    t1 = (theta * theta).sum() / (nums * nums)
    l1 = (- theta * S + lt.data).sum()
    l2 = (U - B).pow(2).sum()
    total_loss = l1 + eta * l2
    return total_loss, l1, l2, t1


# 调整学习率
def AdjustLearningRate(optimizer, epoch, learning_rate):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


# 生成图像二值哈希码
def GenerateCode(model, data_loader, num_data, bit, use_gpu):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for iter, data in enumerate(data_loader, 0):
        data_input, _, data_ind = data
        if use_gpu:
            data_input = Variable(data_input.cuda())
        else:
            data_input = Variable(data_input)
        output = model(data_input)
        if use_gpu:
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
        else:
            B[data_ind.numpy(), :] = torch.sign(output.data).numpy()
    return B
