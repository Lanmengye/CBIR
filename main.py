import torch
import net_train
from _datetime import datetime
import pickle
import os
from DataUtils import MNIST_Util

if __name__ == '__main__':
    bits = [12, 24, 32, 48],
    net_params = {
        'model_name': 'resnet',
        'bit': 12,
        'batch_size': 40,
        'num_of_classes': 10,
        'use_gpu': torch.cuda.is_available(),
        'epochs': 100,
        'learning_rate': 0.05,
        'weight_decay': 10 ** -5,
        'eta': 50
    }

    srcPath = os.path.abspath('Data/mnist/')

    dataHelper = MNIST_Util.MNIST_Helper(srcPath, net_params['batch_size'], net_params['num_of_classes'])
    print(srcPath)

    for bit in bits:
        filename = 'log/' + str(bit) + 'bit' + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '.pkl'
        result = net_train.deephash(net_params, dataHelper)
        fp = open(result['filename'], 'wb')
        pickle.dump(result, fp)
        fp.close()
