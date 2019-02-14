import torch
import net_train
from _datetime import datetime
import pickle
import os

if __name__ == '__main__':
    net_params = {
        'model_name': 'resnet',
        'bits': [12, 24, 32, 48],
        'batch_size': 40,
        'num_of_classes': 10,
        'use_gpu': torch.cuda.is_available(),
        'epochs': 100,
        'learning_rate': 0.05,
        'weight_decay': 10 ** -5,
        'eta': 50
    }


    srcPath = os.path.abspath('../Data/cifar-10/')

    for bit in net_params['bits']:
        filename = 'log/' + str(net_params['bit']) + 'bit' + datetime.now().strftime("%y-%m-%d-%H-%M-%S") + '.pkl'
        result = net_train.deephash(net_params, srcPath)
        fp = open(result['filename'], 'wb')
        pickle.dump(result, fp)
        fp.close()
