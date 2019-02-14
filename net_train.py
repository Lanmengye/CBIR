import Preparing as Pre
from DataUtils import DataSetUtil as DS
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
from torch.autograd import Variable
import CalcHammingRanking as CalcHR


def getTrainDataAndTestData(srcPath, net_params):
    ds = {
        'train': DS.CIFAR_10_DS(srcPath + 'train/', 'train_img.txt', 'train_label.txt'),
        'test': DS.CIFAR_10_DS(srcPath + 'test/', 'test_img.txt', 'test_label.txt')
    }
    labels = {
        'train': Pre.loadLabels('train_label.txt', srcPath + 'train/'),
        'test': Pre.loadLabels('test_label.txt', srcPath + 'test/'),
    }
    trainData = {
        'num': len(ds['train']),
        'dataloader': DataLoader(ds['train'], batch_size=net_params['batch_size'], shuffle=True, num_workers=4),
        'label': labels['train'],
        'one_hots': Pre.getOnehotCode(labels['train'], net_params['num_of_classes'])
    }

    testData = {
        'num': len(ds['test']),
        'dataloader': DataLoader(ds['test'], batch_size=net_params['batch_size'], shuffle=True, num_workers=4),
        'label': labels['test'],
        'one_hots': Pre.getOnehotCode(labels['test'], net_params['num_of_classes'])
    }

    return trainData, testData


def getDataBaseData(srcPath, net_params):
    ds = DS.CIFAR_10_DS(srcPath + 'database/', 'database_img.txt', 'database_label.txt')
    labels = Pre.loadLabels('database_label.txt', srcPath + 'database/'),
    databaseData = {
        'num': len(ds),
        'dataloader': DataLoader(ds, batch_size=net_params['batch_size'], shuffle=True, num_workers=4),
        'label': labels,
        'one_hots': Pre.getOnehotCode(labels, net_params['num_of_classes'])
    }
    return databaseData


def train(model, trainData, testData, net_params, optimizer, record):
    B, U = torch.zeros(trainData['num'], net_params['bit']), torch.zeros(trainData['num'], net_params['bit'])
    train_S = Pre.CalcSim(trainData['one_hots'], trainData['one_hots'])
    for epoch in range(net_params['epochs']):
        epoch_loss = 0.0
        for iter, traindata in enumerate(trainData['dataloader'], 0):
            train_input, train_label, batch_ind = traindata
            train_label = torch.squeeze(train_label)
            train_label_onehot = Pre.getOnehotCode(train_label, net_params['num_of_classes'])
            if net_params['use_gpu']:
                train_input, train_label = Variable(train_input.cuda()), Variable(train_label.cuda())
                S = Pre.CalcSim(train_label_onehot, trainData['one_hots'])
            else:
                train_input, train_label = Variable(train_input), Variable(train_label)
                S = Pre.CalcSim(train_label_onehot, trainData['one_hots'])
            model.zero_grad()
            train_outputs = model(train_input)
            for i, ind in enumerate(batch_ind):
                U[ind, :] = train_outputs.data[i]
                B[ind, :] = torch.sign(train_outputs.data[i])

            Bbatch = torch.sign(train_outputs)
            if net_params['use_gpu']:
                theta_x = train_outputs.mm(Variable(U.cuda()).t()) / 2
                logloss = (Variable(S.cuda()) * theta_x - Pre.Logtrick(theta_x, net_params['use_gpu'])).sum()(
                    trainData['num'] * len(train_label))
                regterm = (Bbatch - train_outputs).pow(2).sum() / (trainData['num'] * len(train_label))
            else:
                theta_x = train_outputs.mm(Variable(U).t()) / 2
                logloss = (Variable(S) * theta_x - Pre.Logtrick(theta_x, net_params['use_gpu'])).sum() / (
                        trainData['num'] * len(train_label))
                regterm = (Bbatch - train_outputs).pow(2).sum() / (trainData['num'] * len(train_label))

            loss = - logloss + net_params['eta'] * regterm
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]

        # 3 ==  len(dataloder)?
        print('[Train Phase][Epoch: %3d/%3d][Loss: %3.5f]' % (
            epoch + 1, net_params['epochs'], epoch_loss / 3), end='')
        optimizer = Pre.AdjustLearningRate(optimizer, epoch, net_params['learning_rate'])
        record['loss'].append(epoch_loss / 3)

        # test
        queryB = Pre.GenerateCode(model, testData['dataloader'], testData['num'], net_params['bit'],
                                  net_params['use_gpu'])
        testB = torch.sign(B).numpy()
        map_ = CalcHR.CalcMap(queryB, testB, testData['one_hots'].numpy(), trainData['one_hots'].numpy())
        record['map'].append(map_)
        print('[Test Phase ][Epoch: %3d/%3d] MAP(retrieval train): %3.5f' % (epoch + 1, net_params['epochs'], map_),
              end='\n')


def deephash(net_params, srcPath):
    model = Pre.get_cnn_model(net_params['model_name'], net_params['bit'])
    optimizer = optim.SGD(model.parameters(), lr=net_params['learning_rate'], weight_decay=net_params['weight_decay'])
    record = {
        'loss': [],
        'map': []
    }

    trainData, testData = getTrainDataAndTestData(srcPath, net_params)
    dataBaseData = getDataBaseData(srcPath, net_params)

    # train and test
    train(model, trainData, testData, net_params, optimizer, record)

    # eval
    model.eval()
    qB = Pre.GenerateCode(model, testData['dataloader'], testData['num'], net_params['bit'], net_params['use_gpu'])
    dB = Pre.GenerateCode(model, dataBaseData['dataloader'], dataBaseData['num'], net_params['bit'],
                          net_params['use_gpu'])

    map = CalcHR.CalcMap(qB, dB, testData['one_hots'].numpy(), dataBaseData['one_hots'].numpy())
    print('[Retrieval Phase] MAP(retrieval database): %3.5f' % map)

    result = {
        'map record': record['map'],
        'map': map,
        'loss_record': record['loss']
    }

    return result
