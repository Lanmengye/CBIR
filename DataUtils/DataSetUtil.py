import torch
import os
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms

transformations = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class MNIST_DS(Dataset):
    def __init__(self, srcPath, imgFileNamesPath, labelFilePath, transform=transformations):
        self.img_path = srcPath
        self.transform = transform

        img_filepath = os.path.join(srcPath, imgFileNamesPath)
        with open(img_filepath, 'r') as f:
            self.img_filename = [x.strip() for x in f]

        label_filepath = os.path.join(srcPath, labelFilePath)
        with open(label_filepath, 'r')as f:
            labels = [int(x.strip()) for x in f]
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([self.label[index]])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


class CIFAR_10_DS(Dataset):
    def __init__(self, data_path, imgFileNames, labelFileNames, transform=transformations):
        self.img_path = data_path
        self.transform = transform

        img_filepath = os.path.join(data_path, imgFileNames)
        with open(img_filepath, 'r') as f:
            self.img_filename = [x.strip() for x in f]

        label_filepath = os.path.join(data_path, labelFileNames)
        with open(label_filepath, 'r')as f:
            labels = [int(x.strip()) for x in f]
        self.label = labels

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.LongTensor([self.label[index]])
        return img, label, index

    def __len__(self):
        return len(self.img_filename)
