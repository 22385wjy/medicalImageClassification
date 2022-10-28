# coding: utf-8
from PIL import Image
import random
from torch.utils.data import Dataset
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

random.seed(1)
DTI_label = {"AD": 0, "NC": 1}


class MyDataset(Dataset):
    def __init__(self, img_path, transform = None, list_path=None):
        self.label_name = {"AD": 0, "NC": 1}
        self.imgs = self.get_img_info(img_path,list_path)  
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')  

        if self.transform is not None:
            img = self.transform(img)  
        return img, label

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def get_img_info(img_path,list_path):
        txt_name=list_path  #"./train_name-label"
        data_info = list()
        c0=0
        c1=0
        for root, dirs, _ in os.walk(img_path):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # print(img_names)

                for i in range(len(img_names)):
                    img_name = img_names[i]
                    # print(img_name)
                    path_img = os.path.join(root, sub_dir, img_name)
                    # print('path_img: ',path_img)
                    label = DTI_label[sub_dir]
                    # print('label: ',label)
                    if label==0:
                        c0=c0+1
                    elif label==1:
                        c1=c1+1

                    save_file = open(txt_name, 'a')
                    save_file.write(str(path_img) + ' ' + str(label) + '\n')

                    data_info.append((path_img, int(label)))
                    # print(data_info)

        print(' total data: ',len(data_info),'  c0/AD=',c0,'  c1/NC=',c1)
        return data_info


def transform_3():
    norm_mean = [0.4948052, 0.48568845, 0.44682974]
    norm_std = [0.24580306, 0.24236229, 0.2603115]

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        # transforms.RandomCrop(96, padding = 4),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])

    return train_transform, valid_transform, test_transform


class Logger(object):
    '''Save training process to log file with simple plot function.'''
    def __init__(self, fpath, title=None, resume=False):
        self.file = None
        self.resume = resume
        self.title = '' if title == None else title
        if fpath is not None:
            if resume:
                self.file = open(fpath, 'r')
                name = self.file.readline()
                self.names = name.rstrip().split('\t')
                self.numbers = {}
                for _, name in enumerate(self.names):
                    self.numbers[name] = []

                for numbers in self.file:
                    numbers = numbers.rstrip().split('\t')
                    for i in range(0, len(numbers)):
                        self.numbers[self.names[i]].append(numbers[i])
                self.file.close()
                self.file = open(fpath, 'a')
            else:
                self.file = open(fpath, 'w')

    def set_names(self, names):
        if self.resume:
            pass
        # initialize numbers as empty list
        self.numbers = {}
        self.names = names
        for _, name in enumerate(self.names):
            self.file.write(name)
            self.file.write('\t')
            self.numbers[name] = []
        self.file.write('\n')
        self.file.flush()


    def append(self, numbers):
        assert len(self.names) == len(numbers), 'Numbers do not match names'
        for index, num in enumerate(numbers):
            self.file.write("{0:.6f}".format(num))
            self.file.write('\t')
            self.numbers[self.names[index]].append(num)
        self.file.write('\n')
        self.file.flush()

    def plot(self, names=None):
        names = self.names if names == None else names
        numbers = self.numbers
        for _, name in enumerate(names):
            x = np.arange(len(numbers[name]))
            plt.plot(x, np.asarray(numbers[name]))
        plt.legend([self.title + '(' + name + ')' for name in names])
        plt.grid(True)

    def close(self):
        if self.file is not None:
            self.file.close()
