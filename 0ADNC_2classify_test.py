# coding: utf-8
from PIL import Image
import random
from torch.utils.data import Dataset
import os
from torchvision import datasets, transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil
from pywt import dwt2, idwt2, wavedec2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import make_scorer,f1_score
from MydataPreprocess.Dataset_ADNC_2c import *
from ADNC_2classify import *


def test_dataLoader(imgPath):
    test_dir = os.path.join(imgPath, "test")
    norm_mean = [0.4948052, 0.48568845, 0.44682974]
    norm_std = [0.24580306, 0.24236229, 0.2603115]
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(norm_mean, norm_std)
    ])
    test_data = MyDataset(img_path=test_dir, transform=test_transform, list_path=imgPath+"test_name-label")

    # DataLoader
    test_loader = DataLoader(dataset=test_data, batch_size=64)
    return test_loader


def TestforClassfy(model,test_loader,p1):
    mean_acc = 0.
    correct_test = 0.
    total_test = 0.

    save_file2 = open(p1 + 'Acc_log1.txt', 'a')
    save_file2.write("iter" + ' ' + "correct_number" + ' ' + "total_number" + ' ' + "mean_acc" + '\n')
    
    for iter in range(20):
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            
            total_test += labels.size(0)
            correct_test += (predicted == labels).squeeze().sum().cpu().numpy()

        mean_acc += correct_test / total_test
        save_file2.write(
            str(iter) + ' ' + str(correct_test) + ' ' + str(total_test) + ' ' + str(correct_test / total_test) + '\n')

    print("test mean Acc: {:.4f}".format(mean_acc / 20))
    save_file2.write(str(mean_acc) + '\n' + '\n')
    save_file2.write(str(mean_acc / 20) + '\n')


if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('device==',device)

    imgPath = "./ADdata_DTI/ADandNC_2data/"
    test_loader = test_dataLoader(imgPath)

    p1 = "./Out_Mynet/ADNC_Classify/"
    p2 = p1 + "trained_model/"
    model = myNet()
    model_path=p2+"best_model.pth"
    #checkpoint = torch.load(model_path) 
    checkpoint = torch.load(model_path,map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    TestforClassfy(model,test_loader,p1)



    


