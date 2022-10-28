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
from MydataPreprocess.Dataset_ADNC_2c import *
import shutil
from pywt import dwt2, idwt2, wavedec2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

class WaveletTransformUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        # bn_layer not used
        super(WaveletTransformUnit, self).__init__()
        self.groups = groups
        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.convW = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bnW = nn.BatchNorm2d(256)

    def forward(self, x):
        x= x.cpu()
        x=x.detach().numpy()
        wave = 'haar'  
        cA, (cH, cV, cD) = dwt2(x, wave)
        # gao=torch.cat((cH, cV, cD),1)
        cA=torch.from_numpy(cA).to(device)
        cA = self.relu(self.bn(self.conv_layer(cA)))

        # print('cA- ',cA.shape)
        cH = self.relu(self.bn(self.conv_layer(torch.from_numpy(cH).to(device))))
        cV = self.relu(self.bn(self.conv_layer(torch.from_numpy(cV).to(device))))
        cD = self.relu(self.bn(self.conv_layer(torch.from_numpy(cD).to(device))))

        cA = cA.cpu().detach().numpy()
        cH = cH.cpu().detach().numpy()
        cV = cV.cpu().detach().numpy()
        cD = cD.cpu().detach().numpy()

        rimg0 = idwt2((cA, (cH, cV, cD)), wave)
        rimg0 =torch.from_numpy(rimg0).to(device)
        # print('rimg0- ', rimg0.shape)

        cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1) = wavedec2(x, wave, level=3, )
        # print('cA3- ',cA3.shape)
        rimg3 = idwt2((cA3, (cH3, cV3, cD3)), wave)
        rimg2 = idwt2((rimg3, (cH2, cV2, cD2)), wave)
        rimg1 = idwt2((rimg2, (cH1, cV1, cD1)), wave)
        rimg1 =torch.from_numpy(rimg1).to(device)
        # print('rimg1- ', rimg1.shape)

        rimg = torch.cat((rimg0, rimg1), 1)
        rimg = self.relu(self.bnW(self.convW(rimg)))

        return rimg

class newClassifier(nn.Module):
    def __init__(self, classCount = 2, p=0.5):
        super(newClassifier, self).__init__()
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(p)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.bn1= nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.WTU = WaveletTransformUnit(in_channels=256, out_channels=256)
        self.convW = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.bnW = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(32)

        self.classifierXray = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, classCount),
        )

    def forward(self, x):
        # print(x.shape)  #[32, 3, 128, 128]
        # print('self.conv1(x): ',(self.conv1(x)).shape)  #[32, 64, 64, 64]
        x1=self.act(self.bn1(self.conv1(x)))
        # print('1- ',x1.shape)
        x2=self.act(self.bn2(self.conv2(x1)))
        # print('2- ',x2.shape)
        x3=self.act(self.bn3(self.conv3(x2)))
        # print('3- ',x3.shape)

        wtu=self.WTU(x3)
        # print('wtu1- ',wtu.shape)
        # wtu = self.bnW(self.act(self.convW(wtu)))
        # print('wtu2- ',wtu.shape)

        x4 = self.act(self.bn4(self.conv4(wtu)))
        # print('4- ',x4.shape)
        x5 = self.act(self.bn5(self.conv5(x4)))
        # print('5- ',x5.shape)
        x6 = self.act(self.bn6(self.conv6(x5)))
        # print('6- ',x6.shape)

        x = x6.view(-1, 128 * 2 * 2)
        # print('after x.view --- ',x.shape)
        x = self.classifierXray(x)
        # print('after classifierXray --- ',x.shape)
        out=torch.softmax(x, dim=1)
        # print('out: ',out)
        # return torch.sigmoid(x)
        return out

def myNet():
    mynet = newClassifier().to(device)
    mynet=mynet.float()
    return mynet


def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, p2+'best_model.pth')

def three_dataLoader(imgPath):
    train_dir = os.path.join(imgPath, "train")
    # print('train_dir--',train_dir)
    valid_dir = os.path.join(imgPath, "val")
    test_dir = os.path.join(imgPath, "test")

    train_transform, valid_transform, test_transform = transform_3()
    train_data = MyDataset(img_path=train_dir, transform=train_transform,
                           list_path=imgPath+"train_name-label")
    valid_data = MyDataset(img_path=valid_dir, transform=valid_transform, list_path=imgPath+"val_name-label")
    test_data = MyDataset(img_path=test_dir, transform=test_transform, list_path=imgPath+"test_name-label")

    # DataLoader
    train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_data, batch_size=64)
    return train_loader,valid_loader,test_loader

def TrainforClassfy(model,train_loader,valid_loader,test_loader,optimizer,scheduler,criterion,p1,p2):
    logger = Logger(os.path.join(p1, 'log.txt'), title='2 classes predict')
    logger.set_names(
        ['Epoch', 'Train Loss', 'Train Acu', 'Valid Loss', 'Valid Acu'])
    writer = SummaryWriter(p1 + 'add_scalar_log')

    # train
    train_curve = list()
    valid_curve = list()
    best_ACC = 0
    train_acc = 0
    val_acc = 0
    val_Acc = []
    for epoch in range(MAX_EPOCH):
        loss_mean = 0.
        correct = 0.
        total = 0.
        model.train()
        for idx, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().sum().cpu().numpy()

            # print
            loss_mean += loss.item()
            train_curve.append(loss.item())
            train_acc = correct / total
            if (idx + 1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, idx + 1, len(train_loader), loss_mean, train_acc))
                loss_mean = 0.

            writer.add_scalar('losses/train_loss', train_curve[-1], epoch)
            writer.add_scalar('accuracy/train_acc', train_acc, epoch)

        scheduler.step()  # update learning rate

        # validation
        # model.eval()
        if (epoch + 1) % val_interval == 0:
            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            model.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().sum().cpu().numpy()

                    loss_val += loss.item()

                valid_curve.append(loss_val / valid_loader.__len__())
                val_acc = correct_val / total_val
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j + 1, len(valid_loader), loss_val, val_acc))

            writer.add_scalar('losses/valid_loss', valid_curve[-1], epoch)
            writer.add_scalar('accuracy/valid_acc', val_acc, epoch)

        is_best = val_acc > best_ACC
        best_ACC = max(val_acc, best_ACC)
        print('best_ACC: ', best_ACC)

        save_file = open(p1 + 'log.txt', 'a')
        save_file.write(
            str(epoch) + ' ' + str(train_curve[-1]) + ' ' + str(train_acc) + ' ' + str(valid_curve[-1]) + ' ' + str(
                val_acc) + '\n')

        # save model
        torch.save(model.state_dict(), p2 + "DTI.pth")
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': val_acc,
            'best_acc': best_ACC,
            'optimizer': optimizer.state_dict(),
        }, is_best, filename=os.path.join(p2, 'model.th'))
        
        val_Acc.append(val_acc)
        logger.close()
        writer.close()

        indx = np.argmax(val_Acc)
        print('Best Val ACC: {} '.format(val_Acc[indx]))

    # test
    mean_acc = 0.
    correct_test = 0.
    total_test = 0.
    save_file2 = open(p1 + 'test_log.txt', 'a')
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
        writer.add_scalar('accuracy/test_acc', mean_acc / (correct_test / total_test), iter)

    print("test mean Acc: {:.4f}".format(mean_acc / 20))
    save_file2.write(str(mean_acc) + '\n' + '\n')
    save_file2.write(str(mean_acc / 20) + '\n')


if __name__=="__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print('device==',device)

    BATCH_SIZE = 32
    MAX_EPOCH = 40
    LR = 0.001
    log_interval = 10
    val_interval = 1
    random.seed(1)
    imgPath = "./ADdata_DTI/ADandNC_2data/"
    train_loader, valid_loader, test_loader = three_dataLoader(imgPath)

    p1 = "./Out_Mynet/ADNC_Classify/"
    p2 = p1 + "trained_model/"
    if not os.path.exists(p1):
        os.makedirs(p1)
    if not os.path.exists(p2):
        os.makedirs(p2)

    model = myNet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr= LR, momentum = 0.85)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= MAX_EPOCH/5, gamma=0.1)    
    TrainforClassfy(model,train_loader,valid_loader,test_loader,optimizer,scheduler,criterion,p1,p2)




