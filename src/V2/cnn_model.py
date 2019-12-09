
'''
CNN模型
'''
import numpy as np
from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import logging
class CNN(nn.Module):
        def __init__(self, batch_size, factor_num, cols, args):
            kernel_size = args[0]
            out_channels = args[1]
            dropout_rate = args[2]
            in1 = args[3]
            in2 = args[4]
            in3 = args[5]
            super(CNN, self).__init__()
            # 卷积
            conv = torch.nn.Conv2d(in_channels=1,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                stride=1,
                                padding=0,
                                bias=True)
            RELU = torch.nn.ReLU()
            dropout = torch.nn.Dropout(p = dropout_rate)
            # xavier初始化
            # conv.weight = init.xavier_uniform_(conv.weight)
            # conv.weight = init.xavier_normal_(conv.weight)
            bn2d = nn.BatchNorm2d(10)  # 传入通道数
            self.layer0 = nn.Sequential(conv, bn2d, RELU)
            # init.constant(self.conv1.bias, 0.1)
            # self.batch_size = batch_size
            # 卷积后
            fc1 = nn.Linear((factor_num - kernel_size + 1) * out_channels, in1)
            fc2 = nn.Linear(in1, in2)
            fc3 = nn.Linear(in2, in3)
            fc4 = nn.Linear(in3, 2)
            # 初始化
            # fc1.weight = init.normal_(fc1.weight)
            # fc2.weight = init.normal_(fc2.weight)
            # fc3.weight = init.normal_(fc3.weight)
            # fc4.weight = init.normal_(fc4.weight)
            self.layer1 = nn.Sequential(fc1,  RELU)
            self.layer2 = nn.Sequential(fc2,  RELU)
            self.layer3 = nn.Sequential(fc3,  RELU)
            self.layer4 = nn.Sequential(fc4)
            # self.sigmoid = nn.Sigmoid()
            # Batch Normalization层,因为输入是有高度H和宽度W的,所以这里用2d
            
            # layer1.weight.data.normal_()
            # self.tanh = nn.Hardtanh()
            # self.softsign = nn.Softsign()
            self.softmax = nn.Softmax(dim=1)
            logging.basicConfig(filename='cnn.log', level=logging.DEBUG)
        def forward(self, x):
            out = self.layer0(x)
            # 需要考虑size问题
            # out = out.view(out.shape[0], -1)
            out = torch.reshape(out, (out.shape[0], -1))
            out = self.layer1(out)
            # logging.info(self.layer0[0].weight)
            
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            
            # out = self.sigmoid(out)
            out = self.softmax(out)
            # print(out)
            return out