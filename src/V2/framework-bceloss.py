'''
训练与测试框架
'''
import time
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Queue, Process
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cnn_model import *
import logging
class framework():
    # num_epochs 测试的epoch数量
    # learning_rate 学习率
    # batch_size batch大小
    # factor_num 因子数量
    # cols 输入矩阵的列数（日期数量）
    # 模型数量
    def __init__(self, num_epochs, learning_rate, batch_size, factor_num, cols, model_num, args):
        # num_epochs = 100
        # learning_rate = 0.01
        # batch_size = 1
        # factor_num = 17
        # cols = 5
        # model_num = ?
        # 设置参数
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.factor_num = factor_num
        self.cols = cols
        self.model_num = model_num
        ### 设置模型
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 多个模型的类
        self.model_arr = np.array([])
        for i in range(0, model_num):
            model = CNN(self.batch_size, self.factor_num, self.cols, args).to(device)
            self.model_arr = np.append(self.model_arr, model)
        # logging.basicConfig(filename='framework.log', level=logging.DEBUG)
    # 去除空的数据
    def delete_null(self, X, Y):
        tmp_Y = pd.DataFrame(Y)
        null_arr = tmp_Y.isnull()
        null_arr = null_arr.values
        delete_arr = np.array([])
        for i in range(0, null_arr.shape[0]):
            if null_arr[i] == True:
                delete_arr = np.append(delete_arr, i)
        delete_arr = np.trunc(delete_arr)
        # print(delete_arr)
        Y = np.delete(Y, delete_arr, axis=0)
        X = np.delete(X, delete_arr, axis=0)
        return X, Y

    def data_input(self, Train_X, Train_Y, Test_X, Test_Y, train_set_num, test_set_num, stock_num):
        # 对数据进行重构
        Train_X = Train_X.values
        Train_Y = Train_Y.values
        Test_X = Test_X.values
        Test_Y = Test_Y.values
        Test_Y = np.array(Test_Y, dtype = float)
        Train_Y = np.array(Train_Y, dtype = float)
        Train_X = Train_X.reshape(stock_num * (train_set_num - self.cols), self.factor_num, self.cols)
        Train_Y = Train_Y.reshape(-1, 1)
        Test_Y = Test_Y.reshape(-1, 1)
        Test_X = Test_X.reshape(stock_num * test_set_num, self.factor_num, self.cols)
        # print('number X: %s %s'%(Train_X.shape, Test_X.shape))    
        # print('number Y: %s %s'%(Train_Y.shape, Test_Y.shape))
        # 去除null
        Train_X, Train_Y = self.delete_null(Train_X, Train_Y)
        Test_X, Test_Y = self.delete_null(Test_X, Test_Y)

        # 重置为四维数组，conv只支持四维数组
        Train_X = Train_X.reshape(-1, 1, self.factor_num, self.cols)
        Train_Y = Train_Y.reshape(-1, 1)
    
        Test_Y = Test_Y.reshape(-1, 1)
        Test_X = Test_X.reshape(-1, 1, self.factor_num, self.cols)
        # print(Train_Y.dtype)
        # print('number X: %s %s'%(Train_X.shape, Test_X.shape))    
        # print('number Y: %s %s'%(Train_Y.shape, Test_Y.shape))
        # model_num 对于每个股票，可以训练多个模型,默认为1
        self.Train_X = Train_X
        self.Train_Y = Train_Y
        self.Test_X = Test_X
        self.Test_Y = Test_Y
        
        # todo: reshape
        # (batch_size, channels, height, width)
        pass
    def train(self, model_index, show_predict_y):
        # 用于在线显示loss
        viz = Visdom()
        # criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.RMSprop(model.parameters(),lr=learning_rate,alpha=0.9)
        optimizer = torch.optim.RMSprop(self.model_arr[model_index].parameters(),lr=self.learning_rate)
        # 动态调整学习率
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, num_epochs, gamma=0.1)
        # Train_X_arr = np.random.rand(500, 1, factor_num, cols)
        # Train_Y_arr = np.random.rand(500, 1, 1)
        Train_X_arr = torch.from_numpy(self.Train_X).float()
        Train_Y_arr = torch.from_numpy(self.Train_Y).float()
        # Train_Y_arr = Variable(torch.LongTensor(self.Train_Y))
        '''
        注意：如果没有使用cuda将其注释
        '''
        Train_X_arr = Train_X_arr.cuda()
        Train_Y_arr = Train_Y_arr.cuda()
        train_data = TensorDataset(Train_X_arr, Train_Y_arr)
        train_data = torch.utils.data.DataLoader(dataset = train_data, batch_size = self.batch_size, shuffle=True, pin_memory = False)

        print('-----------begin to train----------')
        train_loss = []
        # 用于显示loss
        # total_step = len(train_data)
        # step_arr = np.array([])
        # # 用于visdom显示
        # for i in range(self.num_epochs):
        #     step_arr = np.append(step_arr, i)
        win_name = 'loss_model index:' + str(model_index)
        # 显示loss
        viz.line([[0.]], [0], win = win_name, opts=dict(title= win_name, legend=['loss']))
        q = 0
        for epoch in range(self.num_epochs):
            for i, (x, y) in enumerate(train_data):
                #forward
                outputs = self.model_arr[model_index](x)
                # BP
                y = y.squeeze(1)
                outputs = outputs.squeeze(1)
                if show_predict_y:
                    print(y)
                # outputs = outputs.squeeze() 
                # # y.shape
                # exit()
                # outputs = torch.t(outputs)
                # print(outputs)
                # exit()
                loss = criterion(outputs, y)
                # print(loss.data[0])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                viz.line([[loss.item()]], [q], win = win_name, update='append')
                q = q + 1
            # 查看每一epoch的权重变化和loss
            current_loss = loss.item()
            train_loss.append(current_loss)
            # params = list(model.named_parameters())
            # print(model.state_dict().keys())
            # scheduler.step()
            # print('learning rate: %s'%(optimizer.param_groups[0]['lr']))
            if ((epoch+1) % 10 == 0):
                print ('Epoch [{}/{}], Loss: {:.8f}' 
                    .format(epoch+1, self.num_epochs,loss.item()))

        print('Loss: {:.8f}'.format(loss.item()))
        # 图像显示与保存
        # plt.subplot(1,1,1)
        # plt.plot(train_loss[50:])
        # plt.show()
        self.model_arr[model_index].eval()
    
    def test_thread(self, test_in_train, model_index, test_rate):
        X = np.array([])
        Y = np.array([])
        if(test_in_train):
            print('######## test in  train set ########')
            X = self.Train_X
            Y = self.Train_Y
        else:
            print('######## test ########')
            X = self.Test_X
            Y = self.Test_Y
        predict = []
        real = []
        if test_rate >= 1.0:
            test_rate = int(1)
        if test_rate < 0.001:
            print('too less data!!!')
            exit()
        for iter in range(0, int(X.shape[0] * test_rate)):
            x = X[iter]
            x = x.reshape(1, 1, self.factor_num, self.cols)
            x = (torch.from_numpy(x)).float()
            x = x.cuda()
            y_predict = self.model_arr[model_index](x)
            y_real = Y[iter]
            predict.append(y_predict.item())
            # print(predict[-1])
            # exit()
            real.append(y_real)
            # eval
        hr = 0.0
        predict = np.array(predict)
        for i in range(0, predict.shape[0]):
            if predict[i] > 1.0 - predict[i]:
                predict[i] = 1
            else:
                predict[i] = 0
        real = np.array(real)
        for i in range(1, len(predict)):
            if (predict[i]*real[i] > 0):
                hr += 1
        hr = hr/len(predict)
        if test_in_train:
            print('test in train set:     hr:%s'%(hr))
        else:
            print('test in test set       hr:%s'%(hr))
    def test(self, model_index, test_rate):
        # p_train = Process(target = self.test_thread, args=([True]))
        # p_test = Process(target = self.test_thread, args=([False]))
        # p_train.start()
        # p_test.start()
        # p_train.join()
        # p_test.join()
        self.test_thread(True, model_index, test_rate)
        self.test_thread(False, model_index, test_rate)
    def save(self, model_index, name):
       torch.save(self.model_arr[model_index].state_dict(), name + '.pkl')
    def load(self, model_index, name):
        self.model_arr[model_index].load_state_dict(torch.load(name + '.pkl'))