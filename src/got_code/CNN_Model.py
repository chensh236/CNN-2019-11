import os
import pandas as pd
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from visdom import Visdom
class CNNnet(torch.nn.Module):
    def __init__(self):
        super(CNNnet,self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=10,
                            kernel_size=(5,5),
                            stride=1,
                            padding=0),
            torch.nn.BatchNorm2d(10),
            torch.nn.ReLU()
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(130, 100),
            torch.nn.ReLU()
        )
        self.fc2 = torch.nn.Sequential(
            torch.nn.Linear(100, 70),
            torch.nn.ReLU()
        )
        self.fc3 = torch.nn.Sequential(
            torch.nn.Linear(70, 40),
            torch.nn.ReLU()
        )
        self.fc4 = torch.nn.Sequential(
            torch.nn.Linear(40, 2),
            torch.nn.Softmax(dim=1)
        )
    def forward(self, x):
        # print('forward:'+str(x.shape))
        x = self.conv1(x)
        # print('conv1->:'+str(x.shape))
        x = x.view(x.shape[0], -1)
        # print('view->'+str(x.shape))
        x = self.fc1(x)
        # print('fc1->:'+str(x.shape))
        x = self.fc2(x)
        # print('fc2->:'+str(x.shape))
        x = self.fc3(x)
        # print('fc3->:'+str(x.shape))
        x = self.fc4(x)
        # print('fc4->:'+str(x.shape))
        # x = self.mlp1(x.view(x.size(0),-1))
        return x

class CNNnet_Model:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.Model = CNNnet().to(self.device)
        self.Set_Parameter()

    def Set_Parameter(self):
        self.LR = 0.0005
        self.Batch_Size = 100
        self.Epochs = 300
    
    def fit(self, X, Y, base, load_model = False):
        if load_model:
            # # 直接load模型
            self.Model.load_state_dict(torch.load(str(base) + '.pkl'))
            # self.Model.load_state_dict(torch.load('./CNN'))
            self.Model.eval()
            return
        viz = Visdom()
        win_name = str(base) + '_loss'
        viz.line([[0.]], [0], win = win_name, opts=dict(title= win_name, legend=['loss']))
        optimizer = torch.optim.Adam(self.Model.parameters(), lr=self.LR)
        loss_func = torch.nn.CrossEntropyLoss()
        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).long()
        X = X.cuda()
        Y = Y.cuda()
        Data = TensorDataset(X, Y)
        Data = torch.utils.data.DataLoader(dataset=Data, batch_size=self.Batch_Size, shuffle=True)
        Loss = []
        for epoch in range(self.Epochs):
            epoch_train_loss = []
            for step, (x, y) in enumerate(Data):
                x = x.to(self.device)
                y = y.to(self.device)
                # Forward pass
                out = self.Model(x)
                # print('out:'+str(out.shape))
                # print('y:'+str(y.shape))
                loss = loss_func(out, y)
                epoch_train_loss.append(loss.item())
                # exit()

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                Loss.append(loss.data.item())
            epoch_train_loss = np.array(epoch_train_loss)
            current_loss = np.mean(epoch_train_loss)
            viz.line([[current_loss]], [epoch], win = win_name, update='append')
            if (epoch+1) % (self.Epochs/10) == 0:
                # plot decoded img
                # for i in range(5):
                #     plt.subplot(5, 1, i+1)
                #     plt.plot(x.detach().numpy()[0,:,i])
                #     plt.plot(decoded.detach().numpy()[0,:,i])
                # plt.show()
                print('Epoch: ', epoch+1, '| train loss: %.4f' % loss.data.item())
        print('Loss: %.4f'%(loss.data.item()))
        # np.savetxt('Loss_' + str(base) + '.txt', loss.data.item(), delimiter=',')
        # 画图
        plt.cla()
        plt.plot(Loss)
        plt.title('Loss')
        plt.savefig('./CNN_Loss_' + str(base) + '.png')
        plt.cla()
        # plt.show()
        # 保存
        torch.save(self.Model.state_dict(), str(base) + '.pkl')
        # torch.save(self.Model.state_dict(), './CNN')
    
    def predict(self, X):
        X = torch.from_numpy(X).float()
        X = X.cuda()
        out = self.Model(X)
        Y = []
        out = out.cpu().detach()
        out = [t.numpy() for t in out]
        out = np.array(out)
        for i in range(out.shape[0]):
            Y.append(0)
            if out[i, 1] > out[i, 0]:
                Y[-1] = 1
            # for j in range(1, out.shape[1]):
            #     if (out[i,j] > out[i,Y[-1]]):
            #         Y[-1] = j
        return np.array(Y)

    def predict_simple(self, X):
        X = torch.from_numpy(X).float()
        X = X.cuda()
        out = self.Model(X)
        out = out.cpu().detach()
        out = [t.numpy() for t in out]
        out = np.array(out)
        Y = out[:, 1]
        return np.array(Y)
