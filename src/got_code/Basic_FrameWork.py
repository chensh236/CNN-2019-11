import os
import pandas as pd
import numpy as np
from CNN_Model import *

class Basic_FrameWork:
    def __init__(self):
        self.Set_Parameter()

    def Set_Parameter(self):
        self.Train_line = 6*12
        self.Test_line = 12
        self.Stocks_Num = 1102
        self.Time = 169
        self.Feature_Num = 17
        self.W = 5
        self.CNNnet_Model = CNNnet_Model()

    def Get_Data(self):
        print('<Get_Data>')
        self.Input = np.loadtxt(open("./my_data/Input.csv","rb"), delimiter=",", skiprows=0)
        self.Target = np.loadtxt(open("./my_data/Target.csv","rb"), delimiter=",", skiprows=0)
        self.Input = self.Input.reshape(self.Stocks_Num, 169, -1)
        self.Target = self.Target.reshape(self.Stocks_Num, 169, -1)
        print('Input:'+str(self.Input.shape))
        print('Target:'+str(self.Target.shape))

        # 正则化
        # for T in range(self.Time):
        #     for F in range(self.Feature_Num):
        #         D = self.Input[:,T,F]
        #         D = D.reshape(-1)
        #         D_mean = (D.sort())[self.Stocks_Num/2]
        #         D_std = np.std(D)
        #         D = (D-D_mean)/D_std
        #         for S in range(self.Stocks_Num):
        #             self.Input[S,T,F] = D[S]

    def Get_Train_Data(self, base):
        print('<Get_Train_Data>')
        self.Train_X = ''
        self.Train_Y = ''
        for T in range(base+self.W, base+self.Train_line):
            # print('finish:%s' %((T - base-self.W) / (base+self.Train_line)))
            for S in range(self.Stocks_Num):
            # for S in range(10):
                x = self.Input[S:S+1,T-self.W: T,:]
                y = self.Target[S:S+1, T, :]
                self.Train_X = x if type(self.Train_X)==str else np.vstack((self.Train_X, x))
                self.Train_Y = y if type(self.Train_Y)==str else np.vstack((self.Train_Y, y))
        self.Train_X, self.Train_Y = self.Delete_NULL(self.Train_X, self.Train_Y)
        self.Train_X = self.Train_X.reshape(self.Train_X.shape[0], 1, self.Train_X.shape[1], self.Train_X.shape[2])
        self.Train_Y = self.Train_Y.reshape(-1).astype(int)
        print('Train_X:'+str(self.Train_X.shape))
        print('Train_Y:'+str(self.Train_Y.shape))
    
    # 用于回测
    def Get_Simple_Data(self, month_base):
        print('<Get_Simple_Data>')
        self.Test_X = ''
        self.Test_Y = ''
        for S in range(self.Stocks_Num):
            x = self.Input[S:S+1,month_base-self.W: month_base,:]
            self.Test_X = x if type(self.Test_X)==str else np.vstack((self.Test_X, x))
        self.Test_X = self.Test_X.reshape(self.Test_X.shape[0], 1, self.Test_X.shape[1], self.Test_X.shape[2])
        print('Test_X:'+str(self.Test_X.shape))

    def Get_Test_Data(self, base):
        print('<Get_Test_Data>')
        self.Test_X = ''
        self.Test_Y = ''
        for T in range(base+self.W+self.Train_line, base+self.Train_line+self.Test_line):
            for S in range(self.Stocks_Num):
                x = self.Input[S:S+1,T-self.W: T,:]
                y = self.Target[S:S+1, T, :]
                self.Test_X = x if type(self.Test_X)==str else np.vstack((self.Test_X, x))
                self.Test_Y = y if type(self.Test_Y)==str else np.vstack((self.Test_Y, y))
        self.Test_X, self.Test_Y = self.Delete_NULL(self.Test_X, self.Test_Y)
        self.Test_X = self.Test_X.reshape(self.Test_X.shape[0], 1, self.Test_X.shape[1], self.Test_X.shape[2])
        self.Test_Y = self.Test_Y.reshape(-1).astype(int)
        print('Test_X:'+str(self.Test_X.shape))
        print('Test_Y:'+str(self.Test_Y.shape))

    def Delete_NULL(self, X, Y):
        tmp_Y = pd.DataFrame(Y)
        null_arr = tmp_Y.isnull()
        null_arr = null_arr.values
        delete_arr = np.array([])
        delete_arr = np.array([])
        for i in range(null_arr.shape[0]):
            if (null_arr[i] == True):
                delete_arr = np.append(delete_arr, i)
        delete_array = np.trunc(delete_arr)
        Y = np.delete(Y, delete_arr, axis=0)
        X = np.delete(X, delete_arr, axis=0)
        return X, Y

    def Train(self, base, load_model, Train):
        print('<Train>')
        if not Train:
            self.CNNnet_Model.fit([], [], base, load_model)
            return
        self.CNNnet_Model.fit(self.Train_X, self.Train_Y, base, load_model)
        Y = self.CNNnet_Model.predict(self.Train_X)
        print(Y.shape)
        print(self.Train_Y.shape)
        count = 0
        for i in range(Y.shape[0]):
            if (Y[i] == self.Train_Y[i]):
                count += 1
        count = count/Y.shape[0]
        print('Train Hr' + str(base) + ':  '+str(count))
        # np.savetxt('Train Hr_' + str(base) + '.txt', count, delimiter=',')
        # print(Y)
        # print(self.Train_Y)
        # 画图
        plt.plot(Y)
        plt.plot(self.Train_Y)
        plt.title('Train')
        plt.savefig('./Train_' + str(base) + '.png')
        plt.cla()
        # plt.show()

    def Test(self, base):
        print('<Test>')
        Y = self.CNNnet_Model.predict(self.Test_X)
        count = 0
        for i in range(Y.shape[0]):
            if (Y[i] == self.Test_Y[i]):
                count += 1
        count = count/Y.shape[0]
        print('Test Hr' + str(base) + ':  '+str(count))
        # np.savetxt('Test Hr_' + str(base) + '.txt', count, delimiter=',')
        # print(Y)
        # print(self.Test_Y)
        # 画图
        plt.plot(Y)
        plt.plot(self.Test_Y)
        plt.title('Test')
        plt.savefig('./Test_' + str(base) + '.png')
        plt.cla()

    def Test_Simple(self, base):
        print('<Test_simple>')
        self.CNNnet_Model.fit([], [], base, True)
        Y = self.CNNnet_Model.predict_simple(self.Test_X)
        return Y