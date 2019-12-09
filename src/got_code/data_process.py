import os
import pandas as pd
import numpy as np

class Handler:
    def __init__(self):
        pass

    def Set_Parameter(self):
        self.Train_line = 150
        

    def Get_Data(self):
        filePath = './processed_data/'
        self.Stocks_csv = []
        for i,j,k in os.walk(filePath):
            self.Stocks_csv = k
        self.Stocks_Num = len(self.Stocks_csv)
        print('Stocks num:'+str(self.Stocks_Num))

        self.Input = ''
        self.Target = ''
        for stocks_csv in self.Stocks_csv:
            Input = pd.read_csv('./processed_data/'+stocks_csv)
            Target = pd.read_csv('./y_data/'+stocks_csv)
            Input = np.array(Input)[:,2:]
            Target = np.array(Target)[:,2:]
            self.Input = Input if type(self.Input)==str else np.vstack((self.Input, Input))
            self.Target = Target if type(self.Target)==str else np.vstack((self.Target, Target))
        self.Feature_Num = self.Input.shape[1]
        self.Length = self.Input.shape[0]
        print('Input:'+str(self.Input.shape))
        print('Target:'+str(self.Target.shape))
        # self.Input = self.Input.reshape(self.Stocks_Num, Input.shape[0], -1)
        # self.Target = self.Target.reshape(self.Stocks_Num, Target.shape[0], -1)
        # print('Input:'+str(self.Input.shape))
        # print('Target:'+str(self.Target.shape))

        # np.loadtxt(open("./my_data/Input.csv","rb"), delimiter=",", skiprows=0)
        np.savetxt("./my_data/Input.csv", self.Input, delimiter=',')
        np.savetxt("./my_data/Target.csv", self.Target, delimiter=',')
        


if __name__ == '__main__':
    F = Handler()
    F.Get_Data()