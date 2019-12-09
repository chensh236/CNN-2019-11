'''
生成训练集和测试集
'''
from framework import *
from cnn_model import *
import numpy as np
from pandas import Series, DataFrame
import pandas as pd


# 生成随机数进行测试
# 30因子
# 1000个时间节点
class Crate_dataset():
    def __init__(self):
        pass

    def read_data(self, start_index, train_set_num, test_set_num, cols, factor_num):
        Train_X = np.array([])
        Train_Y = np.array([])
        Test_X = np.array([])
        Test_Y = np.array([])
        # 股票索引
        stock_index = pd.read_csv('./source_data/index.csv')
        stock_index = stock_index['index']
        stock_index = stock_index.values
        for i in range(0, stock_index.shape[0]):
        # for i in range(0, 1):
            # print('get data in stock:%s'%(i))
            print('finish:%s'%(i/stock_index.shape[0]))
            # 获得因子
            factor_table = pd.read_csv('./processed_data/' + stock_index[i] + '.csv')
            y_table = pd.read_csv('./y_data/' + stock_index[i] + '.csv')
            factor_table = factor_table[['BIAS', 'MACD', 'RSI', 'close', 'dividendyield2', 'grossprofitmargin_ttm', 'pb_lyr', 'pcf_ncf_ttm', 'pcf_ocf_ttm', 'pe_ttm', 'roa_ttm', 'roe_ttm', 'tech_psy', 'turn', 'val_lnmv', 'val_ortomv_ttm', 'val_pe_deducted_ttm']]
            factor_table = factor_table.values
            # 获得y
            y_table = y_table['y']
            y_table = y_table.values
            # y_table = y_table.astype(int)
            # Rt - 5, Rt - 4, Rt - 3, Rt - 2, Rt - 1预测Rt
            Train_Y = np.row_stack((Train_Y, y_table[start_index + cols + 1: start_index + train_set_num + 1])) if Train_Y.shape[0] != 0 else y_table[start_index + cols + 1: start_index + train_set_num + 1]
            # print(Train_Y)
            Test_Y = np.row_stack((Test_Y, y_table[start_index + train_set_num + 1: start_index + train_set_num + test_set_num + 1]))  if Test_Y.shape[0] != 0 else y_table[start_index + train_set_num + 1 : start_index + train_set_num + test_set_num + 1]
            # 获得x
            for i in range(start_index + cols, start_index + train_set_num):
                tmp = factor_table[i - cols : i, :]
                # print(tmp.T)
                Train_X = np.row_stack((Train_X, tmp.T)) if Train_X.shape[0] != 0 else tmp.T
            for i in range(start_index + train_set_num, start_index + train_set_num + test_set_num):
                tmp = factor_table[i - cols : i, :]
                Test_X = np.row_stack((Test_X, tmp.T)) if Test_X.shape[0] != 0 else tmp.T
        # print('number X: %s %s'%(Train_X.shape, Test_X.shape))    
        # print('number Y: %s %s'%(Train_Y.shape, Test_Y.shape))
        return Train_X, Train_Y, Test_X, Test_Y

    def read(self, year, train_set_num, test_set_num, cols, factor_num):
        start_index = 0
        for i in range(2011, year):
            start_index = start_index + 12
        return self.read_data(start_index, train_set_num, test_set_num, cols, factor_num)
if __name__ == '__main__':
    c = Crate_dataset()
    c.read(2011, 72, 12, 5, 17)