'''
合并数据，生成因子列表
'''
# 拼整数据，对齐日期
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import os
import math
from multiprocessing import Queue, Process
from sklearn.preprocessing import normalize

def read_data():
    for i in range(0, factor_num):
        df = pd.read_csv('./source_data/median/' + str(i) + '.csv', header=None)
        df = df.values
        if factors.shape[0] == 0:
            factors = df
        else:
            factors = np.row_stack((factors, df))
    factors = factors.reshape(factor_num, day_num, -1)

def process_stock(stock):
    # 每只股票占的行数
    day_num = 169
    '''
    因子个数
    '''
    factor_num = 17
    delete_index = np.array([3, 4, 6, 10, 23, 27, 28, 29, 9, 15, 16, 17, 18])
    delete_index = [x + 8 for x in delete_index]
    delete_index = np.append(delete_index, [0, 2, 3, 4, 5, 6, 7])
    df = pd.read_csv('./source_data/split/' + stock + '.csv')
    date = df['date']
    date = date.values
    names = df.columns.values
    names = np.delete(names, delete_index, axis=0)
    names = names.reshape(1, -1)
    df = pd.read_csv('./source_data/clean_data/' + stock + '.csv',  header=None)
    df = df.values
    
    # 归一化
    for i in range(0, factor_num):
        col = df[:, i]
        std = np.std(col, axis=0)
        # 避免误差
        if std < 10e-8:
            std = 0
        # print(std)
        if(std == 0):
            col = [x - x for x in col]
        else:
            col = (col - np.mean(col, axis = 0)) / std
        df[:, i] = col
    # np.savetxt('test.csv', date, delimiter = ',')
    # exit()
    df = np.column_stack((date, df))
    df = pd.DataFrame(df, columns = names.tolist())
    outdir = './processed_data'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outname = stock + '.csv'
    fullname = os.path.join(outdir, outname)
    df.to_csv(fullname)
if __name__ == '__main__':
    # 中位数去极值
    stock_index = pd.read_csv('./source_data/index.csv')
    # 股票索引
    stock_index = stock_index['index']
    stock_index = stock_index.values

    epoch = 16
    for i in range(0, int(stock_index.shape[0] / epoch) + 1):
        p_arr = []
        for j in range(0, epoch):
            if i * epoch + j >= stock_index.shape[0]:
                break
            p = Process(target = process_stock, args=([stock_index[i * epoch + j]]))
            p_arr.append(p)
            p.start()
        for iter in p_arr:
            iter.join()
    