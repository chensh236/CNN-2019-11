'''
获取收益率
'''
# 拼整数据，对齐日期
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import os
import math
from multiprocessing import Queue, Process
from sklearn.preprocessing import normalize

def process_stock(stock):
    # 每只股票占的行数
    day_num = 169
    df = pd.read_csv('./source_data/split/' + stock + '.csv')
    date = df['date']
    date = date.values
    y = df['pct_chg']
    y = y.values
    for i in range(0, y.shape[0]):
        if math.isnan(y[i]):
            continue
        elif y[i] > 0:
            y[i] = 1
        else:
            y[i] = 0
    y = np.column_stack((date, y))
    df = pd.DataFrame(y, columns = ['date', 'y'])
    outdir = './y_data'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outname = stock + '.csv'
    fullname = os.path.join(outdir, outname)
    df.to_csv(fullname)
if __name__ == '__main__':
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