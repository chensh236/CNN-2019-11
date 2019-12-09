'''
清洗数据
'''
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
import os
import math
from multiprocessing import Queue, Process
from sklearn.preprocessing import normalize

def read_file(read_path, stock_index, day_num):
    # 读取数据
    df = pd.read_csv(read_path)
    df_index = 0
    while df_index < df.shape[0]:
        data = df.iloc[df_index : df_index + day_num, :]
        # 检查是否为空，如果空跳开
        # 检查内容：市值、收益率
        df_index = df_index + day_num

        # 去除有null的数据
        # ev = data['ev']
        # pct_chg = data['pct_chg']
        # if ev.isnull().T.any() == True or pct_chg.isnull().T.any() == True:
        #     print('null!%s %s %s'%(data.iloc[0, 1], ev.isnull().T.any(), pct_chg.isnull().T.any()))
        #     continue
        peace_arr = data.values
        stock_index = np.append(stock_index, [peace_arr[0, 1], peace_arr[0, 3]])
        
        # 检查
        for i in range(peace_arr.shape[0] - 1):
            if not peace_arr[i, 3] == peace_arr[i + 1, 3]:
                print('err!%s'%(peace_arr[i, 1]))
                exit()
        # 写入股票数据
        outdir = './source_data/split'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outname = str(peace_arr[0, 1]) + '.csv'
        fullname = os.path.join(outdir, outname)
        data.to_csv(fullname)
    return stock_index

# 用于清除无用值 转换为float
def clean_arr(arr):
    tmp = np.array([])
    for row in arr:
        if math.isnan(row):
            continue
        else:
            tmp = np.append(tmp, float(row))
    return tmp

def multi_median(day_num, stock_index, i):
    print('process factor:%s'%(i))
    # 对日期进行遍历
    factor_col = np.array([])
    for j in range(0, day_num):
        print('process date:%s'%(j))
        # 因子的值
        factor_content = np.array([])
        for row in stock_index:
            data = pd.read_csv('./source_data/split/' + str(row) + '.csv')
            factor_content = np.append(factor_content, data.values[j, i + 8])
        cleaned_factor_content = clean_arr(factor_content)
        # 全空
        if cleaned_factor_content.shape[0] == 0:
            factor_content = factor_content.reshape(1, -1)
            if factor_col.shape[0] == 0:
                factor_col = factor_content
            else:
                factor_col = np.row_stack((factor_col, factor_content))
            continue
        # print(cleaned_factor_content)
        DM = np.median(cleaned_factor_content)
        residue = [abs(x - DM) for x in cleaned_factor_content]
        DM1 = np.median(residue)
        for k in range(0, factor_content.shape[0]):
            if math.isnan(factor_content[k]):
                factor_content[k] = float('nan')
            elif float(factor_content[k]) > DM + 5.0 * DM1:
                factor_content[k] = DM + 5.0 * DM1
            elif float(factor_content[k]) < DM - 5.0 * DM1:
                factor_content[k] = DM - 5.0 * DM1
        factor_content = factor_content.reshape(1, -1)
        if factor_col.shape[0] == 0:
            factor_col = factor_content
        else:
            factor_col = np.row_stack((factor_col, factor_content))
    outdir = './source_data/median'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    outname = str(i) + '.csv'
    fullname = os.path.join(outdir, outname)
    np.savetxt(fullname, factor_col, delimiter=',')

def median_process(factor_num, day_num, stock_index):
    factors = np.array([])
    '''
    # 对因子进行遍历
    epoch = factor_num
    for i in range(int(factor_num / epoch)):
        p_arr = []
        for j in range(0, epoch):
            p = Process(target = multi_median, args=([day_num, stock_index, epoch * i + j]))
            p_arr.append(p)
            p.start()
        for iter in p_arr:
            iter.join()
    # multi_median(day_num, stock_index, 29)
    exit()
    '''
    for i in range(0, factor_num):
        df = pd.read_csv('./source_data/median/' + str(i) + '.csv', header=None)
        df = df.values
        if factors.shape[0] == 0:
            factors = df
        else:
            factors = np.row_stack((factors, df))
    factors = factors.reshape(factor_num, day_num, -1)
    # 因子 - 日期 - 股票
    return factors

# factors[j, k, i]   因子 - 日期 - 股票 
def fill_process(factor_num, day_num, stock_index, factors, stock_belongs):
    # 查看哪些因子出现问题
    # 设定阈值
    # threshold = 0.1
    # for i in range(0, factor_num):
    #     count = 0
    #     for j in range(0,day_num):
    #         if(np.isnan(factors[i, j, :]).T.all()):
    #             count = count + 1
    #     if count > 0:
    #         print('empty factor:%s'%(i))
    # exit()
    
    # 保护原始数据
    factor_raw = factors
    # 取所有行业的因子数据，用于填充
    factor_avg = np.array([])
    for i in range(0, factor_num):
        factor_day = np.array([])
        for j in range(0,day_num):
            # print(factor_raw[i, j, :])
            # print(np.nanmean(factor_raw[i, j, :]))
            if(np.isnan(factor_raw[i, j, :]).T.all()):
                print('nan! factor:%s day:%s'%(i, j))
            factor_day = np.append(factor_day, np.nanmean(factor_raw[i, j, :]))
        if factor_avg.shape[0] == 0:
            factor_avg = factor_day
        else:
            factor_avg = np.row_stack((factor_avg, factor_day))
    # np.savetxt('test.csv', factor_avg, delimiter = ',')
    # exit()


    for i in range(0, stock_index.shape[0]):
        print('process stock:%s'%(i))
        # 查询个股所属行业
        peers = np.array([])
        for j in range(0, stock_belongs.shape[0]):
            if(i == j):
                continue
            if(stock_belongs[i] == stock_belongs[j]):
                peers = np.append(peers, j)
        # 填充数据
        for j in range(0, factor_num):
            for k in range(0, day_num):
                if(not math.isnan(factor_raw[j, k, i])):
                    continue
                # 出现空值
                # 如果因子出现空值，不填充
                if np.isnan(factor_raw[:, k, i]).T.all():
                    continue
                peer_values = np.array([])
                for peer_stock in peers:
                    peer_values = np.append(peer_values, factor_raw[j, k, int(peer_stock)])
                mean_value = np.nanmean(peer_values)
                if np.isnan(mean_value):
                    # 处理空数据，取行业平均值
                    print('null value in factor:%s    stock:%s'%(j, i))
                    factors[j, k, i] = factor_avg[j, k]
                else:
                    factors[j, k, i] = mean_value
    return factors
    


if __name__ == '__main__':
    # 每只股票占的行数
    day_num = 169
    # 因子个数
    factor_num = 30
    '''
    数据读取
    '''
    # 股票的索引
    # stock_index = np.array([])
    # for i in range(0, 3):
    #     outdir = './source_data'
    #     outname = 'data' + str(i + 1) + '.csv'
    #     fullname = os.path.join(outdir, outname)
    #     stock_index = read_file(fullname, stock_index, day_num)
    # stock_index = stock_index.reshape(-1, 2)
    # stock_index = pd.DataFrame(stock_index, columns = ['index', 'belong'])
    # outdir = './source_data'
    # outname = 'index.csv'
    # fullname = os.path.join(outdir, outname)
    # stock_index.to_csv(fullname)
    # exit()
    '''
    数据筛选
    '''
    # 预处理
    # 中位数去极值
    stock_index = pd.read_csv('./source_data/index.csv')
    # 所属行业
    stock_belongs = stock_index['belong']
    stock_belongs = stock_belongs.values
    # 股票索引
    stock_index = stock_index['index']
    stock_index = stock_index.values
    factors = median_process(factor_num, day_num, stock_index)
    
    '''
    去除空的因子
    '''
    delete_index = np.array([3, 4, 6, 10, 23, 27, 28, 29, 9, 15, 16, 17, 18])
    factors = np.delete(factors, delete_index, axis=0)
    factor_num = factors.shape[0]
    '''
    处理缺失值
    '''
    factors = fill_process(factor_num, day_num, stock_index, factors, stock_belongs)

    # factors = factors.reshape(factor_num, day_num, stock_index.shape[0])
    print(factors.shape)
    # 将所有的数据按股票顺序排列
    for i in range(0, stock_index.shape[0]):
        stock_factor_set = np.array([])
        for j in range(0, factor_num):
            factor = np.array([])
            for k in range(0, day_num):
                factor = np.append(factor, factors[j, k, i])
            # 归一化
            # factor = normalize(factor.reshape(-1, 1))
            if stock_factor_set.shape[0] == 0:
                stock_factor_set = factor
            else:
                stock_factor_set = np.column_stack((stock_factor_set, factor))
        # save
        outdir = './source_data/clean_data'
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outname = str(stock_index[i]) + '.csv'
        fullname = os.path.join(outdir, outname)
        np.savetxt(fullname, stock_factor_set, delimiter=',')
