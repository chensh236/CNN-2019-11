# -*- coding: utf-8 -*-
from atrader import *
from datetime import datetime
import numpy as np
import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm
from Basic_FrameWork import *

def init(context: 'ContextBackReal'):
    """用户初始化函数"""
    set_backtest(initial_cash=10000000.000000)
    # context.data = pd.read_pickle('data.pkl').query("code=="+str(context.target_list))
    # context.data = context.data.fillna(0) #测试阶段，缺失值先用零填充，后期删除
    # context.factor_list = [i for i in context.data.columns if i not in ["industry_sw",'ev', 'Date', 'pct_chg']]
    # context.date_list = get_trade_date_list('2005-01-31','2019-01-31',_period='monthly',begin_or_end='end')
    # context.data_split = []
    # for year in range(2011,2020):
    #     trade_date = [i for i in context.date_list if i.startswith(str(year))][0]
    #     arg = context.date_list.index(trade_date)
    #     date_window_str = context.date_list[arg-72:arg+1]
    #     print(str(year)+'\n'+str(date_window_str))
    #     date_window = [datetime.strptime(i,'%Y-%m-%d') for i in date_window_str]
    #     factor = context.data.loc[date_window[:-1], context.factor_list]
    #     pct_chg = context.data.loc[date_window[1:], 'pct_chg']
    #     context.data_split.append((factor,pct_chg))
    # for year in range(2011, 2020):
    #     ## 保存csv
    #     outname = str(year) + '_factor.csv'
    #     outdir = './cache'
    #     if not os.path.exists(outdir):
    #         os.mkdir(outdir)
    #     fullname = os.path.join(outdir, outname)
    #     tmp = np.array(context.data_split[year - 2011][0].values)
    #     np.savetxt(fullname, tmp, delimiter=',')
    #     outname = str(year) + '_pct_chg.csv'
    #     outdir = './cache'
    #     fullname = os.path.join(outdir, outname)
    #     tmp = np.array(context.data_split[year - 2011][1].values)
    #     np.savetxt(fullname, tmp, delimiter=',')


def on_data(context: 'ContextBackReal'):
    """刷新bar函数"""
    #获取数据
    # current_date = context.now.strftime('%Y-%m-%d')
    current_year = context.now.strftime('%Y')
    current_month = context.now.strftime('%m')
    current_year = int(current_year)
    current_month = int(current_month)
    year_base = 0
    for i in range(2011, current_year):
        year_base = year_base + 12
    month_base = year_base + current_month - 1 + 72
    F = Basic_FrameWork()
    F.Get_Data()
    F.Get_Simple_Data(month_base)
    score = F.Test_Simple(year_base)
    stock_index = pd.read_csv('./source_data/index.csv')
    # 股票索引
    stock_index = stock_index['index']
    stock_index = stock_index.values
    # buy_target_len = int(score.shape[0] / 5)
    buy_target_len = 1
    score = score.argsort()[-buy_target_len:][::-1]
    buy_target = []
    for i in score:
        buy_target.append(stock_index[i])
    # buy_target = score[:buy_target_len].index.to_list()
    print('此次需购买的股票代码为：'+str(buy_target))
    buy_target_idx = []
    for i in buy_target:
        try:
            buy_target_idx.append(context.target_list.index(i))
        except:
            print('%s is not in the list!'%(i))
    
    positions = context.account().positions
    #卖出不在标的池中的股票
    for target_idx in positions.target_idx.astype(int):
        if target_idx not in buy_target_idx:
            if positions['volume_long'].iloc[target_idx] > 0:
                order_volume(account_idx=0, target_idx=target_idx,
                             volume=int(positions['volume_long'].iloc[target_idx]),
                             side=2, position_effect=2, order_type=2, price=0)
    # 获取股票的权重
    percent = 1/buy_target_len
    # 买在标的池中的股票
    for target_idx in buy_target_idx:
        order_target_percent(account_idx=0, target_idx=int(target_idx), target_percent=percent, side=1, order_type=2, price=0)    

        
        
##########################################################################################################    

def get_trade_date_list(begin_date,end_date,_period='monthly',begin_or_end='begin'):
    trade_date_list = get_trading_days('sse',begin_date,end_date)
    time_series = pd.Series(trade_date_list)
    week = time_series.apply(lambda x:x.week)
    month = time_series.apply(lambda x:x.month)
    quarter = time_series.apply(lambda x:x.quarter)
    year = time_series.apply(lambda x:x.year)
    if _period =='daily':
        trade_date_list = time_series.apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    if _period == 'weekly' and begin_or_end == 'begin':
        trade_date_list = time_series[week!=week.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'weekly' and begin_or_end == 'end':
        trade_date_list = time_series[week!=week.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'monthly' and begin_or_end == 'begin':
        trade_date_list = time_series[month != month.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'monthly' and begin_or_end == 'end':
        trade_date_list = time_series[month != month.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'quarterly' and begin_or_end == 'begin':
        trade_date_list = time_series[quarter != quarter.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'quarterly' and begin_or_end == 'end':
        trade_date_list = time_series[quarter != quarter.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'yearly' and begin_or_end == 'begin':
        trade_date_list = time_series[year != year.shift(1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    elif _period == 'yearly' and begin_or_end == 'end':
        trade_date_list = time_series[year != year.shift(-1)].apply(lambda x: x.strftime('%Y-%m-%d')).tolist()
    return trade_date_list
