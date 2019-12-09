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
    set_backtest(initial_cash=10000000.000000,stock_cost_fee=40)
    context.F = Basic_FrameWork()
    context.F.Get_Data()


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
    
    context.F.Get_Simple_Data(month_base)
    score = context.F.Test_Simple(int(year_base / 12))
    stock_index = pd.read_csv('./source_data/index.csv')
    # 股票索引
    stock_index = stock_index['index']
    stock_index = stock_index.values
    # buy_target_len = int(score.shape[0] / 5)
    buy_target_len = 5
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
