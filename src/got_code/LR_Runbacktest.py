# -*- coding: utf-8 -*-
# 创建时间:2019/11/16 20:11:02
from atrader import *


if __name__ == '__main__':
    begin_date = '2011-01-31'
    end_date = '2018-12-31'
    target_list=get_code_list('szse_a', date=end_date)['code'].to_list()
    
    run_backtest(strategy_name='LR',
                file_path='LR_Strategy_ml.py',
                target_list=target_list,
                frequency='month',
                fre_num=1,
                begin_date=begin_date,
                end_date=end_date,
                fq=enums.FQ_NA)