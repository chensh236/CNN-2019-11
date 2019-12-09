# -*- coding: utf-8 -*-
# 创建时间:2019/11/16 20:11:02
from atrader import *


if __name__ == '__main__':
	begin_date = '2011-01-31'
	target_list = []

	run_realtrade(strategy_name='LR',
				file_path='LR_Strategy.py',
				account_list=[''],
				target_list=target_list,
				frequency='month',
				fre_num=1,
				begin_date=begin_date,
				fq=enums.FQ_NA)
