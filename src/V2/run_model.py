#                    _ooOoo_
#                   o8888888o
#                   88" . "88
#                   (| -_- |)
#                   O\  =  /O
#                ____/`---'\____
#              .'  \\|     |//  `.
#             /  \\|||  :  |||//  \
#            /  _||||| -:- |||||-  \
#            |   | \\\  -  /// |   |
#            | \_|  ''\---/''  |   |
#            \  .-\__  `-`  ___/-. /
#          ___`. .'  /--.--\  `. . __
#       ."" '<  `.___\_<|>_/___.'  >'"".
#      | | :  `- \`.;`\ _ /`;.`/ - ` : | |
#      \  \ `-.   \_ __\ /__ _/   .-` /  /
# ======`-.____`-.___\_____/___.-`____.-'======
#                    `=---='
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#             佛祖保佑       永无BUG


'''
生成训练集和测试集
'''
from framework import *
from cnn_model import *
import numpy as np
from pandas import Series, DataFrame
import pandas as pd
from framework import *
from cnn_model import *
from create_dataset import *
def run(year, train_set_num, test_set_num, cols, factor_num, stock_num):
    c = Crate_dataset()
    Train_X, Train_Y, Test_X, Test_Y = c.read(year, train_set_num, test_set_num, cols, factor_num)
    # num_epochs 测试的次数
    # learning_rate 学习率
    # batch_size 每次学习中数据量大小
    # factor_num 因子数量
    # cols 列大小
    # model_num 模型数量，默认为1
    # [] cnn模型参数
        # kernel_size = 5  卷积核
        # out_channels = 10 卷积核数量
        # dropout_rate = 0.8 失活率
        # in1 = 100 第一层神经元数量
        # in2 = 70 第二层神经元数量
        # in3 = 40 第三层神经元数量

    # args_arr = [5, 10, 0.0, 64, 43, 22]
    args_arr = [5, 10, 0.0, 100, 70, 40]
    print('lr = 0.001 batch = 100， 去掉初始化')
    # batch size: [32:256]
    model_tmp = 2
    F = framework(100, 0.001 , 100, 17, 5, 1 + model_tmp, args_arr)
    F.data_input(Train_X, Train_Y, Test_X, Test_Y, train_set_num, test_set_num, stock_num)
    # model_index 第0个模型，只有一个模型默认为0，不要修改
    # show_predict_y: 判断是否显示预测的值
    F.train(model_tmp, False)
    # 保存模型, 第二个参数为名字
    model_name = 'cnn_model' + str(model_tmp)
    F.save(model_tmp, model_name)
    # 加载模型
    # F.load(model_tmp, model_name)
    # model_index 第0个模型，只有一个模型默认为0，不要修改
    # test_rate 测试的范围 0-1 1则测试全部
    F.test(model_tmp, 1)

if __name__ == '__main__':
    # 修改年份，获得对应下标

    # 读取数据 划分训练和测试集
    # 训练集大小72个月
    train_set_num = 72
    # 测试集大小12个月
    test_set_num = 12
    # 5列
    cols = 5
    # 17个因子
    factor_num = 17
    # 年份
    year = 2011
    # 根据年份在cache中找到对应数据
    start_index = 0
    # 股票数量
    stock_num = 1102
    # for i in range(2011, year):
    #     start_index = start_index + 169
    run(year, train_set_num, test_set_num, cols, factor_num, stock_num)

    # 用于多线程
    # epoch = 1
    # for i in range(0, 1):
    # # for i in range(0, int(stock_index.shape[0] / epoch) + 1):
    #     p_arr = []
    #     for j in range(0, epoch):
    #         if i * epoch + j >= stock_index.shape[0]:
    #             break
    #         p = Process(target = run, args=([stock_index[i * epoch + j]]))
    #         p_arr.append(p)
    #         p.start()
    #     for iter in p_arr:
    #         iter.join()