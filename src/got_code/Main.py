from Basic_FrameWork import *
from multiprocessing import Queue, Process
if __name__ == '__main__':
    F = Basic_FrameWork()
    F.Get_Data()
    begin = 8
    print(str(begin))
    for i in range(begin, begin + 1):
        F.Get_Train_Data(i)
        F.Train(i, False, True)
        F.Get_Test_Data(i)
        F.Test(i)