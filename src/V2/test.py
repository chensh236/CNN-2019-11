import torch
import torch.nn as nn
import math

m = nn.Sigmoid()

loss = nn.BCELoss(size_average=False, reduce=False)
input = torch.randn(1, requires_grad=True)
target = torch.empty(1).random_(2)
lossinput = m(input)
output = loss(lossinput, target)

print("输入值:")
print(lossinput)
print("输出的目标值:")
print(target)
print("计算loss的结果:")
print(output)
print("自己计算的第一个loss：")