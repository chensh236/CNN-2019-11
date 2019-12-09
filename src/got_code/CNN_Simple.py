import torch
import torch.nn as nn
import torchvision
import torch.utils.data as Data

# Hyper Parameters
EPOCH = 1  # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001  # 学习率
DOWNLOAD_MNIST = True  # 已经下载好的话，会自动跳过的

# Mnist 手写数字
train_data = torchvision.datasets.MNIST(
    root='./mnist/',  # 保存或者提取位置
    train=True,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # 转换 PIL.Image or numpy.ndarray 成
    # torch.FloatTensor (C x H x W), 训练的时候 normalize 成 [0.0, 1.0] 区间
    download=DOWNLOAD_MNIST,  # 没下载就下载, 下载了就不用再下了
)

test_data = torchvision.datasets.MNIST(
    root='./mnist/',
    train=False
)

# 训练集丢BATCH_SIZE个, 图片大小为28*28
train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True  # 是否打乱顺序
)

# test_data为 [10000, 28, 28]这样的一个数组，这里就只截取一段就好了
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)
test_y = test_data.test_labels


# cnn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            # (1, 28, 28)
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,  # 卷积filter, 移动块长
                stride=1,  # filter的每次移动步长
                padding=2,
                groups=1
            ),
            # (16, 28, 38)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            # (16, 14, 14)
        )
        self.layer2 = nn.Sequential(

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        print('forward->'+str(x.shape))
        x = self.layer1(x)
        print('layer1->'+str(x.shape))
        x = self.layer2(x)
        print('layer2->'+str(x.shape))
        x = x.view(x.size(0), -1)
        print('view->'+str(x.shape))
        x = self.layer3(x)
        print('layer3->'+str(x.shape))
        return x
 

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_function = nn.CrossEntropyLoss()
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        output = cnn(b_x)
        loss = loss_function(output, b_y)
        print('output:'+str(output.shape))
        print('b_y:'+str(b_y.shape))
        
        exit()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print('finished training')
test_out = cnn(test_x)
predict_y = torch.argmax(test_out, 1).data.numpy()
print('Accuracy in Test : %.4f%%' % (sum(predict_y == test_y.data.numpy()) * 100/ len(predict_y)))