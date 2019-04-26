---
layout:     post
title:      pytorch实现LeNet
subtitle:   pytorch实践
date:       2019-04-26
author:     WEW
header-img: img/post-bg-BJJ.jpg
catalog: true
tags:
    - Pytorch
    - 框架
    - 深度学习
---

# 建个LeNet

    '''
    class LeNet(nn.Module):
        def __init__(self):
            super(LeNet, self).__init__() #调用基类的__init__()函数
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, 
    stride=1, padding=0, dilation=1, groups=1, bias=True)
 
            torch.nn.MaxPool2d(kernel_size, stride=None, 
    padding=0, dilation=1, return_indices=False, ceil_mode=False)
        stride – the stride of the window. Default value is kernel_size
            self.conv=nn.Sequential( #顺序网络结构
                nn.Conv2d(1,6,5,stride=1,padding=2), #卷积层 输入1通道，输出6通道，kernel_size=5*5
                nn.ReLU(),         #激活函数
                nn.MaxPool2d(2,2),     #最大池化，kernel_size=2*2，stride=2*2
        #输出大小为14*14
                nn.Conv2d(6,16,5,stride=1,padding=2), #卷积层 输入6通道，输出16通道，kernel_size=5*5
                nn.ReLU(),
                nn.MaxPool2d(2,2),
        # 输出大小为7*7
                nn.Conv2d(16,120,5,stride=1,padding=2), #卷积层 输入16通道，输出120通道，kernel_size=5*5
                nn.ReLU(),
            )
            self.fc=nn.Sequential( #全连接层
                nn.Linear(7*7*120,120),
                nn.ReLU(),
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Linear(84,10),
                nn.Sigmoid(),
            )
    '''
    import torch
    import torch.nn as nn
    import pdb
    import torch.nn.functional as F

    inputs=torch.randn(1,3,28,28)
    class LeNet(nn.Module):
        def __init__(self):#创建构造函数
            super(LeNet,self).__init__()#调用基类构造函数
        
            self.conv=nn.Sequential(
            nn.Conv2d(3,6,5,stride=1,padding=2),
            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5,stride=1,padding=2),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16,120,5,stride=1,padding=2)
            )
        
            self.fc=nn.Sequential(
            nn.Linear(120*7*7,120),
            nn.Linear(120,84),
            nn.Linear(84,10),
            #nn.ReLU()
            )
        def forward(self,x):
            x=self.conv(x)
            print(x.shape)
            x=x.view(x.size(0),-1)
            out=self.fc(x)
            out=F.relu(out)
            #out=F.softmax(out,1)
            #print(out.shape)
            #pdb.set_trace()
            return out

    leNet=LeNet()
    oupt=leNet(inputs)
    print(oupt)
