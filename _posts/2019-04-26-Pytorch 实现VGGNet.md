---
layout:     post
title:      Pytorch 实现VGGNet
subtitle:   Pytorch实践
date:       2019-04-26
author:     WEW
header-img: img/post-bg-iWatch.jpg
catalog: true
tags:
    - Pytorch
    - 框架
    - 深度学习
---

## VGG的网络结构
如下图所示
![](https://wx4.sinaimg.cn/large/d7b90c85ly1fxmo2fpvavj21650u079w.jpg)

## 使用Pytorch框架构建VGG16

    # 定义VGG16网络
    import torch
    import pdb
    import torch.nn as nn
    import torch.nn.functional as F
    chanels=[64,64,'MP',128,128,'MP',256,256,256,'MP',512,512,512,'MP',512,512,512,'MP3','FC','FC1','FC2']
    class VGGNet(nn.Module):
        #定义构造函数
        def __init__(self,Chanels,num_class):
            super(VGGNet,self).__init__()
            self.num_class=num_class
            layers=[]
            input_chanel=3
            for ar in Chanels:
               if(ar=='MP'):
                   layers.append(nn.MaxPool2d(kernel_size=2,stride=2))
               elif(ar=='MP3'):
                   layers.append(nn.MaxPool2d(3,stride=1,padding=1))
               elif(ar=='FC'):
                  layers.append(nn.Conv2d(512,1024,3,padding=6,dilation=6))
                 layers.append(nn.ReLU(inplace=True))
              elif(ar=='FC1'):
                   layers.append(nn.Conv2d(1024,1024,1))
                   layers.append(nn.ReLU(inplace=True))
              elif(ar=='FC2'):
                   layers.append(nn.Conv2d(1024,self.num_class,1))
              else:
                   layers.append(nn.Conv2d(in_channels=input_chanel,out_channels=ar,kernel_size=3,padding=1))
                   layers.append(nn.ReLU(inplace=True))
                   input_chanel=ar
          self.Layers=nn.ModuleList(layers)
     def forward(self,x):
          for lays in self.Layers:
              x=lays(x)
          out=x
          return out
    inputs=torch.randn(1,3,300,300)
    vgg16=VGGNet(chanels,10)
    output=vgg16(inputs)
    print(output.shape)

    
