---
layout:     post
title:      Pytorch torch
subtitle:   开始认知
date:       2019-04-11
author:     WEW
header-img: img/pytorch_learning.jpeg
catalog: true
tags:
    - Pytorch
    - 框架
    - 深度学习
---

[//]:!torch函数
    
# torch
        
torch package 包含了多维张量的数据结构以及在其之上的多种数学操作，另外该包也提供了多种工具可以针对多维张量及任意类型进行序列化

#### torch.is_tensor(obj)

判断obj是否为tensor类型

#### torch.is_stroage(obj)

判断obj是否为stroage类型

#### torch.numel(input)

返回input张量的元素个数

    >>> a = torch.randn(1,2,3,4,5)
    >>> torch.numel(a)
    120
    >>> a = torch.zeros(4,4)
    >>> torch.numel(a)
    16

#### torch.eye(n,m,output)
        
返回一个nxm的单位对角阵，m默认为n

#### torch.from_numpy(narrays)

将输入的narrys numpy类型的数组转化为tensor，值得注意的是转化后，二者共享一段内存空间，其值保持同步

    >>> a = numpy.array([1, 2, 3])
    >>> t = torch.from_numpy(a)
    >>> t
    torch.LongTensor([1, 2, 3])
    >>> t[0] = -1
    >>> a
    array([-1,  2,  3])

#### torch.linspace(start, end, steps, out=None) 
        
返回start到end中的steps个点组成的tensor

    >>> torch.linspace(3, 10, steps=5)

      3.0000
      4.7500
      6.5000
      8.2500
     10.0000
    [torch.FloatTensor of size 5]

    >>> torch.linspace(-10, 10, steps=5)

    -10
     -5
      0
      5
     10
    [torch.FloatTensor of size 5]
    
#### torch.logspace(start,end,steps,out=None)
        
返回以10为底，从start 到end的steps步长个点为指数，组成的tensor

    >>> torch.logspace(start=-10, end=10, steps=5)

     1.0000e-10
     1.0000e-05
     1.0000e+00
     1.0000e+05
     1.0000e+10
    [torch.FloatTensor of size 5]

#### torch.ones(*size,out=None)

返回以可变参数size组成的元素为全1的张量

    >>>torch.ones(2, 3)

     1  1  1
     1  1  1
    [torch.FloatTensor of size 2x3]

#### torch.rand(*sizes, out=None) → Tensor

返回从区间[0,1)采样的随机tensor，tensor的大小有不定参数sizes

#### torch.randn(*sizes, out=None) → Tensor

返回从以均值为0，方差为1的正态分布采样的随机tensor，大小由sizes决定

    >>> torch.randn(4)

    -0.1145
     0.0094
    -1.1717
     0.9846
    [torch.FloatTensor of size 4]
    
#### torch.arange(start, end, step=1, out=None) → Tensor
      
返回以大于等于start且小于end的，步长为step的一些点，长度为floor((end-start)/step)

      >>> torch.arange(1, 4)

     1
     2
     3
    [torch.FloatTensor of size 3]

    >>> torch.arange(1, 2.5, 0.5)

     1.0000
     1.5000
     2.0000
    [torch.FloatTensor of size 3]
    
#### torch.zeros(*sizes, out=None) → Tensor
      
返回全0的tensor，大小由sizes决定

# 索引,切片,连接,换位Indexing, Slicing, Joining, Mutating Ops

#### torch.cat(seq, dimension=0) → Tensor

返回以seq为目标 按照dimension上合并的tensor

    >>> x = torch.randn(2, 3)
    >>> x

     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size 2x3]

    >>> torch.cat((x, x, x), 0)

     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
     0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size 6x3]

    >>> torch.cat((x, x, x), 1)

     0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918  0.5983 -0.0341  2.4918
     1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735  1.5981 -0.5265 -0.8735
    [torch.FloatTensor of size 2x9]
    
#### torch.chunk(tensor, chunks, dim=0)
      
将输入的tensor 按照dim维度分成chunks块， 切块
