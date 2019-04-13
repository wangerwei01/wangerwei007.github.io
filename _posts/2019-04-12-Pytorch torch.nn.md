---
layout:     post
title:      Pytorch torch.nn
subtitle:   开始认知
date:       2019-04-12
author:     WEW
header-img: img/pytorch_learning.jpeg
catalog: true
tags:
    - Pytorch
    - 框架
    - 深度学习
---

[//]:!torch函数
    
# Parameters
        
### class torch.nn.Parameter(data,requires_grad)
        
该类是属于变量(variable)的一种,常用作模块参数，Parameter是variable类的子类，Parameter和module一起使用的时候会产生特殊的属性，它会被自动添加到
module的参数列表中(即会出现在parameter()迭代器中)，此外parameter和variable还有一个区别就是，parameter 不能被volatile(因优化而被省略)，且默认
requires_grad=True，而变量的默认requires_grad=False


# Containers容器

### class torch.nn.Module

所有网络的基类，所有的网络模型都基于该类产生，modules包含其他modules，允许以树的形式嵌入到它们，你可以将子模块赋值给模块属性

    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
    
        def __init__(self):#构造函数
            super(Model,self).__init__()#运行基类的构造函数
            self.conv1=nn.Conv2d(1,20,5)#subModuel：Conv2d 子模块
            self.conv2=nn.Conv2d(20,20,5)
    
        def forward(self,x):
        
            x=F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
    
    model = Model()
    print(model.conv1)
        
    输出：Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))

#### add_module()

也可以通过该条语句实现添加子模块的属性，子模块通过name属性来获取

    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
    
        def __init__(self):#构造函数
            super(Model,self).__init__()#运行基类的构造函数
            self.add_module("conv1",nn.Conv2d(1,20,5))#该种方式添加和以上方式等价
            self.add_module("conv2",nn.Conv2d(20,20,5))
    
        def forward(self,x):
        
            x=F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
    
    model = Model()
    print(model.conv1)
        
#### children()
        
返回当前模式的子模式

    for sub_module in model.children():
        print(sub_module)
        
    Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
    Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1))
        
#### cpu(devie_id=None)
        
将当前module的所有参数parameters和buffers都复制到cpu

#### cuda(device_id=None)

将当前module的所有参数parameters和buffers都复制到gpu

#### double() float() half()

将当前module的所有参数parameter和buffers都转为double、float、half类型

#### eval()

将模型设置成 evaluation 模式，只有当模型中存在dropout和batchNorm是才会影响的

#### forward(* input)

定义了每次执行的计算步骤，在所有子类都要重写这个函数

#### load_state_dict(state_dict)

将state_dict中的参数和buffers都复制到该moduel及其后代中，要求state_dict返回的key要和model.state_dict的key保持一致
state_dict (dict) – 保存parameters和persistent buffers的字典

#### modules()

和children差不多，只不过返回的包含本身model
NOTE： 重复的模块只被返回一次(children()也是)。 在下面的例子中, submodule 只会被返回一次

    import torch.nn as nn
    import torch.nn.functional as F

    class Model(nn.Module):
    
        def __init__(self):#构造函数
            super(Model,self).__init__()#运行基类的构造函数
            submodule=nn.Conv2d(20,20,5)
            self.add_module("conv1",submodule)#该种方式添加和以上方式等价
            self.add_module("conv2",submodule)
    
        def forward(self,x):
        
            x=F.relu(self.conv1(x))
            return F.relu(self.conv2(x))
    model=Model()
    for sub_module in model.modules():
        print(sub_module)
        
    Model(
      (conv1): Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1))
    )
    Conv2d(20, 20, kernel_size=(5, 5), stride=(1, 1))
        
#### named_children()
        
返回 包含 模型当前子模块 的迭代器，yield 模块名字和模块本身。

    for name, module in model.named_children():
        if name in ['conv4', 'conv5']:
            print(module)
        
#### parameters(memo=None)
        
返回一个 包含模型所有参数 的迭代器。一般用来当作optimizer的参数。

    for param in model.parameters():
        print(type(param.data), param.size())

    <class 'torch.Tensor'> torch.Size([20, 20, 5, 5])
    <class 'torch.Tensor'> torch.Size([20])
    <class 'torch.Tensor'> torch.Size([20, 20, 5, 5])
    <class 'torch.Tensor'> torch.Size([20])

#### state_dict(destination=None, prefix='')[source]
        
返回当前状态下module的状态值，以字典的形式返回

    import torch
    from torch.autograd import Variable
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self):
            super(Model, self).__init__()
            self.conv2 = nn.Linear(1, 2)
            self.vari = Variable(torch.rand([1]))
            self.par = nn.Parameter(torch.rand([1]))
            self.register_buffer("buffer", torch.randn([2,3]))

    model = Model()
    print(model.state_dict().keys())
        
    odict_keys(['par', 'buffer', 'conv2.weight', 'conv2.bias'])
    
#### train(mode=True)

将模型设置为training状态，仅仅当模型中有Dropout和BatchNorm是才会有影响。

#### zero_grad()

将模型中所有的参数梯度置为0

### class torch.nn.Sequential(* args)

一个时序容器。Modules 会以他们传入的顺序被添加到容器中。当然，也可以传入一个OrderedDict

    # Example of using Sequential

    model = nn.Sequential(
          nn.Conv2d(1,20,5),
          nn.ReLU(),
          nn.Conv2d(20,64,5),
          nn.ReLU()
        )
    # Example of using Sequential with OrderedDict
    model = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1,20,5)),
          ('relu1', nn.ReLU()),
          ('conv2', nn.Conv2d(20,64,5)),
          ('relu2', nn.ReLU())
        ]))
        
### class torch.nn.ModuleList(modules=None)
        
将submodules保存在一个list中。ModuleList可以像一般的Python list一样被索引。而且ModuleList中包含的modules已经被正确的注册，对所有的module method可见。

    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

        def forward(self, x):
            # ModuleList can act as an iterable, or be indexed         using ints
            for i, l in enumerate(self.linears):
                x = self.linears[i // 2](x) + l(x)
            return x
            
            
### CLASS torch.nn.ModuleDict(modules=None)

将所有的submodules保存在一个字典中，可以向字典一样被索引，对所有model可见

    class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.choices = nn.ModuleDict({
                'conv': nn.Conv2d(10, 10, 3),
                'pool': nn.MaxPool2d(3)
        })
        self.activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['prelu', nn.PReLU()]
        ])

    def forward(self, x, choice, act):
        x = self.choices[choice](x)
        x = self.activations[act](x)
        return x
            
### class torch.nn.ParameterList(parameters=None)

将parameter保存在一个list中,ParameterList可以像一般的Python list一样被索引。而且ParameterList中包含的parameters已经被正确的注册，对所有的module method可见。

    class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(10, 10)) for i in range(10)])

    def forward(self, x):
        # ParameterList can act as an iterable, or be indexed using ints
        for i, p in enumerate(self.params):
            x = self.params[i // 2].mm(x) + p.mm(x)
        return x

### CLASS torch.nn.ParameterDict(parameters=None)

将所有的参数保存在一个字典中，可以向字典一样被索引，对所有model可见

    class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.params = nn.ParameterDict({
                'left': nn.Parameter(torch.randn(5, 10)),
                'right': nn.Parameter(torch.randn(5, 10))
        })

    def forward(self, x, choice):
        x = self.params[choice].mm(x)
        return x


# Convolution layers 卷积层

## Conv1d

### CLASS torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

Conv1d卷积主要应用于文本上，该层参数为 weight(tensor) - 卷积的权重，大小是(out_channels, in_channels, kernel_size) 
bias(tensor) - 卷积的偏置系数，大小是（out_channel）

    m=nn.Conv1d(16,33,3,stride=2)
    input=torch.autograd.Variable(torch.randn(20, 16, 50))
    output = m(input)
    print(output.shape[0],output.shape[1],output.shape[2])
                            
    20 33 24
                            
## Conv2d
                            
### torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)

Conv2d卷积主要应用于图像上，该层参数为 weight(tensor) - 卷积的权重，大小是(out_channels, in_channels, kernel_size) 
bias(tensor) - 卷积的偏置系数，大小是（out_channel）

    ![](img/计算卷积.jpeg)

    #m=nn.Conv2d(16,33,3,stride=2)
    m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
    #m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
    input=torch.randn(20, 16, 50,100)
    output = m(input)
    print(output.shape[0],output.shape[1],output.shape[2],output.shape[3])
                            
    20 33 28 100
                            
### torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1)
        
一维反卷积过程

[//]:!arXiv-A guide to convolution arithmetic for deep learning#这是一篇写卷积和反卷积的论文，还不错
![](https://blog.csdn.net/Fate_fjh/article/details/52882134)
$$L_{out}=(L_{in}-1)stride-2padding+kernel_size+output_padding$$

# pooling层，池化层

### torch.nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
             
其实pool层和卷积层差不多操作，只不过把卷积核覆盖所有数乘积改成了求最大值，输出方式和卷积一样

    m = nn.MaxPool1d(3, stride=2)
    input = torch.autograd.Variable(torch.randn(20, 16, 50))
    output = m(input)
    print(output.shape[0],output.shape[1],output.shape[2])
    
    20 16 24
    
### torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
    
    # pool of square window of size=3, stride=2
    #m = nn.MaxPool2d(3, stride=2)
    # pool of non-square window
    m = nn.MaxPool2d((3, 2), stride=(2, 1))
    input = torch.randn(20, 16, 50, 32)
    output = m(input)
    print(output.shape[0],output.shape[1],output.shape[2],output.shape[3])
    
    20 16 24 31
    
### torch.nn.AvgPool1d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
### torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)
    
解释同最大池化

### class torch.nn.AdaptiveAvgPool1d(output_size)

对输入信号，提供1维的自适应平均池化操作 对于任何输入大小的输入，可以将输出尺寸指定为H*W，但是输入和输出特征的数目不会变化。

    >>> # target output size of 5
    >>> m = nn.AdaptiveAvgPool1d(5)
    >>> input = autograd.Variable(torch.randn(1, 64, 8))
    >>> output = m(input)

### class torch.nn.AdaptiveAvgPool2d(output_size)
    
解释同上

    # target output size of 5x7
    m = nn.AdaptiveAvgPool2d((5,7))
    input = torch.autograd.Variable(torch.randn(1, 64, 8, 9))
    output = m(input)
    print(output.shape[0],output.shape[1],output.shape[2],output.shape[3])
    # target output size of 7x7 (square)
    m = nn.AdaptiveAvgPool2d(7)
    input = torch.autograd.Variable(torch.randn(1, 64, 10, 9))
    output = m(input)
    print(output.shape[0],output.shape[1],output.shape[2],output.shape[3])
    
    1 64 5 7
    1 64 7 7
    
# linear layers 线性层
    
### torch.nn.Linear(in_features, out_features, bias=True)
    
线性层其实无非拟合的一种线性关系<a href="https://www.codecogs.com/eqnedit.php?latex=y=xA^{T}&plus;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=xA^{T}&plus;b" title="y=xA^{T}+b" /></a>


    m = nn.Linear(20, 30)
    input = torch.randn(128, 20)
    output = m(input)
    print(output.size())
    
    torch.Size([128, 30])
    
### torch.nn.Bilinear(in1_features, in2_features, out_features, bias=True)
    
该层为双线性层，计算的是<a href="https://www.codecogs.com/eqnedit.php?latex=y=xA^{T}&plus;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y=x1Ax2&plus;b" title="y=x1Ax2+b" /></a>
，其实相当于两层输入共同输入到线性中，可以理解为

两种A矩阵，A1.size= 20*40 A2.size = 30*40 纯属个人理解

    m = nn.Bilinear(20, 30, 40)
    input1 = torch.randn(128, 20)
    input2 = torch.randn(128, 30)
    output = m(input1, input2)
    print(output.size())
    
    128 40

# Drop layers
    
### torch.nn.Dropout(p=0.5, inplace=False)
    
以p的概率是一些权重置为0

    m = nn.Dropout(p=0.2)
    input = torch.randn(20, 16)
    output = m(input)
    y=output==0
    z=output[y]
    print(z.size())
    
    64

### torch.nn.Dropout2d(p=0.5, inplace=False)
    
同上

# Vision layers

### class torch.nn.PixelShuffle(upscale_factor)

将shape为$[N, Cr^2, H, W]$的Tensor重新排列为shape为$[N, C, Hr, W*r]$的Tensor

    >>> ps = nn.PixelShuffle(3)
    >>> input = autograd.Variable(torch.Tensor(1, 9, 4, 4))
    >>> output = ps(input)
    >>> print(output.size())
    torch.Size([1, 1, 12, 12])

### torch.nn.Upsample(size=None, scale_factor=None, mode='nearest', align_corners=None)
    
    Input: $(N, C, W_{in}), (N, C, H_{in}, W_{in}) or (N, C, D_{in}, H_{in}, W_{in})$
    Output: (N, C, W_{out}), (N, C, H_{out}, W_{out}) or (N, C, D_{out}, H_{out}, W_{out})$
    
    $D_{out}=[D_{in} * scale_factor] or size[-3]$
    $H_{out}=[H_{in} * scale_factor] or size[-2]$
    $W_{out}=[W_{in} * scale_factor] or size[-1]$
    
    
### torch.nn.UpsamplingNearest2d(size=None, scale_factor=None)

    同上
    
### torch.nn.UpsamplingBilinear2d(size=None, scale_factor=None)

    同上
    
# Normalization layers

### torch.nn.BatchNorm1d(num_features, eps=1e-05, momentum=0.1, affine=True)

对小批量(mini-batch)的2d或3d输入进行批标准化(Batch Normalization)操作

    $$y={\frac{x-mean[x]}{{\sqrt{var[x]}}+\epsilon} * gamma + beta$$

其中$\gamma,beta$均是学习的量，大小为C的参数向量（C为输入大小）
在训练时，该层计算每次输入的均值与方差，并进行移动平均。移动平均默认的动量值为0.1。在验证时，训练求得的均值/方差将用于标准化验证数据。

### torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)

同上
