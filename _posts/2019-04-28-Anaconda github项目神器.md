---
layout:     post
title:      Anaconda github项目神器
subtitle:   Stay Hungary，Stay Young
date:       2019-04-28
author:     WEW
header-img: img/post-bg-art.jpg
catalog: true
tags:
    - github
    - 工具
    - 框架
---

#### 题记
TMD，感觉自己蠢爆了，之前练习github框架的时候，都是下载源码，然后一点点在本机配置环境，过程实在艰辛(原谅我之前没用过anaconda)，不是少安装了这个，就是缺那个包，今天在学习CenterNet源码的时候，发现作者推荐了Anaconda这个安装，我操，发现了一片蓝天感觉，世外桃源的感觉，原谅我的无知。。。。汗！

# Anaconda 安装
安装Anaconda是比较简单的，从[官网](https://www.anaconda.com/distribution/)下载对应的linux版本即可，然后./xxx.sh文件一路Enter键即可顺利安装成功。这里安装成功后需要配置下环境变量

    * 1. 打开~/.bashrc
    * 2. 在文本后面添加export PATH=/root/anaconda3/bin:$PATH
    * 3. source ~/.bashrc

# Anaconda 环境(env)
环境，我的理解类似于docker中的沙盒，每个环境互不相连，每个环境也相当于linux下的一个独立用户，可以进行环境安装对应的包，我们可以针对学习的每个框架源码建立个环境，学习完后，直接delete该环境即可，对原来系统毫无影响，简直神器。

## 新建环境
anaconda环境的建立是通过以下命令进行的

    conda create -n envname 建立一个名叫envname的环境
    conda create -n envname python=3.7.1 建立一个名叫envname的环境，且环境初始包为python3.7.1
    
## 查看环境

    conda env list
    
## 进入环境

    conda activate envname
    
在进入环境后可以使用 --file file.conf配置文件进行批零安装包

    conda install --file file.conf
    
## 退出环境

    conda deactivate
    
## 删除环境
    
    conda env remove -n py3删除名为py3的环境
    
# 查看当前环境安装的包

    conda list
