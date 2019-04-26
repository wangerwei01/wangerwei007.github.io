---
layout:     post
title:      git与github的关联
subtitle:   初识git
date:       2019-04-26
author:     WEW
header-img: img/post-bg-ios10.jpg
catalog: true
tags:
    - git
    - github
---


## git 连接 github

git是一个版本控制工具，而github是一个代码托管系统，二者可以进行互联互通。

#### git 配置
这里默认是安装了git，配置用户名和邮箱

     git config --global user.name ”githubUsername”
     git config --global user.email "githubUseremaill"
     cat ~/.gitconfig 查看配置

#### git init 本地仓库初始化
使用git init 命令可以使当前目录变成一个受git控制的仓库，自动生成.git目录，通常是隐藏的，本地仓库必须存在才能与远程仓库建立对应关系

#### 生成 SSHkey
使用以下命令，可以在本地root用户下生成/.ssh/目录，内含id_rsa是私钥，id_rsa.pub是公钥 等两个文件

    ssh-keygen -t rsa -C"你的邮箱"

#### 配置github SSHkey
登录github中setting中的SSH部分，复制 id_rsa.pub内容，粘贴至 SSHkey部分，add成功

#### 测试SSH连接github
使用以下命令测试ssh连接github，若显示You’ve successfully authenticated, but GitHub does not provide shell access ，则操作成功

    ssh -T git@github.com


## 本地仓库和github之间连接

（在当前仓库下那个目录下就是指的是操作的那个仓库）

    git clone git@github.com:wangerwei007/LinuxShell.git 克隆远程仓库到本地
    git remote add origin git@github.com:wangerwei007/LinuxShell.git 关联远程仓库

#### 添加文件至远程仓库(示例)

    1.本地新建 test.txt 文件
    2.git add test.txt 添加test.txt至git缓存区(缓存区后续会提到)
    3.git commit -m test.txt 推至远程仓库
    4.git push -u origin master 更新本地仓库和远程仓库保持一致
