---
layout:     post
title:      Linux shell常用命令积累
subtitle:   Shell命令
date:       2019-04-16
author:     WEW
header-img: img/post-bg-kuaidi.jpg
catalog: true
tags:
    - Linux
    - Shell
---

## 命令积累(长期持续更新)

### 文件操作

cp filename  dir/filename2 拷贝文件到另一个文件或目录下

mv filename  dir  移动文件到另一个目录下

rm * 删除当前目录下所有文件；若要删除目录及其子目录，要使用 -rf参数

tar czvf my.tar file1 单个文件压缩打包

tar czvf my.tar file1 file2, 多个文件压缩打包 

tar czvf my.tar dir1 单个目录压缩打包 

多个目录压缩打包 tar czvf my.tar dir1 dir2

解包至当前目录：tar xzvf my.tar

### gpu相关

watch nvidia-smi 查看gpu显存实况

torch.cuda.empty_cache()使用pytorch释放gpu显存



