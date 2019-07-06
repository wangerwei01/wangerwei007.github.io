---
layout:     post
title:      人脸检测Wider Face、FDDB数据集评估姿势
subtitle:   人脸检测
date:       2019-05-16
author:     WEW
header-img: img/post-bg-art.jpg
catalog: true
tags:
    - 人脸检测
    - 工具
    - 数据
---

# Wider数据集的获取
http://shuoyang1213.me/WIDERFACE/
# 评估工具的获取
http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip
# Wider数据集的predict形式
针对每一个图片要生成对应的txt文档，文档内容形式为：
```txt
图片名称
检测框个数
xmin ymin w h score
xmin ymin w h score
xmin ymin wh score
```
# 评估姿势
将数据打包解压到eval_tools文件夹下命名文件包为pred，然后直接运行即可。



* * *
# FDDB数据集的获取
https://drive.google.com/open?id=17t4WULUDgZgiSy5kpCax4aooyPaz3GQH
# FDDB评估工具的获取
https://bitbucket.org/marcopede/face-eval
http://vis-www.cs.umass.edu/fddb/evaluation.tgz
# FDDB数据集的predict形式
针对所有的图片文件列表生成txt文档，文档内容是：
```txt
图片名称1
检测框个数
xmin ymin w h score
xmin ymin w h score
xmin ymin wh score
图片名称2
检测框个数
xmin ymin w h score
xmin ymin w h score
xmin ymin wh score
...
```
# FDDB评估姿势
先安装opencv(C++环境)编译evaluation.tgz文件，运行./evalution 可获取需要参数一一进行带入即可，最后会生成一个dicROC.txt文件，复制该文件到face-eval/detection/下，运行eval-fddb.py文件即可

