---
layout:     post
title:      RetinaFace
subtitle:   随笔
date:       2019-07-08
author:     WEW
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 论文
    - 人脸检测
---

# RetinaFace
论文题目：: RetinaFace：Single-stage Dense Face Localisation in the Wild

code github：https://github.com/deepinsight/insightface/tree/master/RetinaFace

# 主要贡献
+ 在WIDER FACE数据集上人工标注了5个人脸面部地标，在hard face数据上借助额外的监督信息实现了重大改善
+ 额外添加了一个自监督网格解码分支，和现存的监督分支并行式逐像素预测3D人脸形状信息
+ 在WIDER FACE hard数据集上实现了91.4%的AP值，超过了现存主流算法1.1%
+ 通过轻量级权重backbone，RetinaFace可以针对VGA分辨率图像(640x480)在单CPU上实时运行。

# 解决的问题
使用额外的5人脸面部地标监督信息能推动Wider Face hard数据集检测精度的提升

# Retina Face Details
#### Multi-task Loss
针对每一个anchor点i，作者设计了下列多任务损失

$a \bmod b$
