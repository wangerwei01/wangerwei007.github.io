---
layout:     post
title:      解析Focal Loss
subtitle:   loss 设计
date:       2019-07-15
author:     WEW
header-img: img/post-bg-hacker.jpg
catalog: true
tags:
    - Loss设计
    - 深度学习
---


# 概述
focal loss的设计使模型专注于一些困难样本的loss计算，在目标检测中主要解决one-stage正负样本框不均衡的问题，该损失函数降低了简单样本在训练中的loss权重，加大了困难样本的关注度。

# 损失函数的设计
focal loss使在交叉熵的基础上设计的，我们这里可以先回顾下交叉熵损失函数
* 交叉熵损失函数
$$L = -ylogy'-(1-y)log(1-y')=\left\{
\begin{aligned}
-logy',y=1\\
-log(1-y'),y=0
\end{aligned}
\right.
$$
其中$y^{'}$是函数的预测值经过归一化到【0-1】之间。对于正样本而言，输出概率值越大损失越小，而反之对于负样本而言，输出概率越小则损失越小。

$$L =\left\{
\begin{aligned}
 -(1-y')^{\gamma}logy',y=1\\
 -y'^{\gamma}log(1-y'),y=0\\
\end{aligned}
\right.
$$

以上交叉熵损失函数的设计建立在正负样本比例正常的情况下，而对于one-stage的目标检测算法，往往正负样本框比例严重失衡，若采用简单的交叉熵损失，极易造成损失函数受负样本影响严重，使模型更加专注于负样本框的优化，正样本优化权重比例严重降低，达不到学习的目的。

# Focal Loss
$$ L_{fl}=\left\{
\begin{aligned}
 -(1-y')^{\gamma}logy',y=1\\
 -y'^{\gamma}log(1-y'),y=0\\
\end{aligned}
\right.
$$
如图所示，focal loss在交叉熵的基础上引入了样本比例系数，这里$\gamma$是大于0的超参。对于正类样本来说，预测结果为0.95的肯定是简单样本，这样(1-0.95)的$\gamma$次方就很小，这时损失函数也会很小，而预测率为0.3的样本其损失相对就很大。对于负样本而言同样，预测为0.1的结果是简单样本，这样对应的损失就会很小，反之假如预测为0.95时，负样本的损失就会很大。

$\gamma$调节简单样本权重降低的速率，当$\gamma$为0时，此时Focal Loss就是交叉熵损失，实验发现$\gamma=2$时效果最好。
* 有时为了平衡正负样本本身比例造成的失衡，也会引入平衡因子$\alpha$
$$ L_{fl}=\left\{
\begin{aligned}
 -\alpha(1-y')^{\gamma}logy',y=1\\
 -(1-\alpha)y'^{\gamma}log(1-y'),y=0\\
\end{aligned}
\right.
$$
通过实验一般$\alpha$为0.25时最好。

