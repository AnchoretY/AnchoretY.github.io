---
title: SVM模型
date: 2018-10-19 10:07:27
tags: 机器学习
categories: 机器学习
---

1.SVM模型的超参数

​	SVM模型主要包括**C**和**gamma**两个超参数。

​	**C是惩罚系数，也就是对误差的宽容度**，C越大代表越不能容忍出现误差，越容易出现过拟合；

​	**gamma是选择RBF核时，RBF核自带的一个参数，隐含的决定数据映射到新空间的后的分布**，gamma越大，支持向量越少。

> 支持向量的个数影响训练和预测的个数

> gamma的物理意义，大家提到很多的RBF的幅宽，它会影响每个支持向量对应的高斯的作用范围，从而影响泛化性能。我的理解：如果gamma设的太大,σ会很小，σ很小的高斯分布长得又高又瘦， 会造成只会作用于支持向量样本附近，对于未知样本分类效果很差，存在训练准确率可以很高，无穷小，则理论上，高斯核的SVM可以拟合任何非线性数据，但容易过拟合)而测试准确率不高的可能，就是通常说的过训练；而如果设的过小，则会造成平滑效应太大，无法在训练集上得到特别高的准确率，也会影响测试集的准确率

2.