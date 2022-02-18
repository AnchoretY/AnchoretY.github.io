---
title: 模型解释——SHAP
copyright: true
mathjax: true
date: 2022-02-18 11:43:54
tags:
categories:
---

概述：首页描述

![]()

<!--more-->

&emsp;&emsp;在使用SHAP进行模型解释之前，通常使用`feature importance`或`partial dependence  plot`来对一些不可解释的模型进行解释。

Feature importance可以直观的反映出特征的重要性，看出哪些特征对最终的模型影响较大。但是无法判断各个特征与对每个样本最终预测结果的最终关系。



### SHAP value

**优势：能够反应除各个特征对每一个样本的影响**

**使用场景：训练好的模型，要查看模型对某个样本的特征作用。**

&emsp;&emsp;假设第i个样本为$x_i$，第i个样本的第j个特征为$x_{i,j}$，模型对第i个样本的预测值为$y_i$，整个模型的基线（通常是所有样本的目标变量的均值）为$y_{base}$，那么SHAP value服从以下等式:

​														$$y_i=y_{base}+f(x_i,1)+f(x_i,2)+⋯+f(x_i,k)$$

&emsp;&emsp;其中f(xi,1)为$x_{i,j}$的SHAP值,从表达式中可以看出当$f(x_i,1)$>0,说明该特征提升了预测值，起正向作用；反之，说明该特征使预测值降低，有反向作用。



### 使用

Python中SHAP值的计算由`shap`这个package实现，可以通过`pip install shap`安装。

#### 1. 基本信息获取

~~~python
import shap

explainer = shap.TreeEcplaniner(model)
# 获取训练集中各个样本各个特征的shap值
shape_values = explainer.shap_values(data)

# 获得y_base值
y_base = explainer。expected_value

~~~

#### 2. 单个样本SHAP值分析



~~~Python
# 获取指定样本shap值
j = 30
player_explainer = pd.DataFrame()
player_explainer["feature_value"] = data.iloc[j].vlaues
palyer_explainer["shap_value"] = shape_vlaues[j]

# 指定样本shap值可视化
shap.initjs()
shap.force_plot(explainer.expected_value,shape_values[j],data.iloc[j])
~~~









