---
title: >-
  论文精读——《BotMark: Automated botnet detection with hybrid analysis of flow-based
  and graph-based traffic behaviors》
copyright: true
mathjax: true
date: 2021-09-06 15:34:44
tags:
categories:
---

概述：本文提出了一种混合了flow和graph两个角度的网络行为进行自动化僵尸网络检测的模型BotMark，该模型中提取了15种基于flow的流量特征和3种基于graph的特征，通过计算与历史上的僵尸网络的相似度分数、结构稳定度分数、异常分数进行僵尸网络的检测，最终达到99.94%的准确率。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.2wrhzmc5w7a.png)

<!--more-->

题目：《BotMark: Automated botnet detection with hybrid analysis of flow-based and graph-based traffic behaviors》

作者：Wei Wang, Yaoyao Shang Yongzhong He, Yidong Li,Jiqiang Liu

出处： Information Sciences

年份：2019



### 综述

**主要贡献**

- 更加全面的检测：从flow与graph两个角度进行检测，不需要任何先验知识。
- 收集了一个由Mirai、Black energy、Zeus、Athena、Ares 5中真实环境中较新的僵尸网络构建的数据集，并已公开。
- 效果比flow或graph更好，最终准确度可达99.94%。

**方法：**

- 数据收集

- 过滤无关流量，减小需要建模的数据量

- 特征提取：15种flow特征、3中graph特征

- 基于flow与graph的综合分析，输出botnet检测结果

- 验证有效性

  ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.2wrhzmc5w7a.png)

### 论文详解

#### 数据预处理

​	数据预处理主要包括流量过滤与cflow整合两部分。首先为了提高建模效率，进行无关流量过滤,过滤包括:

- 过滤未建立完整连接的flow，这种flow大部分都是扫描行为。

- 过滤不是内部到外部的flow。

- 过滤目的域名为明显合法域名的flow，以alexa top1k为合法域名。

  随后将同一个时期内具有相同的传输层协议、源IP、源端口、目的IP、目的端口的flow整合为cflow。

#### 