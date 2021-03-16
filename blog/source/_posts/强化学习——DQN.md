---
title: 强化学习——DQN
copyright: true
mathjax: true
date: 2021-03-15 15:03:24
tags:
categories:
---

概述：首页描述

![]()

<!--more-->

&emsp;&emsp;DQN是一种融合了深度神经网络与Q learning的方法，这种方式的核心思想是：使用神经网络来当做Q表来存储大量的Q值。

### 神经网络更新









### DQN两个关键点

#### experience replay

&emsp;&emsp;DQN每次更新时，都会随机抽取之前的一些经历进行学习

> 作用：打乱了过往之间经验的相关性，也使得神经网络更新更有效率



#### Fixed Q-target

&emsp;&emsp;Fixed Q-target也是一种打乱相关性的机理, 如果使用 fixed Q-targets, 我们就会在 DQN 中**使用到两个结构相同但参数不同的神经网络**,分别预测Q估计和Q现实，预测Q估计的神经网络具备最新的参数，而预测Q现实的神经网络则是一段时间以前的。

 











##### 参考文献

- xxx
- xxx