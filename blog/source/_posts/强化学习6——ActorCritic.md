---
title: 强化学习6——ActorCritic
copyright: true
mathjax: true
date: 2021-03-27 23:04:46
tags:
categories: 强化学习
---

概述：首页描述

![]()

<!--more-->

## Actor Critic

&emsp;&emsp;虽然Poclicy Gradient成功解决了Value-base算法不能解决在连续动作中进行选择的问题，但是由于其只能进行回合制更新，因此具有更新效率低，算法稳定性差等原因，需要进一步完善，因此就发明了一种全新的算法——Actor Critic，这种算法将Value-base算法与Gradient算法进行结合，使Policy Gradient也能在每个step进行更新。



------



### 原理介绍

&emsp;&emsp;Actor Critic通过建立两个深度神经网络，Policy Gradient神经网络作为**Actor**，**基于概率进行行为选择**，而Value-base的神经网络作为**Critic，基于ACtor采取的行为评判行为得分**，Actor根据Critic的评分修改选行为的概率。

> **Critic: Value-base**，对每个action进行评价，指导Actor更新
>
> **Actor: Policy Gradient**，行动选择，根据Critic的评分修改选行为概率

**算法目标：改善Policy Gradient只能回合制更新导致的低效率问题**

**优势**：可以进行单步更新

**缺点**：难收敛



### 算法框架











##### 参考文献

- xxx
- xxx