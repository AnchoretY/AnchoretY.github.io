---
title: 强化学习6——ActorCritic
copyright: true
mathjax: true
date: 2021-03-27 23:04:46
tags:
categories:
---

概述：首页描述

![]()

<!--more-->

## Actor Critic

&emsp;&emsp;虽然Poclicy Gradient成功解决了Value-base算法不能解决在连续动作中进行选择的问题，但是由于其只能进行回合制更新，因此具有更新效率低，算法稳定性差等原因，需要进一步完善，因此就发明了一种全新的算法——Actor Critic，这种算法将Value-base算法与Gradient算法进行结合，使Policy Gradient也能在每个step进行更新。

### 原理介绍

&emsp;&emsp;Actor Critic通过建立两个深度神经网络，Policy Gradient神经网络作为Actor，选择行为进行表演，而Value-base的神经网络利用其单步更新的特性作为Critic对Actor的行为进行评价，从而使Actor Critic模型可以实现单步更新，提升学习效率。

> Critic: Value-base，对每个action进行评价，指导Actor更新
>
> Actor: Policy Gradient，进行行动











##### 参考文献

- xxx
- xxx