---
title: 强化学习3——Sarsa
copyright: true
mathjax: true
date: 2021-03-21 21:57:55
tags:
categories:
---

概述：强化学习经典算法Sarsa算法从算法过程、伪代码、代码角度进行介绍。

![]()

<!--more-->

## Sarsa

&emsp;&emsp;Sarsa 跟 Q-Learning 非常相似，也是基于 Q-Table 进行决策的。不同点在于**决定下一状态所执行的动作的策略**，Q-Learning **在当前状态更新 Q-Table 时会用到下一状态Q值最大的那个动作，但是下一状态未必就会选择那个动作；但是 Sarsa 会在当前状态先决定下一状态要执行的动作，并且用下一状态要执行的动作的 Q 值来更新当前状态的 Q 值**；具体差异可以看下图：

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.t9i2i16cjea.png" alt="image" style="zoom:33%;" />

那么，这两者的区别在哪里呢？[这篇文章](https://studywolf.wordpress.com/2013/07/01/reinforcement-learning-sarsa-vs-q-learning/)里面是这样讲的

> This means that SARSA takes into account the control policy by which the agent is moving, and incorporates that into its update of action values, where Q-learning simply assumes that an optimal policy is being followed. 

简单来说就是 Sarsa 在执行action时会考虑到全局（如更新当前的 Q 值时会先确定下一步要走的动作）， 而 Q-Learning 则显得更加的贪婪和”短视”, 每次都会选择当前利益最大的动作(不考虑 ϵϵ-greedy)，而不考虑其他状态。

那么该如何选择，根据这个问题：[When to choose SARSA vs. Q Learning](https://stats.stackexchange.com/questions/326788/when-to-choose-sarsa-vs-q-learning)，有如下结论

> If your goal is to train an optimal agent in simulation, or in a low-cost and fast-iterating environment, then Q-learning is a good choice, due to the first point (learning optimal policy directly). If your agent learns online, and you care about rewards gained whilst learning, then SARSA may be a better choice.

简单来说就是如果要在线学习，同时兼顾 reward 和总体的策略(如不能太激进，agent 不能很快挂掉)，那么选择 Sarsa；而如果没有在线的需求的话，可以通过 Q-Learning 线下模拟找到最好的 agent。所以也称 Sarsa 为on-policy，Q-Leanring 为 off-policy。





##### 参考文献

- xxx
- xxx