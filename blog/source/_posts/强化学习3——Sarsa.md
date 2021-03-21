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

&emsp;&emsp;Sarsa 跟 Q-Learning 非常相似，也是基于 Q-Table 进行决策的。

#### 

不同点在于**决定下一状态所执行的动作的策略**，Q-Learning 在当前状态更新 Q-Table 时会用到下一状态Q值最大的那个动作，但是下一状态未必就会选择那个动作；但是 Sarsa 会在当前状态先决定下一状态要执行的动作，并且用下一状态要执行的动作的 Q 值来更新当前状态的 Q 值；说的好像很绕，但是看一下下面的流程便可知道这两者的具体差异了，图片摘自[这里](https://morvanzhou.github.io/tutorials/machine-learning/ML-intro/4-04-sarsa/)

[![Q-Learning vs Sarsa](http://static.zybuluo.com/WuLiangchao/4huj5yu2kwgjlihhvhhrqhcw/image_1cd24f88k2ae10911ipacg01cv49.png)](http://static.zybuluo.com/WuLiangchao/4huj5yu2kwgjlihhvhhrqhcw/image_1cd24f88k2ae10911ipacg01cv49.png)

那么，这两者的区别在哪里呢？[这篇文章](https://studywolf.wordpress.com/2013/07/01/reinforcement-learning-sarsa-vs-q-learning/)里面是这样讲的

> This means that SARSA takes into account the control policy by which the agent is moving, and incorporates that into its update of action values, where Q-learning simply assumes that an optimal policy is being followed. 

简单来说就是 Sarsa 在执行action时会考虑到全局（如更新当前的 Q 值时会先确定下一步要走的动作）， 而 Q-Learning 则显得更加的贪婪和”短视”, 每次都会选择当前利益最大的动作(不考虑 ϵϵ-greedy)，而不考虑其他状态。

那么该如何选择，根据这个问题：[When to choose SARSA vs. Q Learning](https://stats.stackexchange.com/questions/326788/when-to-choose-sarsa-vs-q-learning)，有如下结论

> If your goal is to train an optimal agent in simulation, or in a low-cost and fast-iterating environment, then Q-learning is a good choice, due to the first point (learning optimal policy directly). If your agent learns online, and you care about rewards gained whilst learning, then SARSA may be a better choice.

简单来说就是如果要在线学习，同时兼顾 reward 和总体的策略(如不能太激进，agent 不能很快挂掉)，那么选择 Sarsa；而如果没有在线的需求的话，可以通过 Q-Learning 线下模拟找到最好的 agent。所以也称 Sarsa 为on-policy，Q-Leanring 为 off-policy。

## DQN

我们前面提到的两种方法都以依赖于 Q-Table，但是其中存在的一个问题就是当 Q-Table 中的状态比较多，可能会导致整个 Q-Table 无法装下内存。因此，DQN 被提了出来，DQN 全称是 Deep Q Network，Deep 指的是通的是深度学习，其实就是通过神经网络来拟合整张 Q-Table。

DQN 能够解决状态无限，动作有限的问题；具体来说就是将当前状态作为输入，输出的是各个动作的 Q 值。以 Flappy Bird 这个游戏为例，输入的状态近乎是无限的（当前 bird 的位置和周围的水管的分布位置等），但是输出的动作只有两个(飞或者不飞)。实际上，已经有人通过 DQN 来玩这个游戏了，具体可参考这个 [DeepLearningFlappyBird](https://github.com/yenchenlin/DeepLearningFlappyBird)

所以在 DQN 中的核心问题在于如何训练整个神经网络，其实训练算法跟 Q-Learning 的训练算法非常相似，需要利用 Q 估计和 Q 现实的差值，然后进行反向传播。

这里放上提出 DQN 的原始论文 [Playing atari with deep reinforcement learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) 中的算法流程图

[![DQN](http://static.zybuluo.com/WuLiangchao/8gbw5uxcymp969jhi7etlbrc/image_1cd2kiaol19ft2kb1dr4ricnorm.png)](http://static.zybuluo.com/WuLiangchao/8gbw5uxcymp969jhi7etlbrc/image_1cd2kiaol19ft2kb1dr4ricnorm.png)

上面的算法跟 Q-Learning 最大的不同就是多了 **Experience Replay** 这个部分，实际上这个机制做的事情就是先进行反复的实验，并将这些实验步骤获取的 sample 存储在 memory 中，每一步就是一个 sample，每个sample是一个四元组，包括：当前的状态，当前状态的各种action的 Q 值，当前采取的action获得的即时回报，下一个状态的各种action的Q值。拿到这样一个 sample 后，就可以根据上面提到的 Q-Learning 更新算法来更新网络，只是这时候需要进行的是反向传播。

**Experience Replay 机制的出发点是按照时间顺序所构造的样本之间是有关的(如上面的 ϕ(st+1)ϕ(st+1) 会受到 ϕ(st)ϕ(st) 的影响)、非静态的（highly correlated and non-stationary），这样会很容易导致训练的结果难以收敛。通过 Experience Replay 机制对存储下来的样本进行随机采样，在一定程度上能够去除这种相关性，进而更容易收敛。**当然，这种方法也有弊端，就是训练的时候是 offline 的形式，无法做到 online 的形式。

除此之外，上面算法流程图中的 aciton-value function 就是一个深度神经网络，因为神经网络是被证明有万有逼近的能力的，也就是能够拟合任意一个函数；一个 episode 相当于 一个 epoch；同时也采用了 ϵϵ-greedy 策略。代码实现可参考上面 FlappyBird 的 DQN 实现。

上面提到的 DQN 是最原始的的网络，后面Deepmind 对其进行了多种改进，比如说 Nature DQN 增加了一种新机制 **separate Target Network**，就是计算上图的yjyj 的时候不采用网络 QQ, 而是采用另外一个网络(也就是 Target Network) Q′Q′, 原因是上面计算 yjyj 和 Q 估计都采用相同的网络 QQ，这样**使得Q大的样本，y也会大，这样模型震荡和发散可能性变大**，其原因其实还是两者的关联性较大。而采用另外一个独立的网络使得训练震荡发散可能性降低，更加稳定。一般 Q′Q′ 会直接采用旧的 QQ, 比如说 10 个 epoch 前的 QQ.

除此之外，大幅度提升 DQN 玩 Atari 性能的主要就是 Double DQN，Prioritised Replay 还有 Dueling Network 三大方法；这里不详细展开，有兴趣可参考这两篇文章：[DQN从入门到放弃6 DQN的各种改进](https://zhuanlan.zhihu.com/p/21547911) 和 [深度强化学习（Deep Reinforcement Learning）入门：RL base & DQN-DDPG-A3C introduction](https://zhuanlan.zhihu.com/p/25239682)。

综上，本文介绍了强化学习中基于 value 的方法：包括 Q-Learning 以及跟 Q-Learning 非常相似的 Sarsa，同时介绍了通过 DQN 解决状态无限导致 Q-Table过大的问题。需要注意的是 DQN 只能解决动作有限的问题，对于动作无限或者说动作取值为连续值的情况，需要依赖于 policy gradient 这一类算法，而这一类算法也是目前更为推崇的算法，在下一章将介绍 Policy Gradient 以及结合 Policy Gradient 和 Q-Learning 的 Actor-Critic 方法。



##### 参考文献

- xxx
- xxx