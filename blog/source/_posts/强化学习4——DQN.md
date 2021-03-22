---
title: 强化学习4——DQN
copyright: true
mathjax: true
date: 2021-03-22 20:17:46
tags:
categories:
---

概述：首页描述

![]()

<!--more-->

## DQN

&emsp;&emsp;DQN全程Deep Q-Learning Network，这种强化学习方式被提出是为了解决当Q-table中状态过多，导致整个Q-Table无法装入内存的问题，在DQN中采用了一个深度神经网络来对Q-Table进行拟合，具体来说就是：向神经网络中输入当前状态，输出为各种操作对应的概率值。

**原论文： [Playing atari with deep reinforcement learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)**

**使用场景：状态无限、动作有限**

**核心思想**：DQN的模型训练过程与Q Learning本质上相同，都是通过**计算Q现实与Q估计的差值，进行反向传播，从而完成模型的训练过程**。

---



### DQN中的两个关键点

#### 1. Experience Replay

&emsp;&emsp;在DQN中与之前的强化学习算法不同，DQN首先进行反复的实验，每一次实验的结果都是一个sample，每个sample都是一个四元组：**当前状态、当前状态对应各种action的Q值、当前状态采取action的即时回报、下一状态各种action的Q值**。将这些sample存储在Memory中，经过一定step以后，再从Memory中随机抽取Sample来进行反向传播，更新eval_net。

**为什么要使用Experience Replay？**

&emsp;&emsp;**按照时间顺序生成的样本是有关系的，后一时刻的Sample中的内容会收到前一时刻的影响，因此很容易导致难以收敛的问题**，因此使用Experience Replay进行先将之前的Sample进行存储下来，然后再对存储下来的样本进行随机采样，从而在一定程度上去除掉这种相关性，使网络更容易收敛。

&emsp;&emsp;但是，与之相伴，因为要存储一定的Memory才能进行随机采样更新网络，因此导致**DQN训练只能offline，无法做到online训练**。

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.94fvo2vpl5a.png" alt="image" style="zoom:33%;" />

#### 2. Fixed Q-target

&emsp;&emsp;Fixed Q-target是另一种打乱sample之间相关性的策略，Fixed Q-target在 DQN 中**使用到两个结构相同但参数不同的神经网络**,分别称为eval_net和target_net, eval_net每次选择action后都会进行及时的反向传播，更新网络参数，代表最新状态下的预测值，而target_net各一段时间才将其参数与eval_net进行同步，代表一段时间以前的预测值。

> eval_net: 样本采取每个action后会都反向传播更新网络参数，保持参数最新，代表了最新状态下的预测值
>
> target_net:  不进行反向传播，每隔一定时间将网络参数与eval_net进行更新，代表了一段时间以前的预测值

**为什么要使用Fixed Q-target？**

&emsp;&emsp;Fixed Q-target策略使用两个独立的网络来表示Q现实与Q估计在一定程度上降低了二者之间的相关性，使网络出现震荡的可能性降低。

---

### 算法实现

1. DQN模型设计

   ~~~
   
   ~~~

   



##### 参考文献

- xxx
- xxx