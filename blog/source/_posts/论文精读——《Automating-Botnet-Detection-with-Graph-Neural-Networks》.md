---
title: 论文精读——《Automating Botnet Detection with Graph Neural Networks》
copyright: true
mathjax: true
date: 2021-08-31 10:25:57
tags: 僵尸网络检测
categories: 僵尸网络检测
---

概述：提出了一种定制的图神经网络在大规模背景流量通信图中识别出僵尸网络的通信拓扑特征，可以较为准确的对集中式僵尸网络分层架构以及分散式僵尸网络快速混合架构做出识别。

![]()

<!--more-->

题目：《Automating Botnet Detection with Graph Neural Networks》

作者：Jiawei Zhou, Zhiying Xu, Alexander M. Rush, Minlan Yu

出处：AutoML for Networking and Systems Workshop of **MLSys** 2020 Conference

年份：2020

github:

### 论文详解

**现状**：传统僵尸网络检测通过基于先验知识的多阶段检测进行，很大程度上依赖于先验知识对流量中的模式识别、以及对于域名、DNS查询的威胁情报等。2007年以来，虽然有些研究工作开始从拓扑角度构建特征来识别僵尸网络，但是仍然是通过人工先去定义拓扑衡量的特征指标，例如：混合率、连通图大小数量等。

- **《Botgrep: Finding p2p bots with structured graph analysis》，*USENIX Security Symposium*,2010**

  从mixing rate角度进行p2p僵尸网络检测，P2P僵尸网络通常有更大的mixing rate，因为P2P僵尸网络的需要快速的传播消息和发布攻击指令。

- **《Hit-list worm detection and bot identification in large networks using protocol graphs》，International Workshop on Recent Advances in Intrusion Detection，2007**

  从组成连通图的数量及大小角度进行僵尸网络检测

- **《Graption: Automated detection of p2p applications using traffic dispersion graphs》，2008**

- **《A graph-theoretic framework for isolating botnets in a network》，Security and communication networks，2015**

**主要贡献**：

- 提出了一种全自动僵尸网络检测方式，不要任何先验知识进行特征提取
- 提出了一种仅根据拓扑结构基于无属性图的专为僵尸网络设计的检测算法，与之前的方法相比能够在保持误报率不变的前提下提升检出率。
- 提出了一个大规模僵尸网络图检测数据集，包含了14w个node，70w个边。

**原理**：

无论是集中式僵尸网络还是分散式僵尸网络都会与背景流量表现出不同的拓扑特性。

- 集中式僵尸网络具有明显的星型结构
- 分散式的僵尸网络混合速率明显高于背景流量

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.0tgm7jyh6yx.png" alt="image" style="zoom:67%;" />

### 模型

1. #### 抽象问题

   这里抽象了一个通信图模型，

   <center> $$ G = \{V,A\} $$ </center>

   其中，V为不同的通信主机节点集合，A矩阵表示两个节点间是否有通信行为，D矩阵表示每个阶段的度。

2. #### 图神经网络模型

   在标准的GNN基础上做了两个改变：

   - 采用$ \overline{A} = D^{-1}A$进行随即游走标准化，其中只涉及到源节点的度归一化邻接矩阵到对应的概率发射矩阵。
   - 将第一层的输入全部设置为1，从而保证与输入节点的顺序无关





##### 参考文献

- xxx
- xxx