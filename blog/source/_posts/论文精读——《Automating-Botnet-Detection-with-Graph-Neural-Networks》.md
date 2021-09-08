---
title: 论文精读——《Automating Botnet Detection with Graph Neural Networks》
copyright: true
mathjax: true
date: 2021-08-31 10:25:57
tags: [僵尸网络,图算法,图神经网络]
categories: [僵尸网络,图算法,图神经网络]
---

概述：提出了一种定制的图神经网络在大规模背景流量通信图中识别出僵尸网络的通信拓扑特征，可以较为准确的对集中式僵尸网络分层架构以及分散式僵尸网络快速混合架构做出识别。

![]()

<!--more-->

题目：《Automating Botnet Detection with Graph Neural Networks》

作者：Jiawei Zhou, Zhiying Xu, Alexander M. Rush, Minlan Yu

出处：AutoML for Networking and Systems Workshop of **MLSys** 2020 Conference

年份：2020

github: https://github.com/harvardnlp/botnet-detection

### 整体情况

**现状**：

​	传统僵尸网络检测通过基于先验知识的多阶段检测进行，很大程度上依赖于先验知识对流量中的模式识别、以及对于域名、DNS查询的威胁情报等。2007年以来，虽然有些研究工作开始从拓扑角度构建特征来识别僵尸网络，但是仍然是通过人工先去定义拓扑衡量的特征指标，例如：混合率、连通图大小数量等。

- **《Botgrep: Finding p2p bots with structured graph analysis》，*USENIX Security Symposium*,2010**

  从mixing rate角度进行p2p僵尸网络检测，P2P僵尸网络通常有更大的mixing rate，因为P2P僵尸网络的需要快速的传播消息和发布攻击指令。

- **《Hit-list worm detection and bot identification in large networks using protocol graphs》，International Workshop on Recent Advances in Intrusion Detection，2007**

  从组成连通图的数量及大小角度进行僵尸网络检测

- **《Graption: Automated detection of p2p applications using traffic dispersion graphs》，2008**

- **《A graph-theoretic framework for isolating botnets in a network》，Security and communication networks，2015**

**主要贡献**：

- 提出了一种全自动僵尸网络检测方式，给定一个通信图，能够检测出其中的僵尸网络节点，不要任何先验知识进行特征提取
- 提出了一种仅根据拓扑结构基于无属性图的专为僵尸网络设计的检测算法，与之前的方法相比能够在保持误报率不变的前提下提升检出率。
- 提出了一个大规模僵尸网络图检测数据集，包含了14w个node，70w个边。

**原理**：

无论是集中式僵尸网络还是分散式僵尸网络都会与背景流量表现出不同的拓扑特性。

- 集中式僵尸网络具有明显的星型结构
- 分散式的僵尸网络混合速率明显高于背景流量

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.0tgm7jyh6yx.png" alt="image" style="zoom:67%;" />

### 具体实现

1. **通信图抽象**

   这里抽象了一个通信**无向图**模型，

   <center> $$ G = \{V,A\} $$ </center>

   其中，V为不同的通信主机节点集合，A矩阵表示两个节点间是否有通信行为，D矩阵表示每个阶段的度。

2. **检测目标**

   输入通信图，**仅通过拓扑信息**检测中图中那些节点为僵尸网络节点。

   ​	**输出**：通信图

   ​	**输出**：僵尸网络节点

3. **图神经网络模型**

   使用经典的图神经网络模型作为检测模型，具体细节如下：

   - 12层GNN
   - 每两层之间使用relu作为非线性激活函数，并使用bias
   - 全部层的embedding大小都为32
   - 最后使用线性层对图信息进行整合判断

   此外，在标准的GNN基础上做了两个改变：

   - 采用$ \overline{A} = D^{-1}A$进行随即游走标准化，其中只涉及到源节点的度归一化邻接矩阵到对应的概率发射矩阵。
   - 将第一层的输入全部设置为1，从而保证与输入节点的顺序无关。

   > **乘以矩阵的逆的本质，就是做矩阵除法完成归一化。左右分别乘以节点i,j度的开方，就是考虑一条边的两边的点的度。**

4. **数据集**

   作者在真实的背景网络拓扑中了合成的以及真实的僵尸网络拓扑形成了数据集。

   - 背景流量：CAIDA (2018)’s 骨干网络通信流量，形成通信图，然后在其中随机选择节点子集替换为僵尸网络节点
   - 僵尸网络流量：
     - 合成的P2P网络拓扑：DE BRUIJN、KADEMLIA、CHORD、LEET- CHORD
     - 真实的僵尸网络：一个去中心化的僵尸网络和一个中心化的僵尸网络

5. **实验结果**

   ​	作者将本文提出的僵尸网络检测方法与一种机器学习的LR模型检测方法、2010年提出多阶段图分析方法BotGrep进行了对比，LR检测方法效果最差，本文提出的端到端的GNN检测方法在多数情况下都要比BotGrep具有更低的误报率和更高的检出率。

   <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.52max38a31c.png" alt="image" style="zoom:30%;" />

   ​	检测性能不受节点僵尸网络规模影响：该GNN检测模型虽然是在1k个节点规模的僵尸网络拓扑上进行训练的，但是依然能够对较小规模的（100个节点）僵尸网络中的节点完成较好的检测。

### 论文分析

​	本文将僵尸网络检测问题抽象为从网络通信图拓扑角度进行节点二分类的问题，考虑了是否存在通讯在实验环境下来看检测效果还比较理想，但是由于使用的图为无向无权图，仅仅考虑了主机之间是否存在通信，没有考虑实际环境环境中僵尸网络节点也会和正常网络节点存在少量通信，因此可能在实际使用中会与实验环境中差距较大，完善可以考虑根据通信次数、通信频率等构建有全图来进行提升检测实际环境中的检测效果。