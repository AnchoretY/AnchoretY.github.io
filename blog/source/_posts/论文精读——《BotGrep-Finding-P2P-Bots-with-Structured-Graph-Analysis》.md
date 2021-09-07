---
title: '论文精读——《BotGrep: Finding P2P Bots with Structured Graph Analysis》'
copyright: true
mathjax: true
date: 2021-09-02 15:05:17
tags:
categories: [僵尸网络,图算法]
---

概述：首页描述

![]()

<!--more-->

题目：《BotGrep: Finding P2P Bots with Structured Graph Analysis》

作者：Shishir Nagaraja, Prateek Mittal, Chi-Yao Hong, Matthew Caesar, Nikita Borisov.

出处： USENIX Security Symposium

年份：2010



### 1. 整体情况

 **现有挑战：**

- 僵尸网络使用端口变化、加密等内容隐身技术逃避内容识别 
- 僵尸网络节点的拓扑与大量正常流量混合在一起

**核心思路：**先通过从主机级别的通信图中根据mixing-rate找出类似于P2P的子图，然后使用恶意软件检测程序中检测到的种子程序识别哪些p2p自图为僵尸网络。

**数据来源：**真实骨干网和CAIDA骨干网流量作为背景流量，将历史僵尸网络（Chord、de Brunijn、Kademlia、robust ring）拓扑混合进背景流量。

**优势**：内容无关，不受端口变换、加密等内容隐身技术的影响。

**缺陷**：需要与其他能够实现恶意软件识别的程序来生成种子节点，才能实现僵尸网络的发现。

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.la6vatj7tbk.png" alt="image" style="zoom:67%;" />

### 2. 具体实现

- 预过滤：从上亿个节点中提取出数量相对较少的P2P候选组，其中包含真正的P2P节点也包括误报
- 去除P2P误报：使用基于SybilInfer算法的聚类技术只对P2P节点进行聚类，去除误报
- 基于mixiing-rate对结果进行验证

#### step1： Prefiltering

**核心点：**因为faster mixing-rate算法的关联状态概率分布会更加接近，因此首先使用kmeans根据状态概率分布的相似性找出P2P候选子图。

**算法实现**：

- **计算各个节点与其他节点关联的状态概率表示$s_i$**

​	P为节点的状态转移矩阵，q为各个节点的概率分布集合，$q^t$为经过t步的随即游走后各个节点的概率分布集合，开始时各节点起始概率为$q^0_i = 1/|V|$，即每个节点的概率均相等。t步随即游走的概率计算使用下面的公式循环进行：

$$q^t = q^{t-1}P $$

​	为了减小度比较高的节点对结构化图检测的影响，这里为每个节点的概率值又添加了如下转化：

$$ s_i = （\frac{q_i^t}{d_i}）^\frac{1}{r}$$

​	其中，$d_i$为当前节点的度，r为常数。

- **根据各个节点与其他节点关联状态概率表示$s_i$进行聚类分析，获得多个P2P候选子图**

  使用$s_i$作为特征使用**k-means算法**进行聚类，最终获得k个子图的集合$V_G$，$V_G = \{G_1,G2,...,G_k\}$，其中$G_i$为P2P候选子图



#### step2：Clustering P2P Nodes

核心点：通过Prefiltering产生的子图虽然能够包含P2P节点，但是其中也很有可能包含非P2P节点。

算法实现：

- **生成Traces**

​	在图中随机选择节点进行n次随机游走，使用下面的转化矩阵来保证静态分布的随机游走每个节点的概率是均匀的：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.2zw65mesb65.png)

​	随机游走的长度为log(|V|),每个节点随机游走的数量为可调参数，自行设定。**最后每次随机游走的起始节点与终止节点构成的顶点对集合称作traces，用T表示。**

- **P2P节点的概率模型**

  ​	在这一步的目标为使用上一步生成的traces通过算法给出图的每个子图是P2P节点的概率，使用**X表示图中的部分节点**，有贝叶斯理论可以有：

  ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.tyf21wna0xa.png)

  ​	其中，**P(T)与X的选择无关为一个常量，使用Z表示；先验概率P(X=P2P)可以使用蜜罐、情报等已有知识获取；因此这个概率模型的核心就在于如何计算P(T|X=P2P)。**

  ​	根据状态概率矩阵更加相关的P2P节点是同构的直觉，P(T|X=P2P)计算采用

  <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.687mp0df7hn.png" alt="image" style="zoom:80%;" />

  ​	其中，w表示一个随即游走trace。**将随机游走的trace分为两类：1. 结束顶点在X中 2. 结束顶点不在X中。**

  ​	对于顶点在X中的trace，采用

  <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.takbjgmvkho.png" alt="image" style="zoom:80%;" />

  ​	$ N_v $表示随机游走终点在X中的数量（对全部X中的节点相同）。对顶点不在X中结束的trace，采用
  
  ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.bzeior3ikr.png)
  
  ​	$N_a$表示随即游走终点不在X中的数量。

- **Metropolis-Hastings采样算法计算Z值**

  由于Z的计算需要涉及到图的全部子集因此很难直接计算，因此采用Metropolis-Hastings采样算法抽取一部分子集$X_i$代表全部子集。给定一个样本集S，计算其中节点数学P2P节点的概率方法如下：

  <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.edjj9it5ewu.png" alt="image" style="zoom:67%;" />

  I为一个指示器，当节点i在P2P样本集$X_j$中则为1，否则为0。最后根据边界概率是否大于阈值确定样本集的fast-mixing和slow-mixing部分。

#### Step 3:

### 3. 效果

​	botgrep能够在各种环境下均能达到93%~99%的检出率，具体检测情况如下：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.j114q0cmv1.png)

特性总结：

- 随着僵尸网络的增大，检测效果小幅下降
- 检测效果与背景流量的规模大小无关
- 当对僵尸网络中的连接存在少量不可见时仅仅会使检测效果略微下降，但是大量不可见使则可能丧失检测能力
- 仅仅从蜜罐获取的已知僵尸网络节点开始随机游走可以大大降低误报率



### 4. 方法评价

