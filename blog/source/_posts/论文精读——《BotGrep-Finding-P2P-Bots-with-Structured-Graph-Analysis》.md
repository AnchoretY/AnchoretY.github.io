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

​	





##### 参考文献

- xxx
- xxx