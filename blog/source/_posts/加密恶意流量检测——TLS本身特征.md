---
title: 加密恶意流量检测——TLS本身特征
copyright: true
mathjax: true
date: 2021-02-05 17:43:32
tags:
categories:
---

概述：首页描述

![]()

<!--more-->

**论文:《Deciphering Malware’s use of TLS (without Decryption)》**

**核心点:**利用加密正常通信与加密恶意通信在TLS握手过程中以及流统计信息中的不同，对加密恶意流量进行识别。

**数据集：**

**检测级别：flow**

**论文生成检测效果：**



### 特征来源

#### 1.Flow原始数据特征

&emsp;&emsp;要提取的

- 入栈流量byte数
- 出栈流量byte数
- 入栈流量packet数
- 出栈流量packet数
- 源端口和目的端口？
- 持续时间（秒为单位）

#### 2.SPLT特征

&emsp;&emsp;SPLT全称Sequence of Packet Lengths and Times，是指包长度与包大小序列两种特征，在论文的实现中，只选择前50个payload长度不为0的包进行建模。

**特征表达方式：HMM状态转移矩阵**

&emsp;&emsp;对于长度数据，使用150字节进行分桶，(1,150]将被分到第一个桶，[150,300]将被分到第二个桶，以此类推，由于互联网MTU（最大传输单元）为1500，因此分为10个桶。然后根据桶50个桶的中各个桶之间的转移次数的得到状态转移矩阵。最后将状态转移矩阵展开作为长度转移序列特征。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.9vbj5j73d89.png)

{% note  default %}
互联网MTU为1500
{% endnote %}

#### 3. Byte Distribution





#### 4. 未加密的TLS头部信息





##### 参考文献

- xxx
- xxx