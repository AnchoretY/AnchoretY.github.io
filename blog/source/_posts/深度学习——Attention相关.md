---
title: 深度学习——Attention相关
date: 2019-01-21 14:54:55
tags: [深度学习,面试]
categories: 深度学习
---



#### 1.为什么要使用Attention机制？

​	Attention机制最初起源于seq2seq中，经典的encoder-decoder做机器翻译时，通常是是使用两个RNN网络，一个用来将待翻译语句进行编码输出一个vector，另一个RNN对上一个RNN网络的输出进行解码，也就是翻译的过程。但是经典的encoder-decoder模式**最大的缺点**在于：**不管输入多么长的语句，最后输出的也只是最后一个vector，这个向量能否有效的表达该语句非常值得怀疑**，而**Attention机制正是利用了RNN整个过程中的各个输出来综合进行编码**

> 原始序列模型的不足：
>
> ​	1.从编码器到解码器的语境矩阵式大小是固定的，这是个瓶颈问题
>
> ​	2.难以对长的序列编码，并且难以回忆长期依赖



#### 2.Attention原理

**1.首先在RNN的过程中保存每个RNN单元的隐藏状态(h1….hn)**

**2.对于decoder的每一个时刻t，因为此时有decoder的输入和上一时刻的输出，所以我们可以的当前步的隐藏状态St**

**3.在每个t时刻用St和hi进行点积得到attention score**

![](https://github.com/AnchoretY/images/blob/master/blog/Attention1.png?raw=true)

**4.利用softmax函数将attention score转化为概率分布**

​	利用下面的公式进行概率分布的计算：

![](https://github.com/AnchoretY/images/blob/master/blog/Attention公式1.png?raw=true)

![](https://github.com/AnchoretY/images/blob/master/blog/attention2.png?raw=true)



**5.利用刚才的计算额Attention值对encoder的hi进行加权求和，得到decoder t时刻的注意力向量（也叫上下文向量）**

​	![](https://github.com/AnchoretY/images/blob/master/blog/Attention公式2.png?raw=true)

![](https://github.com/AnchoretY/images/blob/master/blog/Attention3.png?raw=true)

**6.最后将注意力向量和decoder t时刻的隐藏状态st并联起来做后续步骤（例如全连接进行分类）**

![](https://github.com/AnchoretY/images/blob/master/blog/Attention4.png?raw=true)



#### 3.Attention计算方式

​	前面一节中，我们的概率分布来自于h与s的点积再做softmax，这只是最基本的方式。在实际中，我们可以有不同的方法来产生这个概率分布，每一种方法都代表了一种具体的Attention机制。在各个attention中，attention的计算方式主要有**加法attention**和**乘法attention**两种。

##### 3.1 加法attention

​	在加法attention中我们不在使用st和hi的点乘，而是使用如下计算:

![](https://github.com/AnchoretY/images/blob/master/blog/加法attention.png?raw=true)

​	其中,va和Wa都是可以训练的参数。使用这种方式产生的数在送往softmax来进行概率分布计算

##### 3.2 乘法attention

​	在乘法attention中使用h和s做点乘运算:

![](https://github.com/AnchoretY/images/blob/master/blog/乘法attention.png?raw=true)

​	显然**乘法attention的参数更少，计算效率更高。**



#### 4.self-attention

​	思想：在没有任何额外信息情况下，句子使用self-attention机制来处理自己，提取关键信息



> 在attention机制中经常出现的一种叫法：
>
> ​	query：在一个时刻不停地要被查询的那个向量（前面的decodert时刻的隐藏状态st）。
>
> ​	key: 要去查询query计算个query相似关度的向量（前面的encoder在各个时刻的隐藏状态hi）
>
> ​	value: 和softmax得到的概率分布相乘得到最终attention上下文向量的向量(前面的encoder在各个时刻的隐藏状态hi)
>
> 这里我们可以明显知道，**任意attention中key和value是相同的**

​	attention就是key、value、和query都来自同一输入的(也是相同的)







