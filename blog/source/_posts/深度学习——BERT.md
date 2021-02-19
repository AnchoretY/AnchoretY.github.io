---
title: 深度学习——BERT
date: 2019-05-25 11:01:33
tags: [深度学习,面试,NLP]
---

### 什么是BERT？

​	**BERT**(Bidirectional Encoder Representations from Transformer)源自论文Google2018年的论文”Pre-training of Deep **bidirectional** Transformers for Language Understanding“，其前身是Google在2017年推出的transormfer模型。

​	**核心点为：**

> 1.预训练
>
> 2.双向的编码表征
>
> 3.深度的Transformer
>
> 4.以语言模型为训练目标

### BERT的两个任务

​	1.语言模型，根据词的上下文预测这个词是什么

​	2.下一句话预测（NSP）模型接收成对的句子作为输入，并学习预测该对中的第二个句子是否是原始文档中的后续句子



### 双向attention

​	在之前常见的attention结构都是单向的attention，顺序的从左到右，而借鉴Bi_LSTM和LSTM的关系，如果能将attention改为双向不是更好吗？

​	将attention改为双向遇到的最大问题就是**深度的增加导致信息泄露问题**，如下图：

![](https://github.com/AnchoretY/images/blob/master/blog/双向attention信息泄露.png?raw=true)

解决该问题主要的解决方案有两种：

1.多层单向RNN，独立建模(ELMo)。前项后项信息不公用，分别为两个网络

![](https://github.com/AnchoretY/images/blob/master/blog/ELMo.png?raw=true)

2.Mask ML(**BERT采用**)

​	解决的问题：**多层的**self-attention信息泄漏问题

​	随机mask语料中15%的token，然后将masked token 位置输出的最终隐层向量送入softmax，来预测masked token。

​	在训练过程中作者随机mask 15%的token，而不是把像cbow一样把每个词都预测一遍。**最终的损失函数只计算被mask掉那个token。**

​	Mask如何做也是有技巧的，如果一直用标记[MASK]代替（在实际预测时是碰不到这个标记的）会影响模型，所以随机mask的时候10%的单词会被替代成其他单词，10%的单词不替换，剩下80%才被替换为[MASK]。]



![](https://github.com/AnchoretY/images/blob/master/blog/Mask_ML.png?raw=true)



### BERT整体结构

#### Input representation

​	输入表征主要由下面**三部分加和**而成：

​			**1.词的向量化编码**

> 就是常用的词向量化，例如Word2vec等或者直接embedding

​			**2.段编码**  

> 使用\[CLS]、[SEP]做标记区分段，每个段用于其各自的向量Ei，属于A段的每个词都要加EA，属于B段的每个词都要加EB...
>
> 主要是为了下句话预测任务

​			**3.位置编码**

> 和transormer不同的是，这里的position embedding是可训练的，不再是适用固定的公式计算

![](https://github.com/AnchoretY/images/blob/master/blog/BERT_input_representation.png?raw=true)



#### Transformer Encoder

​	这里还会沿用Transformer的Encoder网络，首先是一个Multi-head self-attention，然后接一个Position-wise前馈网络，并且每个结构上都有残差连接.

![](https://github.com/AnchoretY/images/blob/master/blog/Transformer Encoder.png?raw=true)



#### Losses

​	Losses就是两部分，一部分是语言模型的任务的损失，一部分是上下文是否连续的损失。

​	**语言模型的任务的损失**

​	对于Mask ML随机选择进行mask的15%的词，是否正确做损失函数(一般为交叉熵损失函数)

​	**上下文是否连续损失**

​	二分类的损失函数，连续/不连续



### 常见问题

##### 1.Bert的mask ml相对Cbow有什么相同和不同？

​	**相同点**：两种方式都采用了使用一个词周围词去预测其自身的模式。

​	**不同点**：1.mask ml是应用在多层的bert中，用来防止 transformer 的全局双向 self-attention所造成的信息泄露的问题；而Cbow时使用在单层的word2vec中，虽然也是双向，但并不存在该问题

​					2.cbow会将语料库中的每个词都预测一遍，而mask ml只会预测其中的15%的被mask掉的词

##### 2.Bert针对以往的模型存在哪些改进？

​	1.创造性的提出了mask-ml来解决多层双向 self-attention所出现的信息泄露问题

​	2.position embedding采用了可训练的网络取到了余弦函数公式



##### 3.Bert的双向体现在那里？

​	Bert的双向并不是说他和transformer相比，模型结构进行了什么更改，而是transformer原始的Encoder部分在使用到语言模型时就是一种双向的结构，而本身transformer之所以不是双向的是因为他并不是每个单词的语言建模，而是一种整体的表征，因此不存在单向双向一说



##### 4.对输入的单词序列，随机地掩盖15%的单词，然后对掩盖的单词做预测任务，预训练阶段随机用符号[MASK]替换掩盖的单词，而下游任务微调阶段并没有Mask操作，会造成预训练跟微调阶段的不匹配，如何金额绝？

​	15%随机掩盖的单词并不是都用符号[MASK]替换，掩盖单词操作进行了以下改进：

​		*80%用符号[MASK]替换：my dog is hairy -> my dog is [MASK]*

​		*10%用其他单词替换：my dog is hairy -> my dog is apple*

​		*10%不做替换操作：my dog is hairy -> my dog is hairy*

##### 5.手写muti-attention

>
>
>



**6、 elmo、GPT、bert三者之间有什么区别？（elmo vs GPT vs bert）**

（1）**特征提取器**：elmo采用LSTM进行提取，GPT和bert则采用Transformer进行提取。很多任务表明Transformer特征提取能力强于LSTM，elmo采用1层静态向量+2层LSTM，多层提取能力有限，而GPT和bert中的Transformer可采用多层，并行计算能力强。

（2）**单/双向语言模型**：

- GPT采用单向语言模型，elmo和bert采用双向语言模型。但是elmo实际上是两个单向语言模型（方向相反）的拼接，这种融合特征的能力比bert一体化融合特征方式弱。
- GPT和bert都采用Transformer，Transformer是encoder-decoder结构，GPT的单向语言模型采用decoder部分，decoder的部分见到的都是不完整的句子；bert的双向语言模型则采用encoder部分，采用了完整句子。





