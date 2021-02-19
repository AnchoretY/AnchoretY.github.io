---
title: NLP—关键词提取算法
date: 2018-11-01 11:19:25
tags: [机器学习,NLP]
categories: [机器学习,NLP]
---

### 关键词提取算法

---

#### tf-idf(词频-逆文档频率)

​	![tf计算公式](https://github.com/AnchoretY/images/blob/master/blog/TF%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.png?raw=true)

​	其中count(w)为关键词出现的次数，|Di|为文档中所有词的数量。

​	![tf计算公式](https://github.com/AnchoretY/images/blob/master/blog/IDF%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.png?raw=true)

​	其中，N为所有文档的总数，I(w,Di)表示文档Di是否包含该关键词，，包含则为1，不包含则为0，若词在所有文档中均未出现，则IDF公式中分母则为0，因此在分母上加1做平滑(smooth)

​	最终关键词在文档中的tf-idf值：

​	![tf计算公式](https://github.com/AnchoretY/images/blob/master/blog/IDF%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.png?raw=true)

> **tf-idf特点：**
>
> ​	**1.一个词在一个文档中的频率越高，在其他文档中出现的次数越少，tf-idf值越大**
>
> ​	**2.tf-idf同时兼顾了词频和新鲜度，可以有效地过滤掉常见词**

​			

---



#### TextRank

​	TextRank算法借鉴于Google的PageRank算法，主要在考虑词的关键度主要考虑**链接数量**和**链接质量（链接到的词的重要度）**两个因素。

​	TextRank算法应用到关键词抽取时连个关键点：1.词与词之间的关联没有权重（即不考虑词与词是否相似）  2.每个词并不是与文档中每个次都有链接关系而是只与一个特定窗口大小内词与才有关联关系。



> TextRank特点：
>
> ​	1.**不需要使用语料库进行训练**，由一篇文章就可以提取出关键词
>
> ​	2.由于TextRank算法涉及到构建词图以及迭代计算，因此**计算速度较慢**
>
> ​	3.虽然考虑了上下文关系，但是**仍然将频繁次作为关键词**
>
> ​	**4.TextRank算法具有将一定的将关键词进行合并提取成关键短语的能力**

​	

---

