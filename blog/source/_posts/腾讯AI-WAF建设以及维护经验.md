---
title: 腾讯AI-WAF建设以及维护经验
copyright: true
mathjax: true
date: 2020-07-02 15:41:12
tags:
categories:
---

概述：本文来自于腾讯安全应急响应中心发布的两篇博客，主要对其中比较有启发性的一些问题做总结。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.d05qy9e6y2t.png)

<!--more-->

&emsp;&emsp;在本篇文章中提出了一种使用使用语义、策略、AI三种方式进行协作的AI WAF建设方式，其主要针对于XSS、SQL等具有明显的语义结构的攻击形式。



### 整体结构

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.m43x64ktusl.png)

#### 1.流量压缩

&emsp;&emsp;这里是大多数实际应用的WAF产品所必需的第一步，因为在真实的互联网环境中，正常流量与攻击流量的比例大约在10000：1，因此一般WAF产品都**使用一定的策略大大减少需要使用WAF进行判断的流量，增加整个系统的处理效率**。在腾讯的门神WAF中提到所使用的方法为：过滤公司出口IP、敏感攻击特征关键字进行字符串匹配（**注意这里是敏感关键字匹配，不是正则，敏感关键字匹配的效率比正则表达式要高**）

#### 2.请求预处理

&emsp;&emsp;请求预处理阶段是无论传统WAF还是AI WAF系统中都需要进行的检测准备，主要包括**解析处理**和**解码处理**两部分。

**解析处理**：对http请求按协议规范**解析提取出各个字段字段的Key-Value**，包括json的一些特殊处理等。

**解码处理**：解码处理主要是`为了避免payload通过各种编码绕过检测`,针对**URL编码、URL多重编码、base64编码、unicode编码、html实体编码**，通过解码阶段处理最终**还原出原始payload**，再输出给后面模块处理。

{% note  info %}
解码通常使用循环解码来保证编码已经被完全解析。
{% endnote %}

##### 容易产生的攻击

&emsp;&emsp;由于采用循环解码的方式进行解码，可能在将循环解析结构输入到语义分析引擎中进行分析时，由于在WAF的预处理阶段解码次数与后端解码次数不一致导致绕过漏洞。具体实例如下：

~~~python
alert('%27')

      => alert('%27') // 语法正确

      => alert(''')   // 再进行一次 url 解码，语法错误

%3Csvg/onload=alert(1)%25111%3E

      =>  <svg/onload=alert(1)%111>  // 进行一次 url 解码，语法正确

      => <svg/onload=alert(1)•1>    // 再进行一次 url 解码，语法错误

      => alert(1)•1
~~~

#### 3.词法分析和文法分析

&emsp;&emsp;词法分析：是指读入源程序，识别出单词，并用记号token方式表示识别出的单词。

&emsp;&emsp;语法分析：在词法分析的基础上，根据语言的语法规则，把单词符号串组成各类语法单位，即在单词流的基础上建立一个层次结构-语法树。

{% note  info %}
这里用到的词法、语法分析工具为**Antlr4**，可以根据需求编写文法，描述我们要解析的语言语法，antlr4会自动生成词法分析器和语法分析器
{% endnote %}

&emsp;&emsp;以下面的payload为例：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.678jyfqhb8l.png)

&emsp;&emsp;经解析、解码处理，**对html内容进行解析，提取JS内容**，**包括：script标签内，on事件内，src/href, data-uri base64编码等**，进行词法分析：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.2r70q9cqbja.png)

&emsp;&emsp;再经过语法分析：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.fy4ad1ow576.png)

#### 4.基于机器学习的打分评判

&emsp;&emsp;这里腾讯所使用的方式与之前我做方法非常类似，都是**使用HMM作为打分模型，以人工经验去拟定阈值**，评分高于阈值则认为为异常，然后再使用过程中根据实际反馈去不断调整阈值。

{% note  info %}
这里之所以会使用HMM模型来进行打分与是因为HMM的序列性，不仅仅可以和一般地模型一样表示里面的敏感词是否出现以及出现次数，还可以很好的表征出词出现的先后顺序
{% endnote %}

&emsp;&emsp;这篇文章主要针对于XSS，**特征工程采用对payload根据XSS攻击模式进行分词，应用专家经验知识干预和特征提取技巧进行特征化**，如可以采用基于词集模型编码。



### 成果

&emsp;&emsp;最终测试攻击检出率可以达到99%，同时误报率控制在0.03%以下。最终提供了一个简单的demo接口，输出风险情况和完整的语法树结构：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.e4tdsh5o35g.png)



### 启发

- 在使用AI进行检测之前必须使用一些高效率的策略对网络流量进行粗筛
- 解码部分采用循环解码保证解码完全，但是同时也要考虑到因为循环解码而产生的安全性问题
- 采用众测的方式来当前AI WAF存在的不足之处，从而快速进行完善



##### 参考文献

- [WAF建设运营及AI应用实践](https://mp.weixin.qq.com/s?__biz=MjM5NzE1NjA0MQ==&mid=2651199346&idx=1&sn=99f470d46554149beebb8f89fbcb1578&chksm=bd2cf2d48a5b7bc2b3aecb501855cc2efedc60f6f01026543ac2df5fa138ab2bf424fc5ab2b0&scene=21#wechat_redirect)
- [门神WAF众测总结](https://mp.weixin.qq.com/s/w5TwFl4Ac1jCTX0A1H_VbQ)