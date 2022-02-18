---
title: '论文精读——FANCI:Feature-based Automated NXDomain Classification and Intelligence'
copyright: true
mathjax: true
date: 2022-02-18 16:17:13
tags:
categories:
---

概述：首页描述

![]()

<!--more-->

**效果：在非常低的误报的情况下可达99%准确率（实验数据集），在实际环境中使用发现了10个未包含在训练集的DGA家族**

**效率：**

- **训练：5.66min（92192个样本）**
- **预测：0.0025s/sample**

**数据来源：**

- [**DGArchive**](https://dgarchive.caad.fkie.fraunhofer.de/welcome/)(DGA来源)
- **大学网络**（正常样本来源）
- **公司内网**（正常样本来源）

正常NXDomain类型：错误输入、错误配置、误用

项目开源：







#### 特征

&emsp;&emsp;FANCI的特征全都使用了比较轻量级的特征，不需要预计算，依赖于指定的自然语言，并且全部特征可以从单个域名中进行提取。

##### 1. 结构特征

&emsp;&emsp;下表中显示FANCI中使用的全部结构特征，共12个。

> 其中比较难以理解的特征有：
>
> - 域名含有重复的前缀（#7）：子域名中的字符串序列在其他级别的子域名中出现了至少一次上。例如：rwth-aachen.derwth-aachen.de则为1.
> - 独立数字子域名的比例（#9）：纯数字组成的子域名占全部子域名数量的比例（不包含公共后缀）。例如：123.itsec.rwth-aachen.de，值为1/3
> - 是否包含IP地址（#12）：域名中是否包含IPv4、IPv6地址。

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.0q9ikcmfdeua.png" alt="image" style="zoom:67%;" />

##### 2. 语言特征

&emsp;&emsp;从语言的角度来显示DGA域名与正常域名中的语言特性的不同，共包含7个语言特征。

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.z6xvmvruy.png" alt="image" style="zoom:67%;" />

> 其中比较难以理解的特征有:
>
> - 字符重复率（#17）：域名中重复的字符占域名中（不包含公共前缀）包含的字符比率。例如：bnxd.rwth-aachen.de，字符重复率为3/12，其中重复字符为n、a、h。
> - 连续辅音比率（#18）：计算域名中连续字符长母大于2的全部字符串长度和占域名总长度（不包含公共前缀）的比例。例如：bnxd.rwth-aachen.de对应的连续辅音比率为（8+2）/15 = 0.67，其中连续字母的长度大于2的子串有：bnxdrwth and ch.
> - 连续数字比率（#19）：与连续辅音比率类似

##### 3. 统计特征

&emsp;&emsp;统计特征包括交叉熵和

<img src="/Users/yhk/Library/Application Support/typora-user-images/image-20220218180056881.png" alt="image-20220218180056881" style="zoom:67%;" />

### 系统设计

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.fp82qbxhve7.png)



#### 情报模块

目标：根据分类模型提供的分类结果找出感染设备并识别新的DGA家族或新的种子。

输入：

- NXdomain
- 源IP、目的IP、每个NXDomain响应的时间戳（**为了能够将DGA域名映射回感染设备**）

预处理：提取域名和对应NXDomain response中的有用信息。

后处理：如果NXDomain与两个白名单进行比对，如果NXDomain以whitelist中的域名结尾，则将该域名移入正常域名

> 白名单1：Alexa top N
>
> 白名单2：本地网络白名单

经过情报模块的聚合以后，输出展示结果供分析人员分析，可以以设备、域名等为单位在不同角度进行分析。



##### 参考文献

- xxx
- xxx