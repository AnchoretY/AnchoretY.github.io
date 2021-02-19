---
title: python——Counter对象
date: 2019-04-14 12:16:39
tags: [python基础]
---



​	**Counter对象就是python内部的一个计数器**，常用来统计列表、字符串中各个字符串出现的频次，以及找到出现频次最该以及最低的元素

​	使用前必须先引入引用：

~~~~python
from collections import Counter
~~~~

​	下面介绍在日常使用过程中常见的用法：

#### 1.统计列表和字符串中各个元素出现的频数

~~~python
s = "acfacs"
l = [1,1,2,4,2,7]
print(Counter(s))
print(Counter(l))

output:
  Counter({'c': 2, 'a': 2, 's': 1, 'f': 1})
	Counter({1: 2, 2: 2, 4: 1, 7: 1})
~~~



#### 2.获取最高频的N个元素及频数

​	Counter对象的most_common方法可以获取列表和字符串的前N高频的元素及频次。

> most_common:
>
> ​	param n:前几个高频对象，从1开始，默认为全部，也就相当于按照频数排序
>
> ​	return list:按照出现的频数高低已经排好序的前N个列表，列表的元素是两元组，第一项代表元素，第二项代表频率

~~~python
s = "acfacs"
print(Counter(s).most_common(1))

output:
	[('c', 2)]
~~~





