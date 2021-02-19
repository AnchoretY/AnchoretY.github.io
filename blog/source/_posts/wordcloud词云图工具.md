---
title: wordcloud词云工具
date: 2019-05-10 12:03:11
tags: [可视化,NLP]
---

wordcloud是一种NLP中常用的可视化工具，主要用途是可视化展示文本中各个词出现的频率多少，将出现频率多的使用更大的字体进行展示。



#### 基本用法

~~~python
import wordcloud
with open("./type1.txt","r") as f:
    type1 = f.read()
  
w = wordcloud.WordCloud()
w.generate(type1)
w.to_file("type1.png")
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/词云图.png?raw=true)



#### wordcloud内部处理流程：

> ​	1 、分隔：以空格分隔单词  
>
> ​	2、统计 ：单词出现的次数并过滤  
>
> ​	3、字体：根据统计搭配相应的字号  
>
> ​	4 、布局



#### 常用参数

​		

![](https://github.com/AnchoretY/images/blob/master/blog/wordcloud常用参数.png?raw=true)