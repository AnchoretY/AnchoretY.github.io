---
title: python不显示警告信息设置
date: 2019-01-02 10:37:47
tags: [python]
---

&emsp;&emsp;在使用python处理多线程或者循环次数较多时，常常会因为系统爆出一些警告信息而影响结果的查看，比如下面的警告：

![](https://github.com/AnchoretY/images/blob/master/blog/警告信息.png?raw=true)

十分影响美观，造成结果混乱，很难找到有效的信息，下面我们使用python自带的warning设置，设置过滤warn级别的告警

~~~python
import warnings
warnings.filterwarnings("ignore")
~~~

结果变为：

![](https://github.com/AnchoretY/images/blob/master/blog/过滤warning后的结果.png?raw=true)

