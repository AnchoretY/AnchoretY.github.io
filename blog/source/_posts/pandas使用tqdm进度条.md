---
title: pandas使用tqdm进度条
copyright: true
mathjax: true
date: 2020-09-04 18:23:14
tags:
categories:
---

概述：tqdm工具常用来在python中执行for循环时生成进度条工具，从而显式的观察到任务的完成状态以及全部完成需要的时间，现在tqdm同样提供了对pandas中apply和gooupby的支持，这里简要说明其使用方法。

![]()

<!--more-->

### pandas 引入进度条

&emsp;&emsp;在pandas中使用tqdm进度条工具很简单，直接使用tqdm.pandas接口，然后将apply替换成progress_apply即可，使用的实例代码如下：

~~~python
from tqdm import tqdm
from tqdm.gui import tqdm as tqdm_gui

tqdm.pandas(ncols=50)
df1["dns_record_nums"] = df1.progress_apply(lambda x:get_DNS_Record_Nums(x.domain),axis=1)
~~~