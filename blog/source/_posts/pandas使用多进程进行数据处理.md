---
title: pandas使用多进程进行数据处理
copyright: true
mathjax: true
date: 2020-09-04 18:23:29
tags:
categories:
---

概述：pandas是一种非常常用的数据分析工具，但是pandas本身并没有任何多线程，因此对于数据量比较大的数据很难进行高效的数据处理，因此本文提供了一种使用joblib库使pandas计算多进程计算的方式，使其可以高效的进行数据处理。

![]()

<!--more-->

### joblib工具

&emsp;&emsp;joblib是一种Python高效计算的工具。主要用于Python多进程并行计算、高效对象持久化。

#### 1.多进程高效计算

&emsp;&emsp;joblib最常用的功能就是

~~~python
>>> from joblib import Parallel, delayed
>>> from math import sqrt
>>> Parallel(n_jobs=1)(delayed(sqrt)(i**2) for i in range(10))
[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
~~~



#### 2.高效的对象持久化

&emsp;&emsp;joblib的另外一个常用功能就是用来替代pickle对对象的持久化与读取，使对象持久化以及持久化对象读取过程更加高效，使用方式与pickle保持一致。

~~~python
joblib.dump()	# 对象进行持久化
joblib.load()	# 读取持久化对象
~~~

### 使用joblib对pandas进行多进程计算

~~~python
from joblib import Parallel,delayed

def tmp_func(df1):
    tqdm.pandas(ncols=50)
    df1["dns_record_nums"] = df1.progress_apply(lambda x:get_DNS_Record_Nums(x.domain),axis=1)
    return df1
                  
def apply_parallel(df_grouped,func):
    results = Parallel(n_jobs=30)(delayed(func)(group) for name,group in df_grouped)
    return pd.concat(results)

# 每条数据一个分组
df_grouped = df1.groupby(df1.index)
%time df1 = apply_parallel(df_grouped,tmp_func)
~~~



##### 参考文献

- xxx
- xxx