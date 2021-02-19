---
title: pandas列表存储被自动转换成字符串问题
date: 2019-10-17 10:39:11
tags: [数据分析,pandas,常见问题]
---

**问题描述:**

​	在数据分析过程中发现当某一列的值为列表类型时，当存储成为csv时将自动将该列的列表存储成为对应的字符串，当重新进行读取时只能按照字符串进行处理。下面是一个真实的事例：

​	首先直接创建一列数据类型为list，直接查看dataframe中存储的数据，发现还是正常的列表类型

~~~python
>>> df = DataFrame(columns=['col1'])
>>> df.append(Series([None]), ignore_index=True)
>>> df['column1'][0] = [1, 2]
>>> df
     col1
0  [1, 2]
~~~

​	然后将dataframe进行存储后进行读取，这里可以看到列表列已经变成了字符串类型：

~~~python
>>> df.to_csv("XXX.csv")
>>> df = pd.read_csv("XXX.csv")
>>> df['column1'][0]
'[1, 2]'
~~~





**解决方案：**

​	**1.存储时不再存储为csv，存储为pickle文件**

~~~python
>>> import pickle
>>> with open("data.pkl",'wb') as file:
>>> 	pickle.dump(df,file)
>>> with open('tmp.pkl', 'rb') as file:
>>> 	new_df =pickle.load(file)
>>> new_df['col1'][0]
[1, 2]
~~~



​	**2.存储后将ast.literal_eval从str转化会list**

~~~python
>>> from ast import literal_eval
>>> literal_eval('[1.23, 2.34]')
[1.23, 2.34]
~~~

