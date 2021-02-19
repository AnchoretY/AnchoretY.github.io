---
title: pyspark dataframe操作进阶
date: 2018-09-26 21:06:18
tags: spark
categories: spark
---

​	这一节主要讲的是spark在机器学习处理过程中常用的一列操作，包括获得各种预处理。

**1.将多列转化成一列**

​	pyspark可以直接使用VectorAssembler来将多列数据直接转化成vector类型的一列数据。

```python
from pyspark.ml.feature import VectorAssembler

discretization_feature_names = [
    'discretization_tag_nums',
    'discretization_in_link_nums',
    'discretization_out_link_nums',
    'discretization_style_nums',
    'discretization_local_img_nums',
    'discretization_out_img_nums',
    'discretization_local_script_nums',
    'discretization_in_script_nums',
    'discretization_out_script_nums'
]

vecAssembler = VectorAssembler(inputCols=discretization_feature_names, outputCol="feature_vec_new")

data_set  = vecAssembler.transform(dataset) #会返回一个在原有的dataframe上面多出来一列的新dataframe

data_set.printscheama()

output：
    root
     |-- discretization_tag_nums: double (nullable = true)
     |-- discretization_in_link_nums: double (nullable = true)
     |-- discretization_out_link_nums: double (nullable = true)
     |-- discretization_style_nums: double (nullable = true)
     |-- discretization_local_img_nums: double (nullable = true)
     |-- discretization_out_img_nums: double (nullable = true)
     |-- discretization_local_script_nums: double (nullable = true)
     |-- discretization_in_script_nums: double (nullable = true)
     |-- discretization_out_script_nums: double (nullable = true)
     |-- feature_vec_new: vector (nullable = true) #多出来的


```





**2.连续数据离散化**

​	pyspark中提供QuantileDiscretizer来根据分位点来进行离散化的操作，可以根据数据整体情况来对某一列进行离散化。

> 常用参数：
>
> ​	numBuckets：将整个空间分为几份，在对应的分为点处将数据进行切分
>
> ​	relativeError：
>
> ​	handleInvalid：	

```python
from pyspark.ml.feature import QuantileDiscretizer


qds = QuantileDiscretizer(numBuckets=3,inputCol=inputCol, outputCol=outputCol, relativeError=0.01, handleInvalid="error")

#这里的setHandleInvalid是代表对缺失值如何进行处理
#keep表示保留缺失值
dataframe = qds.setHandleInvalid("keep").fit(dataframe).transform(dataframe)

```



**3.增加递增的id列**

​	monotonically_increasing_id() 方法给每一条记录提供了一个唯一并且递增的ID。

```python
from pyspark.sql.functions import monotonically_increasing_id

df.select("*",monotonically_increasing_id().alias("id")).show()
```



**4.指定读取或创建dataframe各列的类型**

​	pyspark可以支持使用schema创建StructType来指定各列的读取或者创建时的类型，一个StructType里面包含多个StructField来进行分别执行列名、类型、是否为空。

```python
from pyspark.sql.types import *
schema = StructType([
    StructField("id",IntegerType(),True),
    StructField("name",StringType(),True)
])

df = spark.createDataFrame([[2,'a'],[1,'b']],schema)
df.printSchema()

output:
    root
     |-- id: integer (nullable = true)
     |-- name: string (nullable = true)

```



**5.查看各类缺失值情况**

```python
import pyspark.sql.functions as fn

data_set.agg(*[(1-(fn.count(i)/fn.count('*'))).alias(i+"_missing") for i in data_set.columns]).show()
```

​	注：在其中agg(*)里面的 ”\*“代表将该列表处理为一组独立的参数传递给函数



**6.使用时间窗口来进行分组聚合**

​	

> 这也是pyspark比pandas多出来的一个时间窗口聚合的使用

```
src_ip_feature = feature_data.groupby("srcIp",F.window("time", "60 seconds")).agg(
    F.count("distIp").alias("request_count"),
```



**7.过滤各种空值**

​	**（1）过滤字符串类型的列中的某列为null行**

​		这里要借助function中的isnull函数来进行

```python
import pyspark.sql.function as F

df.filter(F.isnull(df["response_body"])).show()    #只留下response_body列为null的

df.filter(~F.isnull(df["response_body"])).show()    #只留下response_body列不为null的

```

​	

**8.列名重命名**

​	借助selectExpr可以实现在select的基础上使用sql表达式来进行进一步的操作这一特性，将列名进行修改

~~~python
#将count列重命名为no_detection_nums,webshelll_names列名不变
df = df.selectExpr("webshell_names","count as no_detection_nums")
~~~

