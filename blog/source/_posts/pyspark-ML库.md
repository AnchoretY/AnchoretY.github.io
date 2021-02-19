---
title: pyspark ML库
date: 2018-09-30 22:08:36
tags: spark
categories: spark
---

​	pyspark的ML软件包主要用于针对spark dataframe的建模（MLlib主要还是针对RDD，准备废弃）,ML包主要包含了**转化器Transformer**、**评估器Estimater**和**管道Pipline**三个部分。



#### 1.转化器

​	转换器通常通过将一个新列附加到DataFrame来转化数据，每个转化器都必须实现.transform()方法。

​	**使用在预处理截断**

>.transorm()方法常用的参数：
>
>​	1.dataframe   这是唯一一个强制性参数（也可以不理解为参数）
>
>​	2.inputCol 	输入列名
>
>​	3.outputCol		输出列名

​	要使用转化器首先需要引入宝feature

```python
from pyspark.ml.feature import ...
```

**(1)Binarizer**

​	根据指定阈值将连续变量进行**二值化**

>注：这里需要输入的那一列的数据类型为DoubleType,InterType和FloatType都不支持

```python
df = spark.createDataFrame([[2.0,'a'],[1.0,'b'],[4.0,'b'],[9.0,'b'],[4.3,'c']],schema=schema)
binarizer = Binarizer(threshold=4.0,inputCol='id',outputCol='binarizer_resulit')
binarizer.transform(df).show()

output:
    +---+---+-----------------+
    | id|age|binarizer_resulit|
    +---+---+-----------------+
    |2.0|  a|              0.0|
    |1.0|  b|              0.0|
    |4.0|  b|              0.0| 		#当值与阈值相同的时候向下取
    |9.0|  b|              1.0|
    |4.3|  c|              1.0|
    +---+---+-----------------+
```



**(2)Bucketizer**

​	根据阈值列表将连续变量值**离散化**

> 注：splits一定要能包含该列所有值

```python
df = spark.createDataFrame([[2.0,'a'],[1.0,'b'],[4.0,'b'],[9.0,'b'],[4.3,'c']],schema=schema)
bucketizer = Bucketizer(splits=[0,2,4,10],inputCol='id',outputCol='bucketizer_result')
bucketizer.setHandleInvalid("keep").transform(df).show()

output:
    +---+---+-----------------+
    | id|age|bucketizer_result|
    +---+---+-----------------+
    |2.0|  a|              1.0|
    |1.0|  b|              0.0|
    |4.0|  b|              2.0|
    |9.0|  b|              2.0|
    |4.3|  c|              2.0|
    +---+---+-----------------+
```

**(3)QuantileDiscretizer**

​	根据数据的近似分位数来将离散变量转化来进行**离散化**

```python
df = spark.createDataFrame([[2.0,'a'],[1.0,'b'],[4.0,'b'],[9.0,'b'],[4.3,'c']],schema=schema)
quantile_discretizer = QuantileDiscretizer(numBuckets=3,inputCol='id',outputCol='quantile_discretizer_result')
bucketizer.setHandleInvalid("keep").transform(df).show()

output:
    +---+---+-----------------+
    | id|age|bucketizer_result|
    +---+---+-----------------+
    |2.0|  a|              1.0|
    |1.0|  b|              0.0|
    |4.0|  b|              2.0|
    |9.0|  b|              2.0|
    |4.3|  c|              2.0|
    +---+---+-----------------+
```



**(4)Ngram**

​	将一个字符串列表转换为ngram列表，以空格分割两个词,一**般要先使用算法来先分词**，然后再进行n-gram操作。

> 注：1.空值将被忽略，返回一个空列表
>
> ​	2.输入的列必须为一个ArrayType(StringType())

```
df = spark.createDataFrame([
    [['a','b','c','d','e']],
    [['s','d','u','y']]
],['word'])

ngram = NGram(n=2,inputCol="word",outputCol="ngram_result")
ngram.transform(df).show()
```



**(5)RegexTokener**

​	正则表达式分词器，用于将一个字符串根据指定的正则表达式来进行分词。

> 参数包括：
>
> ​	pattern：用于指定分词正则表达式，默认为遇到任何空白字符则分词
>
> ​	minTokenLength:  最小分词长度过滤，小于这个长度则过滤掉

```

```

**(6)VectorIndexer**

​	VectorIndexer是**对数据集特征向量中的类别（离散值）特征进行编号**。它能够自动判断那些特征是离散值型的特征，并对他们进行编号，具体做法是通过**设置一个maxCategories**，特征向量中**某一个特征不重复取值个数小于maxCategories，则被重新编号为0～K（K<=maxCategories-1**）。**某一个特征不重复取值个数大于maxCategories，则该特征视为连续值，不会重新编号**

> 主要作用：提升决策树、随机森林等ML算法的效果
>
> 参数：
>
> ​	1.MaxCategories  是否被判为离散类型的标准
>
> ​	2.inputCol 	输入列名
>
> ​	3.outputCol	输出列名

~~~python
+-------------------------+-------------------------+
|features                 |indexedFeatures          |
+-------------------------+-------------------------+
|(3,[0,1,2],[2.0,5.0,7.0])|(3,[0,1,2],[2.0,1.0,1.0])|
|(3,[0,1,2],[3.0,5.0,9.0])|(3,[0,1,2],[3.0,1.0,2.0])|
|(3,[0,1,2],[4.0,7.0,9.0])|(3,[0,1,2],[4.0,3.0,2.0])|
|(3,[0,1,2],[2.0,4.0,9.0])|(3,[0,1,2],[2.0,0.0,2.0])|
|(3,[0,1,2],[9.0,5.0,7.0])|(3,[0,1,2],[9.0,1.0,1.0])|
|(3,[0,1,2],[2.0,5.0,9.0])|(3,[0,1,2],[2.0,1.0,2.0])|
|(3,[0,1,2],[3.0,4.0,9.0])|(3,[0,1,2],[3.0,0.0,2.0])|
|(3,[0,1,2],[8.0,4.0,9.0])|(3,[0,1,2],[8.0,0.0,2.0])|
|(3,[0,1,2],[3.0,6.0,2.0])|(3,[0,1,2],[3.0,2.0,0.0])|
|(3,[0,1,2],[5.0,9.0,2.0])|(3,[0,1,2],[5.0,4.0,0.0])|
+-------------------------+-------------------------+
结果分析：特征向量包含3个特征，即特征0，特征1，特征2。如Row=1,对应的特征分别是2.0,5.0,7.0.被转换为2.0,1.0,1.0。
我们发现只有特征1，特征2被转换了，特征0没有被转换。这是因为特征0有6中取值（2，3，4，5，8，9），多于前面的设置setMaxCategories(5)
，因此被视为连续值了，不会被转换。
特征1中，（4，5，6，7，9）-->(0,1,2,3,4,5)
特征2中,  (2,7,9)-->(0,1,2)

~~~

**(7)StringIndexer**

​	将label标签进行重新设置，出现的最多的标签被设置为0，最少的设置最大。

~~~python
按label出现的频次，转换成0～num numOfLabels-1(分类个数)，频次最高的转换为0，以此类推：
label=3，出现次数最多，出现了4次，转换（编号）为0
其次是label=2，出现了3次，编号为1，以此类推
+-----+------------+
|label|indexedLabel|
+-----+------------+
|3.0  |0.0         |
|4.0  |3.0         |
|1.0  |2.0         |
|3.0  |0.0         |
|2.0  |1.0         |
|3.0  |0.0         |
|2.0  |1.0         |
|3.0  |0.0         |
|2.0  |1.0         |
|1.0  |2.0         |
+-----+------------+

~~~

**(8)StringToIndex**

​	功能与StringIndexer完全相反，用于使用StringIndexer后的标签进行训练后，再将标签对应会原来的标签

> 作用：恢复StringIndexer之前的标签
>
> 参数：
>
> ​	1.inputCol 	输入列名
>
> ​	2.outputCol	输出列名

~~~
|label|prediction|convetedPrediction|
+-----+----------+------------------+
|3.0  |0.0       |3.0               |
|4.0  |1.0       |2.0               |
|1.0  |2.0       |1.0               |
|3.0  |0.0       |3.0               |
|2.0  |1.0       |2.0               |
|3.0  |0.0       |3.0               |
|2.0  |1.0       |2.0               |
|3.0  |0.0       |3.0               |
|2.0  |1.0       |2.0               |
|1.0  |2.0       |1.0               |

~~~



### 2.评估器

​	评估器就是机器学习模型，通过统计数据从而进行预测工作，每个评估器都必须实现.fit主要分为分类和回归两大类，这里只针对分类评估器进行介绍。

​	



​	Pyspark的分类评估器包含以下七种：

**1.LogisticRegression**

​	逻辑回归模型

**2.DecisionTreeClassifier**

​	决策树模型

**3.GBTClassifier**

​	梯度提升决策树

**4.RandomForestClassifier**

​	随机森林

**5.MultilayerPerceptronClassifier**

​	多层感知机分类器

