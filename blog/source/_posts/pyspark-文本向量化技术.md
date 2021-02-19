---
title: pyspark-向量化技术
date: 2018-11-20 20:09:24
Tags: pyspark
catefories: pyspark
---

​	在pyspark中文本的向量化技术主要在包pyspark.ml.feature中，主要包括以下几种：

> 1.Ngram
>
> 2.tf-idf
>
> 3.Word2Vec



#### 1.Ngram

​	

### 2.tf-idf

​	在pyspark中tf和idf是分开的两个步骤

​	（1）tf

​	整个tf的过程就是一个将将各个文本进行计算频数统计的过程，使用前要先使用也定的算法来对语句进行分词，然后指定统计特征的数量再进行tf统计

​	

> 常用参数：
>
> ​	1.numsFeatures   统计的特征数量，这个值一般通过`ParamGridBuilder`尝试得出最合适的值
>
> ​	2.inputCol   输入列，输入类列为ArrayType的数据
>
> ​	3.outputCol 输出列 ,输出列为Vector类型的数据

~~~python
df = spark.createDataFrame([(["this", "is", "apple"],),(["this", "is", "apple","watch","or","apple"],)], ["words"])

hashingTF = HashingTF(numFeatures=10, inputCol="words", outputCol="tf")
hashingTF.transform(df).show(10,False)

output:
+-----------------------------------+--------------------------------+
|words                              |tf                              |
+-----------------------------------+--------------------------------+
|[this, is, apple]                  |(10,[1,3],[2.0,1.0])            |
|[this, is, apple, watch, or, apple]|(10,[1,2,3,7],[3.0,1.0,1.0,1.0])|
+-----------------------------------+--------------------------------+

~~~

​	其中，10代表了特征数，[1,3]代表了this和is对应的哈希值，[2.0,1.0]代表了this和is出现的频数.

​	(2)idf

> 常用参数：
>
> ​	1.minDocFreq 最少要出现的频数，如果超过minDocFreq个样本中出现了这个关键词，这个频数将不tf-idf特征，直接为0
>
> ​	2.inputCol 	输入列
>
> ​	3.ouputCol	输出列

~~~python
idf = IDF(inputCol="tf",outputCol="tf-idf")
idf_model = idf.fit(df)

idf_model.transform(df).show(10,False)

output:
+-----------------------------------+--------------------------------+--------------------------------------------------------------+
|words                              |tf                              |tf-idf                                                        |
+-----------------------------------+--------------------------------+--------------------------------------------------------------+
|[this, is, apple]                  |(10,[1,3],[2.0,1.0])            |(10,[1,3],[0.0,0.0])                                          |
|[this, is, apple, watch, or, apple]|(10,[1,2,3,7],[3.0,1.0,1.0,1.0])|(10,[1,2,3,7],[0.0,0.4054651081081644,0.0,0.4054651081081644])|
+-----------------------------------+--------------------------------+--------------------------------------------------------------+
~~~





### 3.CountVec

​	CountVec是一种直接进行文本向量，直接词频统计的向量化方式，可以

> 常用参数包括：
>
> ​	minDF：要保证出现词的代表性。当minDF值大于1时，表示词汇表中出现的词最少要在minDf个文档中出现过，否则去除掉不进入词汇表；当minDF小于1，表示词汇表中出现的词最少要在包分之minDF*100个文档中出现才进入词汇表
>
> ​	minTF：过滤文档中出现的过于罕见的词，因为这类词机乎不在什么文本中出现因此作为特征可区分的样本数量比较少。当minTF大于1时，表示这个词出现的频率必须高于这个才会进入词汇表；小于1时，表示这个大于一个分数时才进入词汇表
>
> ​	binary:  是否只计算0/1,即是否出现该词。默认值为False。
>
> ​	inputCol:输入列名，默认为None
>
> ​	outputCol:输出列名，默认为None

~~~python
df = spark.createDataFrame([(["this", "is", "apple"],),(["this", "is", "apple","watch","or","apple"],)], ["words"])

#使用Word2Vec进行词向量化
countvec = CountVectorizer(inputCol='words',outputCol='countvec')
countvec_model = countvec.fit(df)
countvec_model.transform(df).show(10,False)

output:
+-----------------------------------+----------------------------------------+-------------------------------------+
|words                              |tf                                      |countvec                             |
+-----------------------------------+----------------------------------------+-------------------------------------+
|[this, is, apple]                  |(20,[1,11,13],[1.0,1.0,1.0])            |(5,[0,1,2],[1.0,1.0,1.0])            |
|[this, is, apple, watch, or, apple]|(20,[1,2,7,11,13],[1.0,1.0,1.0,2.0,1.0])|(5,[0,1,2,3,4],[2.0,1.0,1.0,1.0,1.0])|
+-----------------------------------+----------------------------------------+-------------------------------------+

#使用CountVec的binary模式进行向量化，
countvec = CountVectorizer(inputCol='words',outputCol='countvec',binary=True)
countvec_model = countvec.fit(df)
countvec_model.transform(df).show(10,False)
output:
+-----------------------------------+----------------------------------------+-------------------------------------+
|words                              |tf                                      |countvec                             |
+-----------------------------------+----------------------------------------+-------------------------------------+
|[this, is, apple]                  |(20,[1,11,13],[1.0,1.0,1.0])            |(5,[0,1,2],[1.0,1.0,1.0])            |
|[this, is, apple, watch, or, apple]|(20,[1,2,7,11,13],[1.0,1.0,1.0,2.0,1.0])|(5,[0,1,2,3,4],[1.0,1.0,1.0,1.0,1.0])|
+-----------------------------------+----------------------------------------+-------------------------------------+
~~~





###4.Word2Vec

​	Word2Vec 是一种常见的文本向量化方式,使用神经网络讲一个词语和他前后的词语来进行表示这个这个词语，主要分为CBOW和Skip-

​	特点：Word2Vec主要是结合了前后词生成各个词向量，具有一定的语义信息

**在pyspark.ml.feature中存在Word2Vec和Word2VecModel两个对象，这两个对象之间存在什么区别和联系呢？**

​	Word2Vec是Word2Vec基本参数设置部分，Word2VecModel是训练好以后的Word2Vec，有些函数只有Word2VecModel训练好以后才能使用

> 常见参数：
>
> ​	1.vectorSize    生成的词向量大小
>
> ​	2.inputCol 	输入列
>
> ​	3.ouputCol	输出列
>
> ​	4.windowSize   输出的词向量和该词前后多少个词与有关
>
> ​	5.maxSentenceLength  输入句子的最大长度，超过改长度直接进行进行截断
>
> ​	6.numPartitions 分区数，影响训练速度
>
>
>
> 常用函数：
>
> ​	这里的常见函数要对Word2VecModel才能使用
>
> ​	getVectors() 		获得词和词向量的对应关系,返回值为dataframe
>
> ​	transform()		传入一个dataframe，将一个词列转换为词向量
>
> ​	save()			保存模型



使用要先使用训练集对其进行训练：

~~~python
输入数据：
	已经使用一定的分词方式已经进行分词后的ArrayType数组
输出：
	当前句子各个词进行word2vec编码后的均值，维度为vectorSize
 
word2vec = Word2Vec(vectorSize=100,inputCol="word",outputCol="word_vector",windowSize=3,numPartitions=300)
word2vec_model = word2vec.fit(data)

#features将会在data的基础上多出一列word_vector，为vectorSize维数组
features = word2vec.trandform(data)

word2vec_model.save("./model/name.word2vec")
~~~

**Word2Vec如何查看是否已经训练的很好：**

> ​	1.选择两个在日常生活中已知词义相近的两个词A、B，再选一个与A词义不那么相近但也有一定相似度的词C
>
> ​    2.计算A和B以及A和C的余弦距离
>
> ​	3.比较其大小，当满足AB距离小于AC时，重新选择三个词重复上过程多次都满足，那么认为模型已经训练完毕；若不满足上述过程，那么继续加入样本进行训练

**当word2vec中为了表达两个比较相近的词的相似性可以怎么做？比如在当前word2vec下tea、cooffe之间的相似性非常高，接近于1**

> ​	增加word2vec的向量维度。可能是在当前维度中向量维度过小，导致这两个词无法表达充分，因此我们可以增加向量维度，以期待在更高维的向量空间中，可以区分这个名词





**过程中可能用的：**

~~~python
#获得某个词对应的词向量
word2vec_model.getVectors().filter("word=='0eva'").collect()[0]['vector']

#计算两个词向量之间距离平方
a1.squared_distance(a2)
~~~

