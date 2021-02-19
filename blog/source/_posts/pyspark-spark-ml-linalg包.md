---
title: pyspark-spark.ml.linalg包
date: 2018-11-21 10:57:11
tags: pyspark
---



pyspark 的**pyspark.ml.linalg**包主要提供了向量相关(矩阵部分不是很常用因此本文不提)的定义以及计算操作

> 主体包括：
>
> ​	1.Vector 
>
> ​	2.DenseVector
>
> ​	3.SparseVector



### 1.Vector

​	是下面所有向量类型的父类。我们使用numpy数组进行存储，计算将交给底层的numpy数组。

> 主要方法：
>
> ​	toArray()	将向量转化为numpy的array



###2.DenseVector

​	创建时，可以使用list、numpy array、等多种方式进行创建

>常用方法：
>
>​	dot() 	计算点乘,支持密集向量和numpy array、list、SparseVector、SciPy Sparse相乘
>
>​	norm()	计算范数
>
>​	numNonzeros()	计算非零元素个数
>
>​	squared_distance()	计算两个元素的平方距离
>
>​	.toArray()	转换为numpy array
>
>​	values	返回一个list

~~~python
#密集矩阵的创建
v = Vectors.dense([1.0, 2.0])
u = DenseVector([3.0, 4.0])

#密集矩阵计算
v + u

output:
	DenseVector([4.0, 6.0])
 
#点乘
v.dot(v)					#密集向量和密集向量之间进行点乘
output:
    5.0
v.dot(numpy.array([1,2]))   #使用密集向量直接和numpy array进行计算
output:
    5.0
   
#计算非零元素个数
DenseVector([1,2,0]).numNonzeros()

#计算两个元素之间的平方距离
a = DenseVector([0,0])
b = DenseVector([3,4])
a.squared_distance(b)
output:
    25.0
    
#密集矩阵转numpy array
v = v.toArray()
v

output:
    array([1., 2.])
 
~~~



### 3.SparseVector

​	简单的系数向量类，用于将数据输送给ml模型。

​	Sparkvector和一般的scipy稀疏向量不太一样，其表示方式为，（数据总维数，该数据第n维存在值列表，各个位置对应的值列表）

> 常用方法：
>
> ​	dot()	SparseVector的点乘不仅可以在SparseVector之间还可以与numpy array相乘
>
> ​	**indices    有值的条目对应的索引列表，返回值为numpy array**
>
> ​	**size	向量维度**
>
> ​	norm()	计算范数
>
> ​	numNonzeros()	计算非零元素个数
>
> ​	squared_distance()	计算两个元素的平方距离
>
> ​	.toArray()	转换为numpy array
>
> ​	values	返回一个list

注：加粗部分为SparseVector特有的

~~~python
#创建稀疏向量
a = SparseVector(4, [1, 3], [3.0, 4.0])
a.toArray()
output:
    array([0., 3., 0., 4.])

#计算点乘
a.dot(array([1., 2., 3., 4.]))
output:
    22.0
    
#获得存值得对应的索引列表
a.indices
output:
   	array([1, 3], dtype=int32)

#获取向量维度
a.size
output:
   	4  
~~~



