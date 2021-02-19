---
title: pyspark dataframe基础操作
date: 2018-09-28 10:40:25
tags: spark
categories: spark
---

**1.select**

​	select用于列选择，选择指定列，也可以用于和udf函数结合新增列

​	**列选择**：

```python
data_set.select("*").show()		#选择全部列
data_set.select("file_name","webshell").show()		#选择file_name，webshell列
```

​	**与udf函数组合新增列**：

```python
from pysaprk.sql.function import udf
def sum(col1,col2):
	return col1+col2
udf_sun = udf(sum,IntergerType())

data_set.select("*",udf_sum("a1","a2")).show()	#新增一列a1和a2列加和后的列
```



**2.filter**

​	filter用于行选择，相当于使用前一半写好的sql语句进行查询。

​	查询某个列等于**固定值**：

```sql
data_set.filter("file_name=='wp-links-opml.php'").show()
```

​	查询符合某个**正则表达式**所有行:​	

```sql
data_set.filter("file_name regexp '\.php$')		#选择所有.php结尾文件
```



**3.join**

​	pyspark的dataframe横向连接只能使用join进行连接，需要有一列为两个dataframe的共有列

```python
df_join = df1.join(df,how="left",on='id').show() 	#id为二者的共有列
```



**4.agg**

​	使用agg来进行不分组聚合操作，获得某些统计项

```python
#获取数据中样本列最大的数值
#方式一：
df.agg({"id":"max"}).show()		

#方式二：
import pyspark.sql.functions as fn	
df.agg(F.min("id")).show()
```



**5.groupby**

​	使用groupby可以用来进行分组聚合。

```
df.groupby(")
```



**6.printSchema**

​	以数的形式显示dataframe的概要

```
df.printSchema()

output:
	root
     |-- id: integer (nullable = true)
     |-- age: integer (nullable = true)
```



**7.subtract**

​	找到那些在这个dataframe但是不在另一个dataframe中的行，返回一个dataframe

```python
df = spark.createDataFrame([[2.0,'a'],[1.0,'b'],[4.0,'b'],[9.0,'b'],[4.3,'c']],schema=schema)
df1 = spark.createDataFrame([[2.0,'a'],[4.3,'c']],schema=schema)

df.subtract(df1).show()		#找到在df中但是不在df1中的行
```



**8.cast**

​	指定某一列的类型

```

```



**9.sort**

​	按照某列来进行排序

>参数：
>
>​	columns: 可以是一列，也可以是几列的列表
>
>​	ascending：升序，默认是True

~~~python
data.sort(['count'],ascending=False).show()
~~~

