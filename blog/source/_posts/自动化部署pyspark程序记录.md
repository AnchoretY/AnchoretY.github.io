---
title: 自动化部署pyspark程序记录
date: 2018-12-07 11:18:20
tags:
---

​	项目要求需要在pyspark的集群中将一部分程序做集群的自动化部署，本文记录了程序部署过程中，使用到的一些技术以及遇到的一些问题。

####1.SparkSession创建时设置不生效

​	首先，要进行程序的自动化部署首先要将程序封装成python文件，在这个过程中**可能会出现sparkSession谁知不能生效的问题，不论对SparkSession进行什么设置，都不会生生效**

> 这种问题是由于SparkSession的创建过程不能写在主程序中，必须要写在所有函数的外层，并且进行的在文件的初始部分穿创建



#### 2.python 文件传入获取参数

​	python文件也可以和shell脚本一样进行运行时传入参数，这里主要使用的的是python自带的sys和getopt包

~~~python
要接受参数的python文件：

import sys
import getopt

opts,args = getopt.getopt(sys.argv[1:],"d:",["d:"])

for opt,arg in opts:
    if opt in ("-d","--d"):
        input_file = arg
#后续可以直接使用input——file获取的变量名进行操作


~~~

####3.将python文件执行封装到shell脚本中

​	这里之所以将python文件进行封装主要是为了方便移植，其实也可以直接设置将python脚本文件执行设置成定时任务，这里是一波瞎操作。主要为了练习和方便移植

~~~shell
#首先在这个shell重要实现获取当前日期或前n天的日期
date = `date -d "1 days ago"+%Y-%m-%d`

#然后在将date作为参数后台执行这个程序并且生成日志
python ***.py -d date > /path/${date}.log 2>&1 &

#=====================注意==============================
#上面直接使用python执行时可能会出现系统中存在多个python导致部署时使用的python和之前测试使用的python不是一个python环境导致的，那么如何确定测试时使用的python环境呢？
#要解决上述问题可以先从新进入到测试用的python环境，然后进行下面操作
import sys
print(sys.execyutable)
#然后将python目录改为上面的python目录
~~~

