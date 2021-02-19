---
title: python2和python3的不同点
date: 2019-01-07 17:24:07
tags: python
categories: python
---



​	因为系统移植过程中一直出现python3程序向python2转化的问题，因此这里记录下我在程序移植过程中遇到过的坑。



1.python2和python3的url编码解码函数接口





2.python2和python3向文件中写入中文时指定编码方式



​	对于python3来说，要在写入文件时指定编码方式是十分简单的，只需要下面的方式即可：

~~~python
with open(filename,'a',encoding='utf-8') as f:
	f.write("中文")
~~~

​	但对于python2，要在写入文件时,手动添加utf-8文件的前缀

~~~python
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
with open(r'd:\sss.txt','w') as f:
　　f.write(unicode("\xEF\xBB\xBF", "utf-8"))#函数将\xEF\xBB\xBF写到文件开头，指示文件为UTF-8编码。
　　f.write(u'中文')
~~~



3.python2和python3外置函数区别

在python3中外置函数文件可以直接进行调用，如在下面的文件结构

~~~
|--main.py
	tools——

~~~





在python2中外置函数文件目录下必须要有__init__.py 空文件，否则无法进行加载

~~~

~~~



4.python2和python3文件中中文问题

​	在python3中，输出和备注等一切位置都可以直接使用中文，不需要任何额外的代码，在python2中，必须要在包含中文的python文件中加入

~~~python
#coding=utf-8
~~~

​	才能出现中文，否则报错。