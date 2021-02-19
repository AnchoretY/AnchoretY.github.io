---
title: sql注入——通过sqlmap进行getshell常见的问题
date: 2020-01-20 10:22:24
tags: [sql注入,Web安全]
---

​	在上一篇文章中，我们提到了要使用sqlmap中自带的os-shell命令直接getshell要有下面四个先条件：

> 1.当前注入点为root权限
>
> 2.已知网站绝对路径
>
> 3.php转义功能关闭
>
> 4.secure_file_priv= 值为空

在本文中将针对在实际环境中使用sqlmap进行getshell如何获取这些先决条件来进行详细介绍。

### 1.确认注入点权限

​	首先要确认注入点权限是否为root权限，可以直接使用sqlmap自带的测试命令is-dba

~~~shell
sqlmap -u 网址 --is-dba
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/产看当前注入点是否为root权限结果.png?raw=true)

###  2.网站的绝对路径

​	获取网站的绝对路径在可以先进入sql-shell:

~~~shell
sqlmap -u 网址 --sql-shell
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/进入sql-shell.png?raw=true)

​	然后再在sql-shell中直接使用sql命令读取数据库文件存放路径：

~~~mysql
sql-shell> select @@datadir;
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/sql-shell获取绝对路径结果.png?raw=true)

然后通过数据库文件的位置进行网站所在的绝对路径进行猜测。

