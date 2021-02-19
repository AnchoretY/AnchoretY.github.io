---
title: hadoop使用
date: 2018-12-25 11:28:18
tags:
---

1.查看hadoop某个目录下的文件

~~~
sudo hadoop  fs -ls path
~~~

2.从hdfs上下拉文件到本地

~~~
sudo hdfs fs -get file
~~~



3.获取部署在docker中的hadoop的挂载信息等元数据

~~~
sudo docker inspect hdp-server
~~~



