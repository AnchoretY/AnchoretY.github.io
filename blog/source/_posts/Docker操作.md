---
title: Docker操作
date: 2018-09-23 23:37:06
tags: Linux
Categories: Linux相关
---



> -d 容器在后台运行
>
> -P 将容器内部的端口映射到我们的主机上



```linux
docker ps    #查看全部正在开启的docker
docker ps 
```







####1.进入以及退出docker

​	进入docker命令主要分为两种，attach和exec命令，但是由于exec命令退出后容器继续运行，因此更为常用。

~~~python
#首先查看正在运行的docker，在其中选择想要进入的docker name
docker ps

#然后使用exec进行进入docker
docker exec --it docker_name /bin/bash/

#进行各种操作

#退出docker
exit或Ctrl+D
~~~



### 2.docker和宿主主机之间传输文件

​	docker 使用docker cp 命令来进行复制，**无论容器有没有进行运行，复制操作都可以进行执行**。

~~~python
#从docker中赋值文件到宿主主机
docker cp docker_name:/docker_file_path local_path

#从宿主主机复制到docker
docker cp local_path docker_name:/docker_file_path
~~~

