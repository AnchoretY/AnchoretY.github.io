---
title: 设置pycharm使用远程服务器中的docker环境进行开发
copyright: true
mathjax: true
date: 2020-11-19 18:20:40
tags:
categories:
---

概述：最近感觉使用jupyter notebook开放项目存在整体性不强，因此决定使用再构建一套pytorch的开发环境来，具体的目标为：在本地进行编码，自动同步到远程服务器中的docker环境内执行，本文记录这套环境的完整构建过程。

![](![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.4eh9hywn30q.png))

<!--more-->

### docker配置（远程）

**整体思路**：在docker中安装ssh，然后将docker的ssh端口映射到宿主机上，远程客户端直接通过ssh访问docker内的环境。



#### 0. 保存已有docker

&emsp;&emsp;在我的应用场景下，已经有了一个使用了很久的pytorch_cuda 的docker环境，这次我希望使用这个docker中的环境作为pychram中要进行调用的环境，因此需要首先将已有的docker环境保存，然后在重新创建容器的时候增加映射即可。保存已有容器中的内容代码如下：

~~~shell
docker commit ${containerName} ${imageTag}
~~~

#### 1. 设置docker对外映射

&emsp;&emsp;在我的实际环境中由于已经有了一个已经运行了很久的docker，要在这个docker上增加新的端口映射，因此比较复杂，如果使用新的镜像可以直接跳过保存保存已有镜像这些步骤，直接在docker创建时设置端口映射。

~~~shell
sudo docker run \
	--name=pytorch_cuda \指定
	--runtime=nvidia \ # 指定运行时使用的nvidia显卡
	-p 12345:8888 \  # jupyter notebook的8888端口映射到宿主机的12345端口
	-p 54321:22   \  # ssh的22端口映射到54321端口
	-v /home/docker_share:/home/  \ # 将docker中的/home/目录映射到/home/docker_share
	pytorch_cuda:latest
~~~

#### 2. 安装、开启docker的ssh服务

&emsp;&emsp;大部分docker的镜像中并没有安装ssh服务，因此一般需要自己安装，在docker中输入下面命令进行ssh安装：

~~~shell
apt update
apt-install openssh-server
~~~

&emsp;&emsp;更改root用户的密码，为了后续登录：

~~~shell
passwd root
~~~

&emsp;&emsp;开启docker服务，在docker中输入下面命令：

~~~shell
service ssh start
~~~

&emsp;&emsp;更改配置文件：

~~~shell
vim /etc/ssh/sshd_config
	> PermitRootLogin的值从prohibit-password改为yes  
	> X11UseLocalhost设置为no 
~~~

&emsp;&emsp;最后重启ssh服务和docker镜像。

~~~shell
service ssh restart  # 重启ssh服务，使ssh配置文件更改生效（docker内执行）
docker restart DOCKER_NAME  # 重启docker，时root密码生效（宿主机执行）
service ssh start  # 重新开启ssh服务（docker内执行）
~~~

&emsp;&emsp;到这里就已经完成了远程docker内的部署，在宿主机使用新更改的root密码尝试ssh登录docker，能正常登录则设置成功。



###  pycharm配置(本地)

#### 1. pycharm与远程docker建立连接

&emsp;&emsp;打开PyCharmTools > Deployment > Configuration, 新建一个SFTP服务器，设置远程登录docker ssh的配置、根目录（想要使用作为根目录的任意docker中的目录）等，

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.e6cpzuem1ak.png)



![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.8vd46qpj8va.png)

&emsp;&emsp;最后在Mappings中配置路径，这里的路径是你本地存放代码的路径，与刚刚配置的Root Path相互映射（意思是Mapping里本机的路径映射到远程的Root Path），方便以后在本地和远程docker中进行代码和其他文件同步。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.r8umpmo2ew.png)

&emsp;&emsp;测试连接，能够成功连接那么这一步就完成了。

#### 2. 配置远程解释器

&emsp;&emsp;点击PyCharm的File > Setting > Project > Project Interpreter右边的设置按钮新建一个项目的远程解释器：

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.jkfv65ayx9.png)

&emsp;&emsp;点击Add按钮新增远程解释器，然后选择在上面一步中已经配置好的连接。x

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.rtx9qdje0n.png)

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.x120gh75e7s.png)

​	&emsp;&emsp;配置完成后等待解释器同步，同步完成后远程解释器可以显示全部的版本。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.9jy9p6n0lc9.png)

&emsp;&emsp;最后就是等待文件本地文件同步到远程服务器了。完成后即可直接在本地编辑文件，保存文件则自动同步到服务器上，执行则为在远程环境中执行。





##### 参考文献

- [运行中的Docker容器增加端口映射](https://blog.opensvc.net/yun-xing-zhong-de-dockerrong-qi/)
- [如何远程登录docker容器](https://blog.csdn.net/thmx43/article/details/106759774)