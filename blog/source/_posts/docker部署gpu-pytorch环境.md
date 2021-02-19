---
title: docker部署gpu-pytorch环境
copyright: true
mathjax: true
date: 2020-07-28 18:04:04
tags:
categories:
---

概述：最近在公司的服务器上里进行环境部署，需要使用GPU进行深度学习，发现之前使用docker部署的环境直接安装nvidia驱动会不停的产生错误，折腾了一整天，终于成功的在docker镜像中成功部署显卡驱动，使用pytorch成功调用显卡进行深度学习训练，本篇博客对整个docker-torch-gpu部署过程进行记录。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.z4c2x3es1u.png)

<!--more-->

### docker中调用GPU

#### 关键点1——使用已经部署好GPU的image

&emsp;&emsp;这里首先给想要在docker中使用GPU的朋友一个忠告，`尽量不要使用已经部署好其他环境的docker来安装GPU驱动，而是直接去找到包含了GPU驱动和cuda的image来安装其他需要的包`。

&emsp;&emsp;首先在docker hub上找到pytorch官方发布的images项目，点击进入。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.53szuedr44r.png)

&emsp;&emsp;然后点击Tags按钮在其中找到对应的版本。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.3dgqnu9tord.png)

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.k76pnql45a9.png)

&emsp;&emsp;按照对应版本的image后面显示的方式进行下拉镜像。

~~~shell
docker pull pytorch/pytorch:1.4-cuda10.1-cudnn7-devel
~~~

&emsp;&emsp;对于需要的cuda版本不清楚的，可以再docker外面使用`nvidia-smi`查看宿主机所使用的cuda版本进行确定。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.dolas5ip08q.png)

> 注意：上面的截图是另一台机器的截图，按照上面的截图前面pull的镜像也cuda版本应该为10.2，而不是10.1



#### 关键点2——使用runtime属性指定nvidia

&emsp;&emsp;使用docker进行GPU利用的第二个关键点就是使用部署好GPU环境的image生成container时，要使用附加参数--runtime指定使用nvidia驱动，创建方式如下：

~~~powershell
sudo docker run --runtime=nvidia \    # 
								--it \       # 指定交互式启动 
								-p 12345:8888 \  	# 指定端口映射，将容器内8888端口映射成外面可访问的12345端口
								-v /home/docker_share:/home/ \ 	# 设置目录映射，将container内的/home/映射到/home/docker_share
								image_id bash
~~~

&emsp;&emsp;使用该命令成功创建并进入docker后，采用nvida-smi命令查看是否GPU可用。出现下面界面证明GPU可用。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.x6u60dt9stm.png)



##### 参考文献

- https://zhuanlan.zhihu.com/p/109477627
- https://bluesmilery.github.io/blogs/252e6902/