---
title: 多电脑hexo博客管理
copyright: true
mathjax: true
date: 2021-03-14 17:30:13
tags:
categories: [博客]
---

概述：最近新添了一台台式机，准备与mbp同时管理博客，这里记录一下设置过程。

![]()

<!--more-->



### 部署过程

#### 源主机

##### 1. 创建新分支

&emsp;&emsp;在正常写博客并进行部署时，只会将博客相关的静态文件同步到github的master分支，master分支上的静态文件如下图所示，以供网站展示使用，而不会将生成这些文件的源文件传输到github上。而我们要在两台电脑上共同进行博客管理，就需要创建而外分支，将源文件传输到新的分支进行同步管理。

&emsp;&emsp;在github上**创建新的hexo分支**

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.630of8n3imr.png)

&emsp;&emsp;将仓库的hexo分支设置为默认分支

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.4g824knxv7p.png)

##### 2.源文件上传到hexo分支

&emsp;&emsp;首先将新建的hexo分支克隆克隆到本地,然后进入该目录

~~~shell
git clone 地址
cd username.github.io
~~~

&emsp;&emsp;确认当前分支为hexo

~~~shell
git branch
output:
	*hexo
~~~

&emsp;&emsp;上传源文件，将本地博客部署的源文件全部拷贝进`username.github.io`文件目录中（源文件文件夹为质保函下图中结构的文件夹，我的博客中为blog文件夹）

~~~shell
cp /blog/blog username.github.io/
git push origin hexo
~~~

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.gqqrwyqh8i7.png)



#### 新主机

&emsp;&emsp;将旧电脑中`username.github.io`全部文件复制进新主机，然后执行下面操作：

##### 1. 安装npm和node

##### 2. 安装hexo

##### 3. 安装git

##### 3. 在新机器生成ssh key添加到github中

&emsp;&emsp;在新的机器生成ssh秘钥

~~~
ssh-keygen -t rsa -C “your email”		
~~~

&emsp;&emsp;连按三个回车，最终得到了生成的ssh公钥和私钥

~~~
id_rsa				私钥
id_rsa.pub    公钥
~~~

&emsp;&emsp;在github账户设置中导入ssh key

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.jsjxl8xgnor.png)





### 多设备共同编写博客

&emsp;&emsp;单设备编写博客时，博客编写完成只需要使用`hexo d -g`即可完成部署，而使用多设备进行共同编辑博客除了需要对博客网站需要的静态文件进行部署以外，还需要将博客相关的源文件同步到hexo分支。

~~~sh
# 从远程hexo分支部署拉取最新的源文件
git pull origin hexo

# 书写博客
...

# 部署博客（静态文件）
hexo d -g

# 将源文件同步到hexo分支
git push origin hexo
~~~



### 遇到的问题

##### 1. hexo安装完成后，使用`hexo d -g`部署博客出现`The "mode" argument must be integer. Received an instance of Object`

&emsp;&emsp;这个问题主要是由于之前的电脑使用的npm版本与现在的电脑安装的版本不同，而这语法不一致导致的问题，这里使用n进行node版本管理解决该问题。

&emsp;&emsp;安装n：

~~~
npm install -g n
~~~

&emsp;&emsp;然后下载与原电脑版本相同的node版本（我这里是12.14.0，下载后会进行自动切换）

~~~
n 12.14.0
~~~

&emsp;&emsp;使用单独的`n`查看已有node版本和当前选择的node版本。











