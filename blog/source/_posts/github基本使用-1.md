---
title: github基本使用
date: 2019-11-13 23:21:19
tags:
---

​	之前对github一直就是简单的使用，最近找完工作终于优势间静下来好好地研究下github和git，这个系列博客就用来记录在github学习过程中的新get到的一些点。

### 1.Git邮箱姓名设置

​	在我们最开始进行本地git设置时，一般都要使用

~~~shell
git config --global user.name "xxx"
git config --gobal user.email "xxx"
~~~

进行姓名和邮箱设置，对这个已知都是使用自己的常用邮箱和真实名，对于安全专业的硕士真是是很蠢的行为。

> **这里设置的姓名和邮箱都是会在github上公开仓库时随着日志一起进行公开的！因此不要使用隐私的信息**

​	要进行更改可以直接修改~/.gitconfig中的内容进行重新设置。

~~~shell
[user]
	name = “abc”
	email = xxx@qq.com
~~~



### 2.github中的watch、star、fork

​	用好github正确的使用好watch、star、fork是非常重要的一步，这关系到你能不能正确的进行喜欢项目的跟踪。下面是对这三张常见的操作进行的介绍：

#### watch

​	watch即观察该项目，对一个项目选择观察后只要有任何人在该项目下面提交了issue或者issue下面有了任何留言，通知中心就会进行通知，如果设置了个人邮箱，邮箱同时也会受到通知。

> [如何正确的接收watching 通知消息](https://github.com/cssmagic/blog/issues/49)推荐看这一篇文章

#### Star

​	Star意思是对项目打星标（也就是点赞）,一个项目的点赞数目的多少很大程度上是衡量一个项目质量的显而易见的指标。

>  Star后的项目会专门加入一个列表，在个人管理中可以回看自己Star的项目。

#### fork

​	使用fork相当于你对该项目拥有了一份自己的拷贝，拷贝是基于当时的项目文件，后续项目发生变化需要通过其他方式去同步。

> 使用很少，除非是想在一个项目的基础上想建设自己的项目才会用到

#### 使用建议

> 1.对于一些不定期更新新功能的好项目使用watch进行关注
>
> 2.认为一个项目做得不错，使用star进行点赞
>
> 3.在一个项目的基础上想建设自己的项目，使用fork



### 3.Git版本回退

#### 已经进行add，但还没有进行commit

~~~
git status 先看一下add 中的文件 
git reset HEAD 如果后面什么都不跟的话 就是上一次add 里面的全部撤销了 
git reset HEAD XXX/XXX/XXX.java 就是对某个文件进行撤销了
~~~

#### 本地已经进行了commit，但是还没有更新到远程分支

~~~python
# 先找到要进行会退的版本id
git log    

# 进行本地仓库回退
git reset --hard 提交的编号
~~~

#### 远程分支已进行进行同步

​	其实就是先进性本地分支回退，然后将本都分支强制push到远程。

~~~python
# 先找到要进行会退的版本id
git log    

# 进行本地仓库回退
git reset --hard 提交的编号

# 强制将本地分支push到远程
git push -f
~~~









​	