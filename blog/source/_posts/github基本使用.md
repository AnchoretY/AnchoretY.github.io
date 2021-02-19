---
title: github基本使用
date: 2018-10-03 11:33:02
tags: Linux
categories: Linux
---

---------------------
**github相关知识**：

​	github项目使用终端进行管理时分为三个区域：**工作目录、暂存区（本地库）、github上的远程项目**，当我们要更新以及操作一个项目时，要遵循以下的格式：

>1.先从github上面pull下远程的项目分支
>
>2.本地的项目文件夹中的文件进行更新（更新工作目录中的文件）
>
>3.使用add将更新的索引添加到本地库
>
>4.使用commit工作目录中的文件提交到暂存区(本地库)
>
>5.将文件push到远程分支或merge到远程分支

**基本操作**

> git clone "ssh项目地址"	克隆远程项目
>
> git pull origin master 		取回远程主机或本地的某个分支的更新，再与本地分支进行合并(这种写法是origin主机的master分支与本地当前分支进行合并)
>
> git push origin master 	将本地的当前分支push到origin主机的master分支上
>
> git add "文件名"			将指定文件提交到本地库
>
> git commit -m "描述信息"    将本地的全部文件都提交到本地库
>
> git log 					打印该项目的版本操作信息
>
> git status 				查看到那个钱仓库状态






**更新github**

```python
#将项目从github上面拉下来(本地已经有的可以跳过,已有则直接进入该文件夹)
git clone github链接

#查看项目状态
git status

output:
    On branch master
    Your branch is up to date with 'origin/master'.

    nothing to commit, working tree clean
 
#创建或者导入新文件到工作区
touch "文件1"

#将文件工作目录的文件提交到暂存区
git add "文件1" 	#提交指定文件
git add -A		#一次提交工作目录中的全部文件

#查看项目状态
 git status
    

#第一次提交描述时需要设置账户信息
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com

#添加描述
git commit -m "此次添加的描述信息"

#查看项目状态
git status 
output:
    On branch master
    Your branch is ahead of 'origin/master' by 1 commit.
      (use "git push" to publish your local commits)

#将修改从暂存区提交到远程分支
git push origin master

```



**删除已经提交到github上面的文件**

~~~python
#在本地拉取远程分支
git pull original master

#在本地删除对应的文件
git rm filename

#添加描述上传到远程分支
git commit -m "删除文件filename"
git push original master
~~~





**已经提交到github上的文件进行版本回退**

```python
#先通过git log获取版本号
git log

#然后使用git revert 版本号来回退到指定版本
git retvert 版本号

#然后:x保存退出就可以了撤回到指定的版本了

#最后再将本地分支push到github上
git push origin master

```



**分支切换**

```python

git checkout ...    #创建分支

git checkout -b ... 	#创建并且换到分支

git checkout ... 	#切换到分支

```

```python
git branch #查看本地分支

git branch -a  #查看本地和远程的全部分支

git push origin --delete dev2	#删除远程分支
```

