---
title: linux常用命令
date: 2018-09-23 10:44:28
tags: [Linux,服务器管理]
categories: Linux操作
---
文件管理
----------

**1.查看当前文件夹中文件**  

    #显示文件夹中全部文件和目录数
    ls | wc -l
    #显示包含指定内容的文件和目录
    ls *3094 | wc -l
    
    #显示文件夹中文件的个数
    find . -type f | wc -l

注：wc 是统计前面管道输出的东西，-l表示按照行统计


磁盘管理
---------

**1.查看文件夹中文件的总大小**

    #查看当前文件夹中文件总大小
    du -h    # -h表示以人可理解的方式输出
    
    #查看指定文件夹中文件的总大小
    du /home/yhk/ -h   


​    
**2.查看磁盘各个分区大小及使用情况**
​    
    df -h


内存、Cpu使用情况查看
-----------

**1.Cpu个数** 

逻辑Cpu个数查看：  

    1.方式一：
    　　　先使用top密令进入top界面，在界面中按1，即可出现cpu个数以及使用情况  
    
    2.方式二：
    　　　cat /proc/cpuinfo |grep "processor"|wc -l 


​    　　　
物理CPU个数查看：  


    cat /proc/cpuinfo |grep "physical id"|sort |uniq|wc -l 

一个物理CPU数几核：
​    
    cat /proc/cpuinfo |grep "cores"|uniq

**2.CPU内存使用情况**  

    top


实用
------------------

**程序以忽略挂起信号方式执行**
​    
    nohup command > myout.file 2>&1 & #文件产生的输出将重定向到myout.file


​            
​            
​    
​    


​        
​        
​       
