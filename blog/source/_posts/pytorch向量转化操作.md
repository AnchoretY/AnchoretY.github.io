---
title: pytorch向量转化操作
date: 2018-09-23 17:49:58
tags: pytorch
categories: pytorch
---





**1.cat**

​	对数据沿着某一维度进行拼接，cat后数据的总维数不变。

~~~python
import torch
torch.manual_seed(1)
x = torch.randn(2,3)
y = torch.randn(1,3)

s = torch.cat((x,y),0)

print(x,y)
print(s)

output:
     0.6614  0.2669  0.0617
     0.6213 -0.4519 -0.1661
    [torch.FloatTensor of size 2x3]

    -1.5228  0.3817 -1.0276
    [torch.FloatTensor of size 1x3]
    
     0.6614  0.2669  0.0617
     0.6213 -0.4519 -0.1661
    -1.5228  0.3817 -1.0276
    [torch.FloatTensor of size 3x3]
~~~

注：torch.cat和torch.concat作用用法完全相同，只是concat的简写形式



**2.unsequeeze和sequeeze**

​	torch.sequeeze主要用于维度压缩，去除掉维数为1的维度。

~~~python
#1.删除指定的维数为1的维度
    #方式一：
    torch.sequeeze(a,2) 
    #方式二：
    a.sequeeze(2)

#2.删除全部维度为1的维度
	torch.sequeeze(a)		
~~~

​	torch.unsequeeze主要用于维度拓展，增加维数为1的维度。

~~~python
	torch.unsequeeze(a,2)   #在维度2增加维数为1的维度
~~~



