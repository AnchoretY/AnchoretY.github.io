---
title: pytorch运算操作
date: 2018-09-23 17:49:58
tags: pytorch
categories: pytorch

---



**1.transponse**

​	torch.transponse操作是转置操作，是一种在矩阵乘法中最常用的几种操作之一。

~~~python
#交换两个维度
torch.transponse(dim1,dim2)
~~~



**2.matmul和bmm**

​	matmul和bmm都是实现batch的矩阵乘法操作。

~~~python
a = torch.rand((2,3,10))
b = torch.rand((2,2,10))

res1 = torch.matmul(a,b.transpose(1,2))
#res1 = torch.matmul(a,b.transose(1,2))
print res1.size()

output:
    [torch.FloatTensor of size 2x3x2]
~~~

