---
title: pytorch——finetune常用代码
date: 2019-11-07 23:42:25
tags: [pytorch,深度学习]
---



### fine-tune整体流程

> **1.加载预训练模型参数**
>
> **2.修改预训练模型，修改其后面的层为适合自己问题的层**
>
> **3.设置各层的可更新性。**前面和预训练模型相同的部分不再进行训练，后面新加的部分还要重新进行训练
>
> **4.检查各层可更新性（可选）**
>
> **5.设置优化器只对更新前面设置为可更新的部分。**



#### 1.加载预训练模型

​	一般在fine-tune中的第一步是首先加载一个已经预训练好的模型的参数，然后将预加载的模型后面的部分结构改造成自己需要的情况。其中包括两种情况：

> 1.单单将其中的一两个单独的层进行简单的改造（如预训练的模型输出的类为1000类，目前想要使用的模型只包含两个类），使用原有的预训练模型。
>
> 2.使用预训练模型的参数，但是后面的层需要更换为比较复杂的模型结构（常见的就是并行结构）

##### 1.使用torchvision中已经预训练好的模型

​	使用torchvision中已经预训练好的模型结构和参数，然后直接将尾部进行修改。

~~~python
from torchvision import models
from torch import nn
# 加载torchvision中已经训练好的resnet18模型，并且采用预训练的参数
resnet = models.resnet18(pretrained=True)
# 最后一层重新随机参数，并且将输出类别改为2
resnet.fc = nn.Linear(512,2)
~~~

##### 2.使用自己预训练好的模型，并且将输出的结果设置为并行结构

​	这里主要实现了之前自己已经预训练了，**重新定义整体模型的结构（创建一个新的模型类），然后将共有部分的参数加载进来，不同的地方使用随机参数**。

> 注意：这里面新旧模型要共用的层名称一定要一致

~~~python
from models import TextCNN

#加载新的模型结构，这里面的Text_CNN_Regression_Class模型结构已经设置为和之前的Text_CNN模型整体结构一致，最后的全连接层改为一个分类输出头加一个回归输出头
model = Text_CNN_Regression_Class(len(FILE_TYPE_COL))

# 加载预训练的模型的参数
pretrained_dict = torch.load("../model/Text_CNN_add_filetype_1:1_epoch5.state")
# 加载新的模型结构的初始化参数
model_dict = model.state_dict()
# 将pretrained_dict里不属于model_dict的键剔除掉
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}#如果你的k在预备训练当中，那么你的参数需要做转换，否则为原先的
# 更新现有的model_dict
model_dict.update(pretrained_dict)#利用预训练模型的参数，更新你的模型
# 加载我们真正需要的state_dict
model.load_state_dict(model_dict)
~~~



#### 2.将指定层设置为参数更新，其余设置为参数不更新

​	在fine-tune过程中经常用到的操作就是将整个神将网络的前半部分直接采用预训练好的模型参数，不再进行更新，这里主要实现了已经加载了预训练模型的参数，固定了除最后一个全连接层全部参数。

~~~python
#对于模型的每个权重，使其不进行反向传播，即固定参数
for param in resnet.parameters():
    param.requires_grad = False

#将其中最后的全连接部分的网路参数设置为可反向传播
for param in resnet.fc.parameters():
    param.requires_grad = True
~~~



#### 3.查看各层参数以及是否进行梯度更新（可选）

​	在fine-tune的过程中需要检查是不是已经将不需要更新梯度的层的各个参数值已经设置为不进行梯度更新，这是可以使用下面的代码进行查看:

~~~python
for child in resnet.children():
    print(child)
    for param in child.parameters():
        print(param.requires_grad)
~~~



#### 4..将优化器设置为只更新需要更新的部分参数

​	这里主要用于前面的各个参数是否进行更新已经设置完成后的最后一步，完成这一部就可以只接将优化器直接用于整个神经网络的重新训练了。

~~~python
optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
~~~

