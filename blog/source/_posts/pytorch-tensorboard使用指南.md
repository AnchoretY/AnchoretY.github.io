---
title: pytorch_tensorboard使用指南
date: 2019-11-15 11:03:07
tags: [pytorch,可视化,数据分析]
---

​	最近pytorch官网推出了对tensorboard支持，因此最近准备对其配置和使用做一个记录。



### 安装

​	要在使用pytorch时使用tensorboard进行可视化第一就是软件的安装，整个过程中最大的问题就是软件的兼容性的问题了，下面是我再使用过程中确定可兼容的版本：

~~~linux
python 3.x
pytorch 1.1.0
tensorboard 1.1.4
~~~

​	兼容的基础软件安装完成后，在安装依赖包

~~~python
pip install tensorboard future jupyter 
~~~

​	安装成功后就可以直接在正常编写的pytorch程序中加入tensorboard相关的可视化代码，并运行。下面是测试代码：

~~~python
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 定义网络
class Test_model(nn.Module):
    def __init__(self):
        super(Test_model, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    def forward(self, x):
        return self.layer(x)

model = Test_model()

writer = SummaryWriter()
writer.add_graph(model, input_to_model=torch.randn((3,3)))
writer.add_scalar(tag="test", scalar_value=torch.tensor(1)
                    , global_step=1)
writer.close()
~~~

​	运行成功后，就可以使用shell进入到项目的运行文件的目录,这是可以看到目录下产生了一个新的runs目录，里面就是运行上面代码产生出的可视化文件。在文件的目录中输入

~~~
tensorboard --logdir=runs
~~~

> 注意：这里输入命令的目录一定要为文件的运行目录，runs文件夹的外面。

​	最后，按照提示在浏览器中打开[http://localhost:6006](https://link.zhihu.com/?target=http%3A//localhost%3A6006/)，显示如下网页，恭喜你成功了

![](https://github.com/AnchoretY/images/blob/master/blog/tensorboard成功部署页面.png?raw=true)



### TensorBoard常用功能

​	tensorBoard之所以如此受到算法开发和的热捧，是因为其只需要是使用很简单的接口，就可以在实现很复杂的可视化功能，可以我们更好的发现模型存在的各种问题，以及更好的解决问题，其核心功能包括：

> 1.模型结构可视化
>
> 2.损失函数、准确率可视化
>
> 3.各层参数更新可视化

在TensorBoard中提供了各种类型的数据向量化的接口，主要包括：

| pytorch生成函数 | pytorch界面栏 | 显示内容                                                     |
| --------------- | ------------- | ------------------------------------------------------------ |
| add_scalar      | SCALARS       | 标量(scalar)数据随着迭代的进行的变化趋势。常用于损失函数和准确率的变化图生成 |
| add_graph       | GRAPHS        | 计算图生成。常用于模型结构的可视化                           |
| add_histogram   | HISTOGRAMS    | 张量分布监控数据随着迭代的变化趋势。常用于各层参数的更新情况的观察 |
| add_text        | TEXT          | 观察文本向量在模型的迭代过程中的变化。                       |

​	下面将具体介绍使用各个生成函数如何常用的功能。

#### 1.模型结构可视化（add_scalae使用）

​	模型结构可视化一般用于形象的观察模型的结构，包括模型的层级和各个层级之间的关系、各个层级之间的数据流动等，这里要使用的就是计算图可视化技术。

​	首先，无论使用TensorBoard的任何功能都要先生成一个SummaryWriter，是一个后续所有内容基础，对应了一个TensorBoard可视化文件。

~~~python
from torch.utils.tensorboard import SummerWriter

# 这里的参数主要有三个
# log_dir 文件的生成位置,默认为runs
# commment 生成文件内容的描述，最后会被添加在文件的结尾
writer = SummaryWriter(logdir="xxx",commit='xxx')
~~~

​	然后正常声明模型结构。

~~~python
class Test_model(nn.Module):
    def __init__(self):
        super(Test_model, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        return self.layer(x)
~~~

​	在**前面创建的writer基础上增加graph**，实现模型结构可视化。

~~~python
model = Test_Model()

# 常见参数
# model 要进行可视化的模型
# input_to_model 要输入到模型中进行结构和速度测试的测试数据
writer.add_graph(model,torch.Tensor([1,2,3]))

# writer关闭
writer.close()
~~~

> 注意：模型结构和各层速度的测试是在模型的正常训练过程中使用，而是在模型结构定义好以后，使用一些随机自定义数据进行结构可视化和速度测试的。

​	最终在TensorBoard的GRAPHS中可以看到模型结构(**点击查看具体的模型结构和各个结构所内存和消耗时间**)

<img src="https://github.com/AnchoretY/images/blob/master/blog/tensorboard成功部署页面.png?raw=true" alt="tensorboard成功部署页面.png" style="zoom:55%;" />

#### 2.损失函数准确率可视化

​	损失函数和准确率更新的可视化主要用于模型的训练过程中观察模型是否正确的在被运行，是否在产生了过拟合等意外情况，这里主要用到的是scalar可视化。

​	损失函数和准确率的可视化主要用在训练部分，因此假设模型的声明已经完成，然后进行后续的操作：

~~~python
# 将模型置于训练模式
model.train()
output = model(input_data)

writer = SummaryWriter(comment='测试文件')

# 标准的训练
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    output_data = model(input_data)
    loss = F.cross_entropy(output_data,label)
    pred = output_data.data.max(1)[1]
    acc = pred.eq(label).sum()
    loss.backward()
    optimizer.step()
    
    # 在每一轮的训练中都进行acc和loss记录，写入tensrboard日志文件
    writer.add_scalar(tag='acc',scalar_value=acc,global_step=epoch)
    writer.add_scalar(tag="loss", scalar_value=loss,global_step=epoch)
    
# 关闭tensorboard写入器
writer.close()
~~~

​	最终效果如下图。

<img src="https://github.com/AnchoretY/images/blob/master/blog/tensorboard损失函数、准确率迭代图.png?raw=True alt=" alt=" tensorboard损失函数、准确率迭代图.png" title="tensorboard成功部署页面.png&quot; style=&quot;zoom:25%; " style="zoom:45%;"	 />

#### 3.各层参数更新可视化

​	各层参数可视化，是发现问题和模型调整的重要依据，我们**常常可以根据再训练过程中模型各层的输出和各层再反向传播时的梯度来进行是否存在梯度消失现象**，具体的使用可以参照文章[如何发现将死的ReLu](https://www.toutiao.com/i6759006512414228995/?tt_from=weixin&utm_campaign=client_share&wxshare_count=1&timestamp=1573973465&app=news_article&utm_source=weixin&utm_medium=toutiao_android&req_id=20191117145104010020047015100AB118&group_id=6759006512414228995)。

​	下面我们来具体讲解如何进行各层参数、输出、以及梯度进行可视化。这里用的主要是add_histgram函数来进行可视化。

~~~python
# 将模型置于训练模式
model.train()
output = model(input_data)

writer = SummaryWriter(comment='测试文件')

# 标准的训练
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    output_data = model(input_data)
    loss = F.cross_entropy(output_data,label)
    pred = output_data.data.max(1)[1]
    acc = pred.eq(label).sum()
    loss.backward()
    optimizer.step()
    
    # 在每一轮的训练中都记录各层的各个参数值和梯度分布，写入tensrboard日志文件
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        # 记录各层的参数值
        writer.add_histogram(tag, value.data.cpu().numpy(), epoch)
        # 记录各层的梯度
        writer.add_histogram(tag+'/grad', value.grad.data.cpu().numpy(), epoch)
    
# 关闭tensorboard写入器
writer.close()
~~~

​	最终效果如下图所示。

<img src="https://github.com/AnchoretY/images/blob/master/blog/tensorboard训练中参数和提取情况图.png?raw=True alt=" alt=" tensorboard损失函数、准确率迭代图.png" title="tensorboard成功部署页面.png&quot; style=&quot;zoom:25%; " style="zoom:45%;"	 />

> 注：在histogram中，横轴表示值，纵轴表示数量，各条线表示不同的时间线(step\epoch)，将鼠标停留在一个点上，会加黑显示三个数字，含义是：在step xxx1时，有xxx2个元素的值（约等于）xxx3。