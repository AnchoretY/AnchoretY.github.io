---
title: 对抗样本生成——VAE
date: 2020-02-12 19:03:16
tags: [对抗样本生成]
categories: 
---

​	最近由于进行一些类文本生成的任务，因此对文本生成的相关的一些经典的可用于样本生成的网络进行了研究，本系列文章主要用于对这些模型及原理与应用做总结，不涉及复杂的公式推导。

<!--more-->

相关文章：

- [对抗样本生成—VAE](https://anchorety.github.io/2020/02/12/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E2%80%94%E2%80%94VAE/)
- [对抗样本生成—GAN]([https://anchorety.github.io/2020/02/13/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E2%80%94%E2%80%94GAN/](https://anchorety.github.io/2020/02/13/对抗样本生成——GAN/))
- [对抗样本生成—DCGAN]([https://anchorety.github.io/2020/02/13/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E2%80%94%E2%80%94DCGAN/](https://anchorety.github.io/2020/02/13/对抗样本生成——DCGAN/))
- [对抗样本生成—文本生成]()

### AE(Auto Encoder)

​	Auto Encoder中文名自动编码机，最开始用于数据压缩任务，例如：Google曾尝试使用该技术将图片再网络上只传输使用AE压缩过的编码值，而在本地进行还原来节约流量。后来也用于样本生成任务，但是用于样本生成存在着一些不可避免的问题，因此很快被VAE所取代。Auto Encoder结构如下所示：

![](https://github.com/AnchoretY/images/blob/master/blog/AE结构图.png?raw=true)

​	主要**由Encoder和Decoder两部分组成**，**Encoder**负责将原始的图片、文本等输入**压缩**成更低纬度的向量进行表示，**Decoder**负责将该向量表示进行**复原**，然后通过最小化Encoder输入与Decoder输出来进行两部分模型参数的优化。

​	训练完成后**，训练好的Encoder部分可以输入图片等数据进行数据压缩**；

**AE进行数据压缩的特点：**

>1.只能压缩与数据高相关度的数据
>
>2.有损压缩

​	**训练好的decoder可以输入随机的向量值生成样本**，下图为样本生成示意图。

![](https://github.com/AnchoretY/images/blob/master/blog/AE生成样本.png?raw=true)

**AE在进行样本生成时存在的问题：**

> 1.当输入随机向量进行样本生成时，decoder部分输入的是一个随机的向量值，而AE只能保证训练集中有的数据具有比较好的效果，但是无法保证与训练集中的数据很接近的值依旧能够准确的进行判断（不能保证不存在跳变）。
>
> 2.没有随心所欲的去构造向量。因为输入的向量必须由原始的样本区进行构造隐藏编码，才能进行样本生成。



### VAE(Varaient Auto Encoder)

​	Variational Autoencoder中文名称变分自动编码器，是Auto Encoder的进化版，主要用于解决AE中存在的无法随心所欲的去生成样本，模型存在跳变等问题。**核心思想为在生成隐藏向量的过程中加入一定的限制，使模型生成的样本近似的遵从标准正态分布，这样要进行样本生成我们就可以直接向模型输入一个标准正态分布的隐向量即可。**有需要完整版代码的同学可以参见我的[github](https://github.com/AnchoretY/Webshell_Sample_Generate/blob/master/VAE%20image%20generate.ipynb)

<img src="https://github.com/AnchoretY/images/blob/master/blog/VAE结构图.png?raw=true =100*100" style="zoom:20%;" />

​		VAE结构如上图所示。与AE一样，VAE的主要结构依然是分为Encoder和Decoder两个主要组成部分，这两部分可以使用任意的网络结构进行实现，而其中的**不同点主要在于隐向量的方式不同和因此导致生成样本所需的原料不同。**

​	VAE的使用过程中，需要在模型生成样本的准确率与生成隐向量符合正态分布的成都之间做一个权衡，因此在VAE中**loss中包含两部分：均方误差、KL散度**。均方误差用来衡量原始图片与生成图片之间的误差，KL散度用于表示隐含向量与标准正态分布之间的差距，其计算公式如下所示：

​																	$$DKL(P||Q) = \int_{-\infty}^{\infty} P(x)log\frac{p(x)}{q(x)}dx$$

​	KL散度很难进行计算，因此在VAE中使用了一种”重新参数化“技巧来解决。即VAE的encoder不再直接输出一个隐含向量，而是生成两个向量，一个代表均值，一个代表方差，然后通过这两个向量与一个标准正态分布向量去合成出一个符合标准整体分布的隐含向量。其合成计算公式为：

​																	$$z = \mu+\sigma \cdot \epsilon$$

​	其中，u为均值向量，$\sigma$为方差向量，$\epsilon$为标准的正态分布向量。

​	而VAE的代码实现也非常的简单，其核心的代码实现如下所示：

~~~python
class VAE(nn.Module):
    def __init__(self, image_size=784, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(image_size, h_dim)
        self.fc2 = nn.Linear(h_dim, z_dim) # 均值 向量
        self.fc3 = nn.Linear(h_dim, z_dim) # 保准方差 向量
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
    # 编码过程
    def encode(self, x):
        h = F.relu(self.fc1(x))
        return self.fc2(h), self.fc3(h)
    
    # 随机生成隐含向量
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 解码过程
    def decode(self, z):
        h = F.relu(self.fc4(z))
        return F.sigmoid(self.fc5(h))
    
    # 整个前向传播过程：编码-》解码
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
~~~

​	在我的[github](https://github.com/AnchoretY/Webshell_Sample_Generate/blob/master/VAE%20image%20generate.ipynb)上还有完整的将VAE应用到手写数字生成的代码，需要的同学可以关注一下。

### 总结

​	VAE与AE的对比：

> **1.隐藏向量的生成方式不同。**
>
> ​	AE的Encoder直接生成隐藏向量，而VAE的Encoder是生成均值向量和方差向量再加上随机生成的正态分布向量来进行合成隐藏向量。
>
> **2.样本生成能力不同。**这也是AE在对抗样本生成领域中很少被使用的主要原因
>
> ​	AE要进行样本生成只能使用已有样本生成的隐含向量作为输入输入到Decoder中，由于已有样本有限，因此能够生成的对抗样本数量有限。
>
> ​	VAE可以直接使用符合正态分布的任意向量直接输入到Decoder中进行样本生成，能够任意进行样本生成。



