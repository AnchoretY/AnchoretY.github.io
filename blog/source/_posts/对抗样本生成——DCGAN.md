---
title: 对抗样本生成——DCGAN
date: 2020-02-13 23:51:32
tags: [对抗样本生成]
mathjax: true
---

​	本文为对抗样本生成系列文章的第三篇文章，主要对DCGAN的原理进行介绍，并对其中关键部分的使用pytorch代码进行介绍，另外如果有需要完整代码的同学可以关注我的[github](https://github.com/AnchoretY/Webshell_Sample_Generate/blob/master/GAN%20image%20generate.ipynb)。

<!--more-->

该系列包含的文章还包括：

- [对抗样本生成—VAE](https://anchorety.github.io/2020/02/12/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E2%80%94%E2%80%94VAE/)
- [对抗样本生成—GAN]([https://anchorety.github.io/2020/02/13/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E2%80%94%E2%80%94GAN/](https://anchorety.github.io/2020/02/13/对抗样本生成——GAN/))
- [对抗样本生成—DCGAN]([https://anchorety.github.io/2020/02/13/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E2%80%94%E2%80%94DCGAN/](https://anchorety.github.io/2020/02/13/对抗样本生成——DCGAN/))
- [对抗样本生成—文本生成]()



​	DCGAN时CNN与GAN相结合的一种实现方式，源自于论文《Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks》，该模型主要讨论了如何将CNN引入GAN网络，将CNN引入GAN网络并非直接将Generator和Discriminator的全连接网络直接替换成CNN即可，而是要对CNN网络进行特定的设计才能使CNN网络有效的快速收敛。本文将对DCGAN中一些核心观点进行详细论述并对其中核心部分的代码实现进行解析，完整的代码实现可以关注我的[github](https://github.com/AnchoretY/Webshell_Sample_Generate/blob/master/DCGAN%20image%20generate.ipynb).

​	DCGAN中一些核心关键点如下：

> **1.激活函数**：
>
> ​	**生成器除最后一层的输出使用Tanh激活函数外统一使用relu**作为激活函数，
>
> ​	**判别器所有层都是用LeakyRelu激活函数**(这里很关键，还是使用relu的话很可能造成模型很难进行有优化，最终模型输出的图像一致和目标图像相差很远)
>
> 2**.生成器和判别器的模型结构复杂度不要差距太大**
>
> ​	复杂度差距过大会导致模型训练后期一个部分的效果非常好，能够不断提升，但是另一个部分的由于模型过于简单无法再优化导致该部分效果不断变差。
>
> 3.**判别器最后的全连接层使用卷积层代替。**
>
> ​	全部的操作均使用卷积操作进行。
>
> 4.**Batch Normalization区别应用：**
>
> ​	不能将BN层应用到生成网络和判别网络的全部部分，**在生成网络的输出层和判别网络的输入层不能使用BN层**，否则可能造成模型的不稳定。



### 原始对抗训练细节实现

>预处理：将输入图像的各个像素点标准化到tanh的值域范围[-1,1]之内
>
>权重初始化:均值为0方差为0.02的正态分布
>
>Relu激活函数斜率：0.2
>
>优化器：Adam  0.01或0.0002

### Generator

​	生成器主要反卷积层、BN层、激活函数层三部分堆叠而成，其结构如下所示：

![](https://github.com/AnchoretY/images/blob/master/blog/DCGAN生成器结构.png?raw=true)

> 在DCGAN中一个最为核心的结构就是反卷积层，那么什么是反卷积层呢？
>
> ​	**反卷积是图像领域中常见的一种上采样操作**，反卷积**并不是正常卷积的逆过程，而是一种特殊的正向卷积**，先按照一定的比例通过补 0来扩大输入图像的尺寸，接着旋转卷积核，再进行正向卷积，这种特殊的卷积操作**只能能够复原矩阵的原始尺寸，不能对原矩阵的各个元素的内容进行复原。**

生成器实现中核心点包括：

> 1.使用反卷积进行一步一步的图片生成
>
> 2.最后的输出层中不使用BN
>
> 3.除输出层使用tanh激活函数外，其它层都使用relu激活函数

​	代码实现如下(该代码为手写数字图片生成项目中的实现，真实维度为28*28)：

~~~python
class Generator(nn.Module):
    def __init__(self):
        super(Generator,self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(latent_size,128,4,1,0,bias=False),  #使用反卷积进行还原(b,512,4,4)
            nn.BatchNorm2d(128),
            nn.ReLU(True)    #生成器中除输出层外均使用relu激活函数
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1,bias=False),  ##使用反卷积进行还原(b,64,8,8)
            nn.BatchNorm2d(64),
            nn.ReLU(True),   #生成器中除输出层外均使用relu激活函数
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1,bias=False),  ##使用反卷积进行还原(b,8,16,16)
            nn.BatchNorm2d(32),
            nn.ReLU(True)   #生成器中除输出层外均使用relu激活函数
        )
        # 生成器的输出层不使用BN
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(32,1,4,2,3,bias=False),  ##使用反卷积进行还原(b,1,28,28)
            nn.Tanh(),         
        ) 
    def forward(self,input_data):
        x = self.layer1(input_data)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
~~~

### Discriminator

​	判别器主要为实现对图片是否为生成图片。在DCGAN中主要使用CNN、BN和LeakyRelu网络来进行，其实现的核心点包括：

> 1.判别网络全部使用卷积操作来搭建，整个过程中不包含全连接层和池化层。
>
> 2.判别器激活函数除最后一层使用Sigmod激活函数外，全部使用LeakyRelu激活函数
>
> 3.判别器的输入层中不能使用BN层

~~~python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1,16,4,2,3),     #(b,13,16,16)
            nn.LeakyReLU(0.2,True)
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(16,32,4,2,1),    #(b,32,8,8)
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2,True),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv2d(32,64,4,2,1),    #(b,64,4,4)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2,True)
        )
        self.cnn4 = nn.Sequential(
            nn.Conv2d(64,1,4,2,0),   #(b,1,1,1)
            nn.Sigmoid()
        )

    def forward(self,input_data):
        x = self.cnn1(input_data)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        return x
~~~



参考文献：

DCGAN pytorch教程：https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html