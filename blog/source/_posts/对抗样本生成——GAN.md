---
title: 对抗样本生成——GAN
date: 2020-02-13 10:54:31
tags: [对抗样本生成]
mathjax: true
---

​	本文为对抗样本生成系列文章的第二篇文章，主要对GAN的原理进行介绍，并对其中关键部分的使用pytorch代码进行介绍，另外如果有需要完整代码的同学可以关注我的[github](https://github.com/AnchoretY/Webshell_Sample_Generate/blob/master/GAN%20image%20generate.ipynb)。

<!--more-->

该系列包含的文章还包括：

- [对抗样本生成—VAE](https://anchorety.github.io/2020/02/12/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E2%80%94%E2%80%94VAE/)
- [对抗样本生成—GAN]([https://anchorety.github.io/2020/02/13/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E2%80%94%E2%80%94GAN/](https://anchorety.github.io/2020/02/13/对抗样本生成——GAN/))
- [对抗样本生成—DCGAN]([https://anchorety.github.io/2020/02/13/%E5%AF%B9%E6%8A%97%E6%A0%B7%E6%9C%AC%E7%94%9F%E6%88%90%E2%80%94%E2%80%94DCGAN/](https://anchorety.github.io/2020/02/13/对抗样本生成——DCGAN/))
- [对抗样本生成—文本生成]()

### GAN(Generative Adversarial Network)

​	GAN中文名称生成对抗网络，是一种利用模型对抗技术来生成指定类型样本的技术，与VAE一起是目前主要的两种文本生成技术之一。GAN主要包含generater(生成器)和discriminator(判别器)两部分，generator负责生成假的样本来骗过discriminator，discriminator负责对样本进行打分，判断是否为生成网络生成的样本。

![](https://github.com/AnchoretY/images/blob/master/blog/GAN结构示意图.png?raw=true)

### Generator

>输入：noise sample（一个随机生成的指定纬度向量）
>
>输出：目标样本（fake image等）

​	Generator在GAN中负责接收随机的噪声输入，进行目标文本、图像的生成,其**目标就是尽可能的生成更加真实的图片、文字去欺骗discriminator**。具体的实现可以使用任何在其他领域证明有效的神经网络，本文使用最简单的全连接网络作为Generator进行实验。

~~~python
### 生成器结构
G = nn.Sequential(
        nn.Linear(latent_size, hidden_size), 
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, image_size),
        nn.Tanh())
~~~

### Discriminator

> 输入：样本（包含生成的样本和真实样本两部分）
>
> 输出：score（一个是否为真实样本的分数，分数越高是真实样本的置信的越高，越低越可能时生成样本）

​	Discriminator在GAN网络中负责将对输入的图像、文本进行判别，对其进行打分，打分越高越接近真实的图片，打分越低越可能是Generator生成的图像、文本，其**目标是尽可能准确的对真实样本与生成样本进行准确的区分**。与Generator一样Discriminator也可以使用任何网络实现，下面是pytorch中最简单的一种实现。

~~~python
### 判别器结构
D = nn.Sequential(
        nn.Linear(image_size, hidden_size), # 判别的输入时图像数据
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, 1),
        nn.Sigmoid())
~~~



### Model train

​	GAN中由于两部分需要进行对抗，因此两部分并不是与一般神经网络一样整个网络同时进行跟新训练的，而是两部分分别进行训练。训练的基本思路如下所示：

> Epoch:
>
>  	1. 生成器使用初始化的参数随机输入向量生成图片。
>
> 	2. 生成器进行判别，使用判别器结果对判器参数进行更新。
>  	3. 固定判别器参数，对生成器使用更新好的判别器进行

~~~python
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1) 
        # 创建标签，随后会用于损失函数BCE loss的计算
        real_labels = torch.ones(batch_size, 1)  # true_label设为1，表示True
        fake_labels = torch.zeros(batch_size, 1) # fake_label设为0，表示False
        # ================================================================== #
        #                      训练判别模型                      
        # ================================================================== #
        # 计算真实样本的损失
        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs
        # 计算生成样本的损失
        # 生成模型根据随机输入生成fake_images
        z = torch.randn(batch_size, latent_size)
        fake_images = G(z) 
        outputs = D(fake_images)
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs
        # 计算判别网络部分的总损失
        d_loss = d_loss_real + d_loss_fake
        # 对判别模型损失进行反向传播和参数优化
        d_optimizer.zero_grad()
    		g_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()
        
        # ================================================================== #
        #                       训练生成模型                       
        # ================================================================== #

        # 生成模型根据随机输入生成fake_images,然后判别模型进行判别
        z = torch.randn(batch_size, latent_size)
        fake_images = G(z)
        outputs = D(fake_images)
        
        # 大致含义就是在训练初期，生成模型G还很菜，判别模型会拒绝高置信度的样本，因为这些样本与训练数据不同。
        # 这样log(1-D(G(z)))就近乎饱和，梯度计算得到的值很小，不利于反向传播和训练。
        # 换一种思路，通过计算最大化log(D(G(z))，就能够在训练初期提供较大的梯度值，利于快速收敛
        g_loss = criterion(outputs, real_labels)
        
        # 反向传播和优化
        reset_grad()
        g_loss.backward()
        g_optimizer.step()
        
~~~

​	从上面的实现过程我们可以发现一个问题：在进行判别模型训练损失函数的计算由两部分组成，而生成模型进行训练时只由一部分组成，并且该部分的交叉熵还是一种反常的使用方式，这是为什么呢？

### 损失函数

​	整体的损失函数表现形式：

​											$$	\min\limits_{G}\max\limits_{D}E_{x\in\ P_{data}}\ [logD(x)]+E_{x\in\ P_{G}}\ [log(1-G(D(x)))]$$

#### Generator Loss

​	对于判别器进行训练时，其目标为：

​												$$\max\limits_{D}E_{x\in\ P_{data}}\ [logD(x)]+E_{x\in\ P_{G}}\ [log(G(1-D(x)))]$$

​	而对比交叉熵损失函数的计算公式：

​												$$ L = -[ylogp+(1-y)log(i-p)]$$

​	二者其实在表现形式形式上是完全一致的，这是因为判别器就是区分样本是否为真实的样本，是一个简单的0/1分类问题，所以形式与交叉熵一致。在另一个角度我们可以观察，当输入样本为真实的样本时，$E_{x\in\ P_{G}}\ [log(1-G(D(x)))]$为0，只剩下$E_{x\in\ P_{data}}\ [logD(x)]$，为了使其最大只能优化网络时D(x)尽可能大，即真实样本判别器给出的得分更高。当输入为生成样本时，$E_{x\in\ P_{data}}\ [logD(x)]$为0，只剩下$E_{x\in\ P_{G}}\ [log(1-G(D(x)))]$，为使其最大只能使D(x)尽可能小，即使生成样本判别器给出的分数尽可能低，使用交叉熵损失函数正好与目标相符。

​	因此，判别器训练相关的代码如下，其中可以看到损失函数**直接使用了二进制交叉熵**进行。

~~~python
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)


# 真实样本的损失
outputs = D(images)
d_loss_real = criterion(outputs, real_labels)
real_score = outputs
# 生成样本的损失
z = torch.randn(batch_size, latent_size)  # 生成模型根据随机输入生成fake_images
fake_images = G(z) 
outputs = D(fake_images)
d_loss_fake = criterion(outputs, fake_labels)
fake_score = outputs
# 计算判别网络部分的总损失
d_loss = d_loss_real + d_loss_fake
# 对判别模型损失进行反向传播和参数优化
d_optimizer.zero_grad()
g_optimizer.zero_grad()
d_loss.backward()
d_optimizer.step()
~~~



#### Discriminator Loss

​	对于生成器其训练的目标为：

​											$$\min\limits_{G}\max\limits_{D}E_{x\in\ P_{data}}\ [logD(x)]+E_{x\in\ P_{G}}\ [log(1-G(D(x)))]（其中D固定）$$

​	对于生成器，在D固定的情况下，$E_{x\in\ P_{data}}\ [logD(x)]$为固定值，因此可以不做考虑，表达式转为：

​												$$	\min\limits_{G}\max\limits_{D}E_{x\in\ P_{G}}\ [log(1-G(D(x)))]（其中D固定）$$

​	使用该表达式作为目标函数进行参数更新存在的问题就是在训练的起始阶段，由于开始时生成样本的质量很低，因此判别器很容易给一个很低的分数，即D(x)非常小，而log(1-x)的函数在值接近0时斜率也很小，因此使用该函数作为损失函数在开始时很难进行参数更新。

<img src="https://github.com/AnchoretY/images/blob/master/blog/GAN生成器损失函数对比.png?raw=true =100*100" style="zoom:50%;" />

​	因此生成器采用了一种与log（1-x）的更新方向一致并且在起始时斜率更大的函数。

​											$$E_{x\in P_{G}}[-logG(D(x))]$$

​	该损失函数在代码实现中一般还是**使用反标签的二进制交叉熵损失函数来进行实现**，所谓反标签即为将生成的样本标注为1进行训练（正常生成样本标签为0），涉及到该部分的代码为：

~~~python
criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)


real_label = torch.ones(batch_size, 1) 

# 生成模型根据随机输入生成fake_images,然后判别模型进行判别
z = torch.randn(batch_size, latent_size)
fake_images = G(z)
outputs = D(fake_images)

# 训练生成模型，使用反标签的二进制交叉熵损失函数
g_loss = criterion(outputs, real_labels)

# 反向传播和优化
reset_grad()
g_loss.backward()
g_optimizer.step()
~~~



### GAN与VAE对比

​	GAN和VAE都是样本生成领域非常常用的两个模型流派，那这两种模型有什么不同点呢？

> 1. VAE进行对抗样本生成时，VAE的Encoder和GAN的Generator输入同样都为图片等真实样本，但**VAE的Encoder输出的中间结果为隐藏向量值**，而**GAN的Generator输出的中间结果为生成的图片等生成样本**。
>
> 2. **最终用来生成样本的部分不同**。VAE最终使用Decoder部分来进行样本生成，GAN使用Generator进行样本生成。

​	在实际的使用过程中还存在这下面的区别使GAN比VAE更被广泛使用：

> 1. VAE生成样本点的连续性不好。VAE进行生成采用的方式是每个像素点进行生成的，很难考虑像素点之间的联系，因此经常出现一些不连续的坏点。
>
> 2. 要生成同样品质的样本，VAE需要更大的神经网络。



【参考文献】

李宏毅在线课程:https://www.youtube.com/watch?v=DQNNMiAP5lw&list=PLJV_el3uVTsMq6JEFPW35BCiOQTsoqwNw  

GAN损失函数详解:https://www.cnblogs.com/walter-xh/p/10051634.html



