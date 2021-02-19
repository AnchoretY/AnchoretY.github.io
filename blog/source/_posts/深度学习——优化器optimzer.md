---
title: 深度学习——优化器optimzer
date: 2018-10-30 20:19:24
tags: [深度学习,面试]
categories: [深度学习,面试]
---

​	在机器学习和深度学习中，选择合适的优化器不仅可**以加快学习速度**，而且可以**避免在训练过程中困到的鞍点**。

#### 1.Gradient Descent （GD）

​	BGD是一种使用全部训练集数据来计算损失函数的梯度来进行参数更新更新的方式，梯度更新计算公式如下：

​	![](https://github.com/AnchoretY/images/blob/master/blog/BGD%E5%8F%82%E6%95%B0%E6%9B%B4%E6%96%B0%E5%85%AC%E5%BC%8F.png?raw=true)

~~~python
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
~~~



> 缺点：
>
> 1.**由于在每一次更新中都会对整个数据及计算梯度，因此计算起来非常慢**，在大数据的情况下很难坐到实时更新。
>
> ​	**2.Batch gradient descent 对于凸函数可以收敛到全局极小值，对于非凸函数可以收敛到局部极小值。**



#### 2.Stochastic Gradient Descent(SGD)

​	SGD是一种最常见的优化方法，这种方式**每次只计算当前的样本的梯度，然后使用该梯度来对参数进行更新**，其计算方法为：

​	![SGD计算公式](https://github.com/AnchoretY/images/blob/master/blog/SGD%E6%A2%AF%E5%BA%A6%E6%9B%B4%E6%96%B0%E8%A7%84%E5%88%99.png?raw=true)

~~~python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
~~~

​	随机梯度下降是通过每个样本来迭代更新一次，如果样本量很大的情况，那么可能只用其中部分的样本，就已经将theta迭代到最优解了，对比上面的批量梯度下降，迭代一次需要用到十几万训练样本，一次迭代不可能最优，如果迭代10次的话就需要遍历训练样本10次。	

> **缺点：1.存在比较严重的震荡**
>
> ​	**2.容易收敛到局部最优点,但有时也可能因为震荡的原因跳过局部最小值**



#### 3.Batch Gradient Descent （BGD）

​	BGD **每一次利用一小批样本，即 n 个样本进行计算**，这样它可以**降低参数更新时的方差，收敛更稳定**，**另一方面可以充分地利用深度学习库中高度优化的矩阵操作来进行更有效的梯度计算**。

​	![](https://github.com/AnchoretY/images/blob/master/blog/MBGD%E5%8F%82%E6%95%B0%E5%85%AC%E5%BC%8F.png?raw=true)

~~~python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
~~~

​	**参数值设定：batch_szie一般在设置在50~256之间**

> **缺点：1.不能保证很好的收敛性。**
>
> ​	**2.对所有参数进行更新时使用的是完全相同的learnnning rate**

​	这两个缺点也是前面这几种优化方式存在的共有缺陷，下面的优化方式主要就是为了晚上前面这些问题



#### 4.Momentum

> **核心思想：用动量来进行加速**
>
> **适用情况：善于处理稀疏数据**

​	为了克服 SGD 振荡比较严重的问题，Momentum 将物理中的动量概念引入到SGD 当中，通过积累之前的动量来替代梯度。即:

![SGD计算公式](https://github.com/AnchoretY/images/blob/master/blog/Momentum%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.png?raw=true)



​	其中，γ 表示动量大小，μ表示学习速率大小。

​	相较于 SGD，Momentum 就相当于在从山坡上不停的向下走，当没有阻力的话，它的动量会越来越大，但是如果遇到了阻力，速度就会变小。也就是说，**在训练的时候，在梯度方向不变的维度上，训练速度变快，梯度方向有所改变的维度上，更新速度变慢，这样就可以加快收敛并减小振荡。**	

​	**超参数设定：一般 γ 取值 0.9 左右。**

> 缺点：**这种情况相当于小球从山上滚下来时是在盲目地沿着坡滚，如果它能具备一些先知，例如快要上坡时，就知道需要减速了的话，适应性会更好。**



#### 5.Adaptive gradient algorithm（Adagrad）

> **核心思想：对学习速率添加约束，前期加速训练，后期提前结束训练以避免震荡，减少了学习速率的手动调节**
>
> **适用情况：这个算法可以对低频参数进行较大的更新，高频参数进行更小的更新，对稀疏数据表现良好，提高了SGD的鲁棒性，善于处理非平稳目标**

​	相较于 SGD，Adagrad 相当于对学习率多加了一个约束，即：

![SGD计算公式](https://github.com/AnchoretY/images/blob/master/blog/Adagrad参数更新公式.png?raw=true)

​	对于经典的SGD：

​		![](https://github.com/AnchoretY/images/blob/master/blog/SGD与Adagrad对比.png?raw=true)

​	而对于Adagrad：

​	![](https://github.com/AnchoretY/images/blob/master/blog/Adagrad和SGD对比.png?raw=true)

其中，r为梯度累积变量，r的初始值为0。ε为全局学习率，需要自己设置。δ为小常数，为了数值稳定大约设置为10-7	

​	**超参数设定：一般η选取0.01，ε一般设置为10-7**

​				

> 缺点：分母会不断积累，这样学习速率就会变得非常小



#### 6.Adadelta

​	超参数设置：p 0.9

​	Adadelta算法是基于Adagrad算法的改进算法，主要改进主要包括下面两点：

> 1.将分母从G换成了**过去梯度平方的衰减的平均值**
>
> 2.将初始学习速率换成了**RMS[Δθ]**(梯度的均方根)

##### part one

​	(1) 将累计梯度信息从**全部的历史信息**变为**当前时间窗口向前一个时间窗口内的累积**：

![](https://github.com/AnchoretY/images/blob/master/blog/Adadelta改进1.png?raw=true)

​	(2)将上述公式进行开方，作为每次迭代更新后的学习率衰减系数

![](https://github.com/AnchoretY/images/blob/master/blog/adadelta改进2.png?raw=true)



记

![](https://github.com/AnchoretY/images/blob/master/blog/Adadelta改进3.png?raw=true)

其中 是为了防止分母为0加上的一个极小值。

​	这里解决了梯度一直会下降到很小的值得问题。

##### part two

​	将原始的学习速率换为参数值在前一时刻的RMS

![](https://github.com/AnchoretY/images/blob/master/blog/Adadelta最终更新公式.png?raw=true)

​	因为原始的学习速率已经换成了前一时刻的RMS值，因此，**对于adadelta已经不需要选择初始的学习速率**



#### 7.RMSprop

​	RMSprop 与 Adadelta 的第一种形式相同：

![](https://github.com/AnchoretY/images/blob/master/blog/RMSprop参数更新公式.png?raw=true)

​	**使用的是指数加权平均，旨在消除梯度下降中的摆动，与Momentum的效果一样，某一维度的导数比较大，则指数加权平均就大，某一维度的导数比较小，则其指数加权平均就小，这样就保证了各维度导数都在一个量级，进而减少了摆动。允许使用一个更大的学习率η**

​	**超参数设置：建议设定 γ 为 0.9, 学习率 η 为 0.001**



#### 7.Adam

> **核心思想：结合了Momentum动量加速和Adagrad对学习速率的约束**
>
> **适用情况：各种数据，前面两种优化器适合的数据Adam都更效果更好**，

​	Adam 是一个结合了 Momentum 与 Adagrad 的产物，它既考虑到了利用动量项来加速训练过程，又考虑到对于学习率的约束。利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率。Adam 的优点主要在于经过偏置校正后，每一次迭代学习率都有个确定范围，使得参数比较平稳。其公式为:	

​	![SGD计算公式](https://github.com/AnchoretY/images/blob/master/blog/Adam%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F1.png?raw=true)

​	其中：

![SGD计算公式](https://github.com/AnchoretY/images/blob/master/blog/Adam%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F2.png?raw=true)

![SGD计算公式](https://github.com/AnchoretY/images/blob/master/blog/Adam%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F3.png?raw=true)



#### 总结：在实际工程中被广泛使用，但是也可看到在一些论文里存在着许多使用Adagrad、Momentum的，杜对于SGD由于其需要更多的训练时间和鞍点问题，因此在实际工程中很少使用



### 如何选择最优化算法

​	1.如果数据是稀疏的，就是自适应系列的方法 Adam、Adagrad、Adadelta

​	2.Adam 就是在 RMSprop 的基础上加了 bias-correction 和 momentum

​	3.随着梯度变的稀疏，Adam 比 RMSprop 效果会好。

​	整体来说Adam是最好的选择



**参考文献:深度学习在美团点评推荐系统中的应用**

https://blog.csdn.net/yukinoai/article/details/84198218