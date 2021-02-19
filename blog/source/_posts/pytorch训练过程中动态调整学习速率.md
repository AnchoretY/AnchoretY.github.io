---
title: pytorch训练过程中动态调整学习速率
copyright: true
mathjax: true
date: 2020-11-02 10:19:39
tags:
categories:
---

概述：本文主要讲述要在pytorch中设置学习速率自动调整的方法，即如何使用`torch.optim.lr_scheduler`。


![]()

<!--more-->

`torch.optim.lr_scheduler` 提供了几种方法来根据epoches的数量调整学习率。

### ReduceLROnPlateau

&emsp;&emsp; `torch.optim.lr_scheduler.ReduceLROnPlateau`允许基于一些验证测量来降低动态学习速率。

```python
class torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
```

1. #### 每隔参数组的

&emsp;&emsp;将每个参数组的学习速率设置为初始的lr乘以一个给定的函数。当last_epoch=-1时，将初始lr设置为lr。

参数：

1. **optimizer** (Optimizer) – 包装的优化器。
2. **lr_lambda** (function or list) – 一个函数来计算一个乘法因子给定一个整数参数的`epoch`，或列表等功能，为每个组`optimizer.param_groups`。
3. **last_epoch** (int) – 最后一个时期的索引。默认: -1.

例子：

```python
>>> # Assuming optimizer has two groups.
>>> lambda1 = lambda epoch: epoch // 30
>>> lambda2 = lambda epoch: 0.95 ** epoch
>>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)
class torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)
```

2. #### 每隔step_size学习速率变化一次

&emsp;&emsp;将每个参数组的学习速率设置为每个step_size时间段由gamma衰减的初始lr。当last_epoch = -1时，将初始lr设置为lr。

1. **optimizer** (Optimizer) – 包装的优化器。
2. **step_size** (int) – 学习率衰减期。
3. **gamma** (float) – 学习率衰减的乘积因子。默认值:-0.1。
4. **last_epoch** (int) – 最后一个时代的指数。默认值:1。

例子：

```python
>>> # Assuming optimizer uses lr = 0.5 for all groups
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 60
>>> # lr = 0.0005   if 60 <= epoch < 90
>>> # ...
>>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)
class torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
```

3. #### MultiStepLR

&emsp;&emsp;一旦时间的数量达到一个里程碑,则将每个参数组的学习率设置为伽玛衰减的初始值。当last_epoch=-1时，将初始lr设置为lr。

参数：

1. optimizer (Optimizer) – 包装的优化器。
2. milestones (list) – 时期指标的列表。必须增加。
3. gamma (float) – 学习率衰减的乘积因子。 默认: -0.1.
4. last_epoch (int) – 最后一个时代的指数。 默认: -1.

例子：

```python
>>> # Assuming optimizer uses lr = 0.5 for all groups
>>> # lr = 0.05     if epoch < 30
>>> # lr = 0.005    if 30 <= epoch < 80
>>> # lr = 0.0005   if epoch >= 80
>>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
>>> for epoch in range(100):
>>>     scheduler.step()
>>>     train(...)
>>>     validate(...)
```

4. #### ExponentialLR

```python
class torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
```

&emsp;&emsp;将每个参数组的学习速率设置为每一个时代的初始lr衰减。当last_epoch=-1时，将初始lr设置为lr。

1. optimizer (Optimizer) – 包装的优化器。
2. gamma (float) – 学习率衰减的乘积因子。
3. last_epoch (int) – 最后一个指数。默认: -1.



5. #### ReduceLROnPlateau

```python
class torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
```

&emsp;&emsp;**当指标停止改善时，降低学习率**。当学习停滞不前时，模型往往会使学习速度降低2-10倍。这个调度程序读取一个指标量，如果没有提高epochs的数量，学习率就会降低。

1. optimizer (Optimizer) – 包装的优化器。
2. mode (str) – min, max中的一个. 在最小模式下，当监测量停止下降时，lr将减少; 在最大模式下，当监控量停止增加时，会减少。默认值：'min'。
3. factor (float) – 使学习率降低的因素。 new_lr = lr * factor. 默认: 0.1.
4. patience (int) –epochs没有改善后，学习率将降低。 默认: 10.
5. verbose (bool) – 如果为True，则会向每个更新的stdout打印一条消息。 默认: False.
6. threshold (float) – 测量新的最优值的阈值，只关注显着变化。 默认: 1e-4.
7. threshold_mode (str) – rel, abs中的一个. 在rel模型, dynamic_threshold = best *( 1 + threshold ) in ‘max’ mode or best* ( 1 - threshold ) 在最小模型. 在绝对值模型中, dynamic_threshold = best + threshold 在最大模式或最佳阈值最小模式. 默认: ‘rel’.
8. cooldown (int) – 在lr减少后恢复正常运行之前等待的时期数。默认的: 0.
9. min_lr (float or list) – 标量或标量的列表。对所有的组群或每组的学习速率的一个较低的限制。 默认: 0.
10. eps (float) – 适用于lr的最小衰减。如果新旧lr之间的差异小于eps，则更新将被忽略。默认: 1e-8.

```python
>>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
>>> scheduler = torch.optim.ReduceLROnPlateau(optimizer, 'min')
>>> for epoch in range(10):
>>>     train(...)
>>>     val_loss = validate(...)
>>>     # Note that step should be called after validate()
>>>     scheduler.step(val_loss)
```



##### 参考文献

- [[pytorch中文文档\] torch.optim - pytorch中文网](https://ptorch.com/docs/1/optim)