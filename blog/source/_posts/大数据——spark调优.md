---
title: 大数据——spark调优
date: 2019-05-20 09:39:03
tags: [大数据,面试]
---

### Spark调优核心参数设置

> **num-executors** 该参数一定被设置， 为当前Application生产指定个数的Executors 实际生产环境分配**80个左右**的Executors
> **executor-memory** 与**JVM OOM(内存溢出)**紧密相关，很多时候甚至决定了spark运行的性能 实际生产环境下建议**8GB左右** 若运行在yarn上，内存占用量不超过yarn的内存资源的50%
> **excutor-cores** 决定了在Executor中能够**并行执行的Task的个数** 实际生产环境建议**2~3个** 
>
> **driver-memory** 作为驱动，默认是1GB 生产环境一般设置4GB
>
> **spark.default.parallelism** 建议至少设置100个，**官方推荐是num-executors*excutor-cores的2~3倍**
> **spark.storage.memoryFraction** 用于存储的比例默认占用60%，如果计算比较依赖于历史数据，则可以适当调高该参数，如果计算严重依赖于shuffle，则需要降低该比例
> **spark.shuffle.memoryFraction** 用于shuffle的内存比例，默认占用20% 如果计算严重依赖于shuffle,则需要提高该比例



### spark生态的主要组件：

> spark core，任务调度，内存管理，错误恢复
> spark sql，结构化数据处理
> spark streaming，流式计算
> spark MLlib，机器学习库
> GraphX,图计算

### spark运行模式：

> 1. local模式
> 2. standalone模式，构建master+slave集群
> 3. Spark on Yarn模式
> 4. Spark on Mesos模式

### 宽窄依赖

> 1.窄依赖是1对1或1对多，宽依赖是多对1
>
> 2.窄依赖前一步map没有完全完成也可以进行下一步，在一个线程里完成不划分stage;宽依赖下一步需要依赖前一步的结果，划分stage
>
> 4.在传输上，窄依赖之间在一个stage内只需要做pipline，每个父RDD的分区只会传入到一个子RDD分区中，通常可以在一个节点内完成转换；宽依赖在stage做shuffle，需要在运行过程中将同一个父RDD的分区传入到不同的子RDD分区中，中间可能涉及多个节点之间的数据传输
>
> 3.容错上，窄依赖只需要重新计算子分区对应的负分区的RDD即可；宽依赖，在极端情况下所有负分区的RDD都要被计算



#### map­reduce中数据倾斜的原因?应该如何处理?

如何处理spark中的数据倾斜?

> **原因**：在物理执行期间，RDD会被分为一系列的分区，每个分区都是整个数据集的子集。当spark调度并运行任务的时候，Spark会为每一个分区中的数据创建一个任务。大部分的任务处理的数据量差不多，但是有少部分的任务处理的数据量很大，因而Spark作业会看起来运行的十分的慢，从而产生**数据倾斜**
>
> **处理方式：**
>
>   1.使用需要进行shuffle人工指定参数并行度
>
>   2.进行数据的清洗,把发生倾斜的刨除,用单独的程序去算倾斜的key
>
>   3.join的时候使用小数据join大数据时，换用map join
>
> 1. 尽量减少shuffle的次数

### Spark分区数设置

> **1、分区数越多越好吗？**
>
> 不是的，分区数太多意味着任务数太多（一个partion对应一个任务），每次调度任务也是很耗时的，所以分区数太多会导致总体耗时增多。
>
> 
>
> **2、分区数太少会有什么影响？**
>
> 分区数太少的话，会导致一些结点没有分配到任务；另一方面，分区数少则每个分区要处理的数据量就会增大，从而对每个结点的内存要求就会提高；还有分区数不合理，会导致数据倾斜问题。
>
> 
>
> **3、合理的分区数是多少？如何设置？**
>
> 总核数=executor-cores * num-executor 
>
> 一般合理的分区数设置为总核数的2~3倍





### Worker、Master、Executor、Driver 4大组件

> **1.master和worker节点**
>
> 搭建spark集群的时候我们就已经设置好了master节点和worker节点，一个集群有一个master节点和多个worker节点。
>
> **master节点常驻master守护进程，负责管理worker节点，我们从master节点提交应用。**
>
> **worker节点常驻worker守护进程，与master节点通信，并且管理executor进程。**
>
> 
>
> **2.driver和executor进程**
>
> driver进程就是应用的main()函数并且构建sparkContext对象，**当我们提交了应用之后，便会启动一个对应的driver进程，driver本身会根据我们设置的参数占有一定的资源**（主要指cpu core和memory）。下面说一说driver和executor会做哪些事。
>
> driver可以运行在master上，也可以运行worker上（根据部署模式的不同）。**driver首先会向集群管理者（standalone、yarn，mesos）申请spark应用所需的资源，也就是executor，然后集群管理者会根据spark应用所设置的参数在各个worker上分配一定数量的executor，每个executor都占用一定数量的cpu和memory**。**在申请到应用所需的资源以后，driver就开始调度和执行我们编写的应用代码了。driver进程会将我们编写的spark应用代码拆分成多个stage，每个stage执行一部分代码片段，并为每个stage创建一批tasks，然后将这些tasks分配到各个executor中执行。**
>
> 
>
> executor进程宿主在worker节点上，一个worker可以有多个executor。每个executor持有一个线程池，每个线程可以执行一个task，**executor执行完task以后将结果返回给driver**，每个executor执行的task都属于同一个应用。**此外executor还有一个功能就是为应用程序中要求缓存的 RDD 提供内存式存储，RDD 是直接缓存在executor进程内的，因此任务可以在运行时充分利用缓存数据加速运算。**
>
> driver进程会将我们编写的spark应用代码拆分成多个stage，每个stage执行一部分代码片段，并为每个stage创建一批tasks，然后将这些tasks分配到各个executor中执行。





### Spark是如何进行资源管理的？

> **1）资源的管理和分配** 
>
> 资源的管理和分配，**由Master和Worker来完成**。
>
> **Master给Worker分配资源， Master时刻知道Worker的资源状况。** 
>
> **客户端向服务器提交作业，实际是提交给Master。**
>
> 
>
> **2）资源的使用** 
>
> 资源的使用，由Driver和Executor。程序运行时候，向Master请求资源。



### Spark和mapreduce点的区别

> 优点：
>
> 1.最大的区别在于.spark把用到的中间数据放入内存，而mapreduce需要通过HDFS从磁盘中取数据。
>
> 2.spark算子多，mapreduce只有map和reduce两种操作
>
> 缺点：
>
> ​	spark过度依赖内存计算，如果参数设置不当，内存不够时就会因频繁GC导致线程等待



### 什么是RDD

> **RDD是一个只读的分布式弹性数据集**，是spark的基本抽象
>
> **主要特性：**
>
> ​	1.分布式。由多个partition组成，可能分布于多台机器，可**并行计算**
>
> ​	2.高效的容错（弹性）。通过RDD之间的依赖关系重新计算丢失的分区
>
> ​	3.只读。不可变



### RDD在spark中的运行流程？

> 1. 创建RDD对象
> 2. sparkContext负责计算RDD之间的依赖关系，构建DAG
> 3. DAGScheduler负责把DAG分解成多个stage(shuffle stage和final stage)，每个stage中包含多个task，每个task会被TAskScheduler分发给WORKER上的Executor执行

###spark任务执行流程：

> 1. Driver端提交任务，向Master申请资源
> 2. Master与Worker进行RPC通信，让Work启动Executor
> 3. Executor启动反向注册Driver，通过Driver--Master--Worker--Executor得到Driver在哪里
> 4. Driver产生Task，提交给Executor中启动Task去真正的做计算



### spark是如何容错的？

> **主要采用Lineage(血统)机制来进行容错，但在某些情况下也需要使用RDD的checkpoint**
>
> 1. 对于窄依赖，只计算父RDD相关数据即可，窄依赖开销较小
>
> 2. 对于宽依赖，需计算所有依赖的父RDD相关数据，会产生冗余计算，宽依赖开销较大。
>
>    
>
>    在两种情况下，RDD需要加checkpoint
>
>    1.DAG中的Lineage过长，如果重算，开销太大
>
>    2.在宽依赖上Cheakpoint的收益更大

### 一个RDD的task数量是又什么决定？一个job能并行多少个任务是由什么决定的？

> task由分区决定，读取时候其实调用的是hadoop的split函数，根据HDFS的block来决定
> 每个job的并行度由core决定



### cache与checkpoint的区别

>cache 和 checkpoint 之间有一个重大的区别，**cache 将 RDD 以及 RDD 的血统(记录了这个RDD如何产生)缓存到内存中**，当缓存的 RDD 失效的时候(如内存损坏)，
>它们可以通过血统重新计算来进行恢复。但是 **checkpoint 将 RDD 缓存到了 HDFS 中，同时忽略了它的血统(也就是RDD之前的那些依赖)**。为什么要丢掉依赖？因为可以**利用 HDFS 多副本特性保证容错**！



### reduceByKey和groupByKey的区别?

> **如果能用reduceByKey,那就用reduceByKey**.因为它**会在map端,先进行本地combine,可以大大减少要传输到reduce端的数据量,减小网络传输的开销**。
>
> groupByKey的性能,相对来说要差很多,因为它**不会在本地进行聚合**,而**是原封不动,把ShuffleMapTask的输出,拉取到ResultTask的内存中**,所以这样的话,就会导致,所有的数据,都要进行网络传输从而导致网络传输性能开销非常大!



### map和mapPartition的区别？

> **1.map是对rdd中的每一个元素进行操作；mapPartitions则是对rdd中的每个分区的迭代器进行操作**
> 如果是普通的map，比如一个partition中有1万条数据。ok，那么你的function要执行和计算1万次。使用MapPartitions操作之后，一个task仅仅会执行一次function，function一次接收所有的partition数据。只要执行一次就可以了，性能比较高。
>
> 2**.如果在map过程中需要频繁创建额外的对象**(例如将rdd中的数据通过jdbc写入数据库,map需要为每个元素
> 创建一个链接而mapPartition为每个partition创建一个链接),**则mapPartitions效率比map高的多**。
>
> 3.S**parkSql或DataFrame默认会对程序进行mapPartition的优化。**
>
> 
>
> mapPartition**缺点**：
> 	**一次性读入整个分区全部内容，分区数据太大会导致内存OOM**

### 详细说明一下GC对spark性能的影响?优化

> **GC会导致spark的性能降低**。因为**spark中的task运行时是工作线程,GC是守护线程**,**守护线程运行时,会让工作线程停止,所以GC运行的时候,会让Task停下来,这样会影响spark**
>
> 程序的运行速度,降低性能。
>     默认情况下,**Executor的内存空间分60%给RDD用来缓存,只分配40%给Task运行期间动态创建对象,这个内存有点小,很可能会发生full gc,**因为内存小就会导致创建的对象很快把内存填满,然后就会GC了,就是**JVM尝试找到不再被使用的对象进行回收,清除出内存空间**。所以如果Task分配的内存空间小,就会频繁的发生GC,从而导致频繁的Task工作线程的停止,从而降低Spark应用程序的性能。
>
> **优化方式**：
>
> ​	1**.增加executor内存**
>
> ​	2**.可以用通过调整executor比例**,比如将RDD缓存空间占比调整为40%,分配给Task的空间变为了60%,这样的话可以降低GC发生的频率   spark.storage.memoryFraction
>
> ​	2.**使用Kryo序列化类库进行序列化**



### 为什么要使用广播变量？

> **当RDD的操作要使用driver中定义的变量时,每次都要把变量发送给worker节点一次,如果这个变量的数据很大的话,会产生很高的负载,导致执行效率低**;
> **使用广播变量可以高效的使一个很大的只读数据发送给多个worker节点,而且对每个worker节点只需要传输一次,每次操作时executor可以直接获取本地保存的数据副本,不需要多次传输**