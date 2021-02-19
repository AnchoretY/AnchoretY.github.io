---
title: python多进程
date: 2018-12-30 11:12:53
tags: python
categories: python进阶操作
---

​	python多进程之前一直在在写一些小程序，这次正好需要写一个比较正式的多进行处理，正好系统的总结下python多进行的一些核心知识。

​	首先python中常用的提高效率的方式主要主要包括多线程、多进程两种，但是在python中的多线程并不是正正的多线程，只有在网络请求密集型的程序中才能有效的提升效率，在计算密集型、IO密集型都不是很work甚至会造成效率下降，因此提升效率的方式就主要以多进程为主。

​	python中多进程需要使用python自带包multiprocessing，multiprocessing支持子进程、通信和共享数据、执行不同形式的同步，提供了Process、Queue、Pipe、Lock等组件

~~~python
from multiprocessing import Lock,Value,Queue,Pool
~~~



### 创建进程类

#### 1.单个进程类的创建

​	**1.创建单进程**

​		

~~~

~~~

####使用进程池创建多进程

​	Pool类可以提供指定数量的进程供用户调用，当有新的请求提交到Pool中时，如果池还没有满，就会创建一个新的进程来执行请求。如果池满，请求就会告知先等待，直到池中有进程结束，才会创建新的进程来执行这些请求。 

​	常用方法：

​	**1.apply**

​	用于传递不定参数，同python中的apply函数一致，主进程会被阻塞直到函数执行结束（**不建议使用**，并且3.x以后不在出现）。

​	**2.apply_async**

​	和apply类似，非阻塞版本，支持结果返回后进行回调

​	**3.map**

​	函数原型:map(func, iterable[, chunksize=None])

​	Pool中的map和python内置的map用法基本上一致，会阻塞直到返回结果

​	**4.map_async**

​	函数原型：map_async(func, iterable[, chunksize[, callback]])

​	和map用法一致，但是它是非阻塞的

​	**5.close**

​	关闭进程池，使其不再接受新的任务

​	**6.terminal**

​	结束工作进程，不再处理未处理的任务

​	**7.join**

​	主进程阻塞等待子进程退出，join方法要在close或terminal方法后面使用

​	





~~~python
from mutiprocess import Pool

process_nums = 20
pool = Pool(process_nums)

~~~



### 使用Lock来避免冲突

​	lock主要用于多个进程之间共享资源时，避免资源访问冲突，主要包括下面两个操作：

​		1.look.acquire()  获得锁

​		2.lock.release()   释放锁

​	下面是不加锁时的程序：

~~~python
import multiprocessing
import time
def add(number,value,lock):
    print ("init add{0} number = {1}".format(value, number))
    for i in xrange(1, 6):
        number += value
        time.sleep(1)
        print ("add{0} number = {1}".format(value, number))
        
if __name__ == "__main__":
    lock = multiprocessing.Lock()
    number = 0
    p1 = multiprocessing.Process(target=add,args=(number, 1, lock))
    p2 = multiprocessing.Process(target=add,args=(number, 3, lock))
    p1.start()
    p2.start()
    print ("main end")
~~~

​	结果为：

~~~python
main end
init add1 number = 0
init add3 number = 0
add1 number = 1
add3 number = 3
add1 number = 2
add3 number = 6
add1 number = 3
add3 number = 9
add1 number = 4
add3 number = 12
add1 number = 5
add3 number = 15
~~~

​	两个进程交替的来对number进行加操作，下面是加锁后的程序：

~~~python
import multiprocessing
import time
def add(number,value,lock):
    lock.acquire()
    try:
        print ("init add{0} number = {1}".format(value, number))
        for i in xrange(1, 6):
            number += value
            time.sleep(1)
            print ("add{0} number = {1}".format(value, number))
    except Exception as e:
        raise e
    finally:
        lock.release()

if __name__ == "__main__":
    lock = multiprocessing.Lock()
    number = 0
    p1 = multiprocessing.Process(target=add,args=(number, 1, lock))
    p2 = multiprocessing.Process(target=add,args=(number, 3, lock))
    p1.start()
    p2.start()
    print ("main end")

~~~

​	结果为：

~~~python
main end
init add1 number = 0			#add1优先抢到锁，优先执行
add1 number = 1
add1 number = 2
add1 number = 3
add1 number = 4
add1 number = 5
init add3 number = 0			#add3被阻塞，等待add1执行完成，释放锁后执行add3
add3 number = 3
add3 number = 6
add3 number = 9
add3 number = 12
add3 number = 15
#注意观察上面add3部分，虽然在add1部分已经将number加到了5，但是由于number变量只是普通变量，不能在各个进程之间进行共享，因此add3开始还要从0开始加
~~~





####使用Value和Array来进行内存之中的共享通信

​	一般的变量在进程之间是没法进行通讯的，multiprocessing 给我们提供了 **Value 和 Array** 模块，他们可以在不通的进程中共同使用。

​	**1.Value多进程共享变量**

​	将前面加锁的程序中的变量使用multiprocessing提供的共享变量来进行

~~~python
import multiprocessing
import time
def add(number,add_value,lock):
    lock.acquire()
    try:
        print ("init add{0} number = {1}".format(add_value, number.value))
        for i in xrange(1, 6):
            number.value += add_value
            print ("***************add{0} has added***********".format(add_value))
            time.sleep(1)
            print ("add{0} number = {1}".format(add_value, number.value))
    except Exception as e:
        raise e
    finally:
        lock.release()
        
if __name__ == "__main__":
    lock = multiprocessing.Lock()
    number = multiprocessing.Value('i', 0)
    p1 = multiprocessing.Process(target=add,args=(number, 1, lock))
    p2 = multiprocessing.Process(target=add,args=(number, 3, lock))
    p1.start()
    p2.start()
    print ("main end")
~~~

​	输出结果为：

~~~python
#add3开始时是在add1的基础上来进行加的
main end
init add1 number = 0
***************add1 has added***********
add1 number = 1
***************add1 has added***********
add1 number = 2
***************add1 has added***********
add1 number = 3
***************add1 has added***********
add1 number = 4
***************add1 has added***********
add1 number = 5
init add3 number = 5
***************add3 has added***********
add3 number = 8
***************add3 has added***********
add3 number = 11
***************add3 has added***********
add3 number = 14
***************add3 has added***********
add3 number = 17
***************add3 has added***********
add3 number = 20
~~~

​	**2.Array实现多进程共享内存变量**

~~~python
import multiprocessing
import time
def add(number,add_value,lock):
    lock.acquire()
    try:
        print ("init add{0} number = {1}".format(add_value, number.value))
        for i in xrange(1, 6):
            number.value += add_value
            print ("***************add{0} has added***********".format(add_value))
            time.sleep(1)
            print ("add{0} number = {1}".format(add_value, number.value))
        except Exception as e:
            raise e
        finally:
            lock.release()

def change(arr):
    for i in range(len(arr)):
        arr[i] = -arr[i]
        
if __name__ == "__main__":
    lock = multiprocessing.Lock()
    number = multiprocessing.Value('i', 0)
    arr = multiprocessing.Array('i', range(10))
    print (arr[:])
    p1 = multiprocessing.Process(target=add,args=(number, 1, lock))
    p2 = multiprocessing.Process(target=add,args=(number, 3, lock))
    p3 = multiprocessing.Process(target=change,args=(arr,))
    p1.start()
    p2.start()
    p3.start()
    p3.join()
    print (arr[:])
    print ("main end")
~~~

​	输出结果为：

~~~python
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
init add3 number = 0
***************add3 has added***********
[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]
main end
add3 number = 3
***************add3 has added***********
add3 number = 6
***************add3 has added***********
add3 number = 9
***************add3 has added***********
add3 number = 12
***************add3 has added***********
add3 number = 15
init add1 number = 15
***************add1 has added***********
add1 number = 16
***************add1 has added***********
add1 number = 17
***************add1 has added***********
add1 number = 18
***************add1 has added***********
add1 number = 19
***************add1 has added***********
add1 number = 20
~~~



####使用Queue来实现多进程之间的数据传递

​	Queue是多进程安全队列，可以使用Queue来实现进程之间的数据传递，使用的方式：

> 1.put 将数据插入到队列中
>
> ​	包括两个可选参数：blocked和timeout
>
> ​		(1)如果blocked为True（默认为True）,并且timeout为正值，该方法会阻塞队列指定时间，直到队列有剩余，如果超时，会抛出Queue.Full
>
> ​		(2)如果blocked为False，且队列已满，那么立刻抛出Queue.Full异常
>
> 2.get 从队列中读取并删除一个元素
>
> ​	包括两个可选参数:block和timeout
>
> ​		（1）blocked为True，并且timeout为正值，那么在等待时间结束后还没有取到元素，那么会抛出Queue.Empty异常
>
> ​		（2）blocked为False，那么对列为空时直接抛出Queue.Empty异常
>
>



### python实现多进程最优方式

​	python中自带的joblib包自带了