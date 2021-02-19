---
title: __new__和__init__
date: 2019-04-26 22:40:33
tags: [python,面试]
---



**执行顺序：类中同时出现new()和init()时，先调用new()，再调用init()**



##### python中\_\_new\__和\_\_init__的区别

> **1.用法不同**：
>
> ​	\_\_new__()用于创建实例，所以该方法是**在实例创建之前被调用**，它是类级别的方法，是个静态方法；
>
> ​	\_\_init__() 用于初始化实例，所以该方法是**在实例对象创建后被调用**，它是实例级别的方法，用于设置对象属性的一些初始值
>
> ​	注：由此可知，\_\_new\_\_()在\_\_init\_\_() 之前被调用。如果\_\_new\_\_() 创建的是当前类的实例，会自动调用\_\_init\_\_()函数，通过return调用的\_\_new\_\_()的参数cls来保证是当前类实例，如果是其他类的类名，那么创建返回的是其他类实例，就不会调用当前类的\_\_init\_\_()函数
>
> 2.传入参数不同：
>
> ​	\_\_new\_\_()**至少有一个参数cls，代表当前类**，此参数在实例化时由Python解释器自动识别；
>
> ​	\_\_init\_\_()**至少有一个参数self**，就是这个\_\_new\_\_()返回的实例，\_\_init\_\_()在\_\_new\_\_()的基础上完成一些初始化的操作。
>
> 3.返回值不同：
>
> ​	\_\_new\_\_()必须有返回值，返回实例对象；
>
> 　\_\_init\_\_()不需要返回值。



#### \_\_new\_\_的两种常见用法

##### 1.继承不可变的类

​	\_\_new\_\_()方法主要用于继承一些不可变的class，比如int, str, tuple， 提供一个自定义这些类的实例化过程的途径，一般通过重载\_\_new\_\_()方法来实现

~~~python
class PostiveInterger(int):
	def __new__(cls,value):
    return super(PostiveInterger,cls).__new__(cls,abs(value))
a = PostiveInterger(-10)
print(a)

output:
  10
~~~



##### 2.实现单例模式

​	可以用来实现单例模式，也就是使**每次实例化时只返回同一个实例对象**。

~~~python
class Singleobject(object):
  def __new__(cls):
    if not cls.instance:
      cls.instance = super(Singleobject,cls).new(cls)
    return cls.instance

object1 = Singleobject()
object2 = Singleobject()

object1.attr = 'value1'

print(object1.attr,object2.attr)
print(object1 is object2)

output:
  value1,value1
  True
      
~~~

