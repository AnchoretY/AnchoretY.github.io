---
title: python进阶-面向对象编程
date: 2019-01-08 19:24:57
tags: python
---



1.\__slots__

​	用于指定class 实例能够指定的属性

> 注意：\__slots__只对当前类起作用，对其子类无效

~~~python
import traceback
class Myclass(object):
	__slots__ = ["name","set_name]
	
s = MyClass()
s.name = "john" #这里可以进行正常的赋值，因为包含在__slots__中
try:
	s.age = 2	#这里不能进行正常赋值
except AttributeError:
	traceback.print_exc()
~~~

Output:

![](https://github.com/AnchoretY/images/blob/master/blog/__slots__.png?raw=true)



**2.@property属性**

​	@property 可以实现比较方便的属性set、get设置

> 1.使用@property相当于讲将一个函数变为get某个属性值
> 2.@属性名称.setter可以实现设置一个属性的set条件

​	使用上面的两种修饰符，可以实现

​		1.对写入属性的限制，只有符合规范的才允许写入

​		2.设置只读属性，只能够读取，不能写入，只能从其他属性处计算出

下面的就是对score属性的写操作进行了一些限制，将double_score属性设置为只读属性

~~~python

class MyClass(object):
    
    @property
    def score(self):
        return self._score
    
    @score.setter
    def score(self,value):
        #不是int类型时引发异常
        if not isinstance(value,int):
            raise ValueError("not int")    #raise的作用是显示的引发异常
        #超出范围时引发异常
        elif (value<0) or (value>100):
            raise ValueError("score must in 0 to 100")
        
        self._score = value
        
    @property
    def double_score(self):
        return self._score*2
        
    
s = MyClass()
s.score = 3
print(s.score)
try:
    s.score = 2300
except ValueError:
    traceback.print_exc()
    
try:
    s.score = "dfsd"
except ValueError:
    traceback.print_exc()
    
print(s.double_score)

try:
    s.double_score = 2
except Exception:
    traceback.print_exc()
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/@property_2.png?raw=true)





描述器，主要是用来读写删除类的行为





  函数可以直接使用\__name__属性来获取函数名称

~~~python
def now():
	print("2012")

print(now.__name__)

output:
	"now"
~~~



​	

