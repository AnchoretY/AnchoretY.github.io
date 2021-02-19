---
title: sql常见面试题
date: 2019-08-23 14:02:22
tags: [sql,面试]
---

#### 1.用一条语句查询出每门课都大于80分的学生姓名

| name | class | Socre |
| ---- | ----- | ----- |
| 张三 | 语文  | 81    |
| 张三 | 数学  | 75    |
| 李四 | 语文  | 76    |
| 李四 | 数学  | 90    |
| 王五 | 语文  | 81    |

解法一：

~~~sql
select distinct name from table where name not in (
	select distinct name from table where score<=80
)
~~~

解法二：

~~~sql
select name from table groupby name having min(fenshu)>80
~~~



#### 2.删除除了自动编号不同外，其他都相同的学生冗余信息

| 自动编号 | 学号    | 姓名 | 课程编号 | 课程名称 | 课程分数 |
| -------- | ------- | ---- | -------- | -------- | -------- |
| 1        | 2005001 | 张三 | 0001     | 数学     | 69       |
| 2        | 2005002 | 李四 | 0001     | 数学     | 80       |
| 3        | 2005001 | 张三 | 0001     | 数学     | 69       |

~~~sql
delete tablename where 自动编号 not in (select min(自动编号) groupby 学号，姓名，课程编号，课程名称，课程分数)
~~~



#### 3.有两个表A 和B ，均有key 和value 两个字段，如果B 的key 在A 中也有，就把B 的value 换为A 中对应的value

~~~mysql
update b set b.value=(select a.value from a where a.key=b.key) where b.id in (select b.id from b,a where b.key=a.key);
~~~



#### 5.查询表A中存在ID重复三次以上的记录

~~~mysql
select * from(select count(ID) as count from table group by ID)T where T.count>3
~~~



#### 6.取出每个班级成绩前两名的同学，表结构为sno、sname、class、score

~~~sql
select sname,class,score from grade where (
  select count(*) from grade as f where f.class==grade.class and f.score>=grade.score
) <=2
~~~









#### 6.经典的学习成绩问题

已知关系模式：

​	s (sno,sname) 学生关系。sno 为学号，sname 为姓名
​	c (cno,cname,cteacher) 课程关系cno 为课程号，cname 为课程名，cteacher 为任课教师
​	sc(sno,cno,scgrade) 选课关系。scgrade 为成绩

**1．找出没有选修过“李明”老师讲授课程的所有学生姓名**

~~~mysql
select sname from s where cno in (select cno from c where cteacher=='李明')
~~~

**2．列出有二门以上（含两门）不及格课程的学生姓名及其平均成绩**

~~~

~~~



3．列出既学过“1”号课程，又学过“2”号课程的所有学生姓名
4．列出“1”号课成绩比“2”号同学该门课成绩高的所有学生的学号
5．列出“1”号课成绩比“2”号课成绩高的所有学生的学号及其“1”号课和“2”号课的成绩

