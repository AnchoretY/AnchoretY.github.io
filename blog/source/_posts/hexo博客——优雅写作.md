---
title: hexo博客——优雅写作
date: 2020-03-19 12:03:36
tags: [博客]
---

&emsp;&emsp;最近时间闲暇萌生了好好进行一些博客管理的念头，为了以后写文章更加方便美观，写了本篇博客意在将本人如何进行博客写作以及如何使博客文章内容更加美观进行记录，以供有同样需要的人进行参考。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.cmcvxwn4wlu.png)

<!-- more-->



## 写作技巧使用

#### 添加背景色

&emsp;为段落添加背景色可以通过note标签来完成，其标准格式为:

>{% note [class] %}
>文本内容 (支持行内标签)
>{% endnote %}

&emsp;&emsp;支持的class类型包括多种，包括`default` `primary` `success` `info` `warning` `danger`，也可以不指定 class。下面各种class对应的颜色效果展示：

> {% note  %}
> Default
> {% endnote %}

{% note  default %}
Default
{% endnote %}

{% note  primary %}
primary
{% endnote %}

{% note  success %}
success
{% endnote %}

{% note  info %}
info
{% endnote %}

{% note  warning %}
warning
{% endnote %}

{% note  danger %}
danger
{% endnote %}

&emsp;&emsp;这里支持的背景色显示形式也支持多种风格，可以直接在主题配置文件中进行设置

```shell themes/next/_config.yml
note:
  # Note 标签样式预设
  style: modern  # simple | modern | flat | disabled
  icons: false  # 是否显示图标
  border_radius: 3  # 圆角半径
  light_bg_offset: 0  # 默认背景减淡效果，以百分比计算
```



#### 流程图

&emsp;&emsp;要在站点中使用流程图，首先要在主题的_config.yml文件中进行设置

```diff themes/next/_config.yml
mermaid:
- enable: false
+ enable: true
  # Available themes: default | dark | forest | neutral
  theme: forest
```

&emsp;&emsp;然后就可以在网站中按照mermaid语法进行流程图的构建了，下面是一个简单的流程图构建的例子，更加详细的流程图构建语法可以参见这里

~~~mermaid
graph LR
A(sql注入)  --> B(普通注入)
A --> C(圆角长方形)
C-->D(布尔型盲注)
C-->E(延时盲注)
~~~









## 模板

评论

