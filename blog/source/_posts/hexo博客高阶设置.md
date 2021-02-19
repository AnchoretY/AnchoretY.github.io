---
title: hexo博客高阶设置
date: 2020-03-16 12:08:18
tags: [博客]
---

&emsp;&emsp;本文主要讲述Hexo博客的目录结构和首页文章展现形式、热文推荐等博客的一些高级用法，提供给对博客有进一步了解需求和向进一步增加博客功能的朋友们。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.tne4lo95zfk.png)

<!--more-->

### Hexo博客目录结构

&emsp;&emsp;Hexo目录博客如下：

```
hexo-install-directory
├── CNAME
├── _config.yml 				//Hexo的配置文件，可以配置主题、语言等
├── avatar.jpg
├── db.json
├── debug.log
├── node_modules
│   ├── hexo
│   ├── hexo-deployer-git
│   ├── hexo-generator-archive
│   ├── hexo-generator-category
│   ├── hexo-generator-feed
│   ├── hexo-generator-index
│   ├── hexo-generator-sitemap
│   ├── hexo-generator-tag
│   ├── hexo-migrator-wordpress
│   ├── hexo-renderer-ejs
│   ├── hexo-renderer-marked
│   ├── hexo-renderer-stylus
│   └── hexo-server
├── package.json
├── public						//执行hexo g命令后，生成的内容会在这里，包括所有的文章、页面、分类、tag等.
│   ├── 2013
│   ├── 2014
│   ├── 2015
│   ├── 2016
│   ├── 404.html
│   ├── Staticfile
│   ├── archives
│   ├── atom.xml
│   ├── categories
│   ├── css
│   ├── images
│   ├── index.html
│   ├── js
│   ├── page
│   ├── sitemap.xml
│   └── vendors
├── scaffolds					//保存着默认模板，自定义模板就是修改该目录下的文件
│   ├── draft.md 				//默认的草稿模板
│   ├── page.md 				//默认的页面模板
│   └── post.md 				//默认的文章模板
├── source 						//Hexo存放编辑页面的地方，可以使用vim或其他编辑器编辑这里的内容
│   ├── 404.html   				//自定义404页面，可以使用腾讯公益404页面
│   ├── Staticfile 				
│   ├── _drafts					//存放所有的草稿文件的目录
│   ├── _posts					//存放所有的文章文件的目录，用的最多，比如执行hexo n "post_name"之后，post_name这篇文章就存放在这个目录下
│   ├── categories
│   └── images					//这是我自己定义的，用于存放个人的avatar
└── themes						//Hexo的所有主题
    ├── landscape				
    ├── next					//这是我目前用的主题
    └── yilia
```



### 博客高级设置

#### 1. 设置首页文章列表不显示全文(只显示预览)

&emsp;&emsp;编辑进入hexo博客项目的**themes/next/_config.yml**文件,搜索"**auto_excerpt**",找到如下部分：

```shell
# Automatically Excerpt. Not recommand.
# Please use <!-- more --> in the post to control excerpt accurately.
  auto_excerpt:
  enable: false
  length: 150
```

​	将**enable修改为true，length设置为150**，然后**hexo d -g**重新进行网站部署。

用户可以在文章中通过 `<!-- more -->` 标记来精确划分摘要信息，标记之前的段落将作为摘要显示在首页。



#### 2.增加文章热度统计





#### 3.网址英文显示



##### 参考文献

- http://yearito.cn/posts/hexo-advanced-settings.html
- https://blog.xinspace.xin/2016/04/11/自定义Hexo博客的文章、草稿和页面的模板/
- https://www.cnblogs.com/zhansu/p/9643066.html