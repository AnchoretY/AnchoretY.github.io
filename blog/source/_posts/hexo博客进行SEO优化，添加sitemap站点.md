---
title: hexo博客进行SEO优化，添加sitemap站点
date: 2020-03-17 12:12:31
tags: [博客]
---

&emsp;一篇关于如何对个人博客网站进行基本的SEO优化的文章,使你的博客可以在Google上被检索.

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.lb78fc2qbm.png)

<!-- more-->

## Sitemap是什么

​	&emsp;&emsp;**Sitemap中文名称站点地图**，他可以是任意形式的文档用来**采用分级的形式描述了一个网站的架构**，从而**帮助访问者以及搜索引擎的机器人找到网站中的页面**，**使所有页面可被找到来增强搜索引擎优化的效果**，因此**添加站点地图对于优化SEO来说很重要**。

图片



## 添加站点地图

#### 1. 安装hexo-generator-sitemap

&emsp;&emsp;常规操作，在Git Bash中输入

```
npm install hexo-generator-sitemap --save   #适用于google
npm install hexo-generator-baidu-sitemap --save   #适用于baidu
```

#### 2.设置内容

- **根目录下的`_config.yml`配置文件**

  &emsp;&emsp;在文件末尾中添加：

  ```xml
  ## Sitemap
  sitemap:
    path: sitemap.xml
  ```

  &emsp;&emsp;修改文件中的url为博客地址：

  ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.oinrou1zu1.png)

- **生成`sitemap.xml`文件**

  &emsp;&emsp;使用`hexo g`生成命令就在`yourwebsite.github.io\public`中生成`sitemap.xml`然后将其中生成的文件移动到网站根目录下（因为我的public目录在浏览器无法访问，可以直接访问可以不做）

  > 这里需要前面的hexo-generator-sitemap工具安装成功了才能生成

- **新建`robots.txt`文件**

  &emsp;&emsp;在网站根目录`/source`目录中新建一个`robots.txt`，该文件为是**Robots协议**的文件，用来**告诉引擎哪些页面可以抓取，哪些不能**。

  ~~~shell
  User-agent: *
  Allow: /
  Allow: /archives/
  Allow: /categories/
  Allow: /tags/
  Allow: /about/
  
  Disallow: /vendors/
  Disallow: /js/
  Disallow: /css/
  Disallow: /fonts/
  Disallow: /fancybox/
  
  Sitemap: https://anchorety.github.io/sitemap.xml
  ~~~

- **使用`hexo d -g`进行部署**

#### 3. Google Search Console设置

- **访问[Google Search Console](https://search.google.com/search-console)网址,使用google账号进行登录**

- **添加博客域名**

  ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.omqozl59j4.png)

  &emsp;&emsp;点击继续后会进入网站所有权验证,选择html标签验证。

  ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.14iwxldo9xy.png)

  &emsp;&emsp;在网站`themes/next/layout/_partials/head.swig`头部加入提示中的连接

  ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.935oysn75jk.png)

  &emsp;&emsp;验证成功后进入网站资源管理页面

  ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.9x38d5f8lyv.png)

- **添加站点地图**

  &emsp;&emsp;点击左侧导航栏中站点地图，然后输入前面sitemap.xml文件存放的地址(我的地址是网站根目录下，直接输入sitemap.xml即可)，提交后出现成功状态即可。

  ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.ps2s9psp4do.png)

>&emsp;&emsp;这里如果出现了404错误，可以在浏览器中直接输入输入的地址尝试是否能够访问，不能访问则将sitemap换到其他可以进行访问的地址再重复操作.

####  4.等待

&emsp;&emsp;在进行完前面的环节后，控制台界面大部分时候都只能显示

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.vpplz2askz.png)

&emsp;&emsp;只需要等待Google进行页面爬取就好了，一般等待一两天后就可以完全被Google收录了。可是使用google搜索下面内容进行收录查验：

~~~
site: xxx.github.com
~~~

&emsp;&emsp;完全被收录后：

图片