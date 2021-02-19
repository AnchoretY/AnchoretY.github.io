---
title: hexo博客升级
date: 2020-03-18 20:50:15
tags: [博客]
---

&emsp;&emsp;最近将hexo博客的next主题从5.11更新到7.7.2，写下本篇博客记录下完整的更新过程，以待有需要的同学使用。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.dnybrnent6k.png)

<!-- more -->

#### 1.将当前目录下已有的next文件夹重命名为next2

~~~shell themes/
	mv next next2
~~~
#### 2.从github上下载最新的next主题

&emsp;&emsp;在网站目录下重新clone一份新的next，将其存储为next

~~~shell
	git clone https://github.com/theme-next/hexo-theme-next themes/next
~~~

#### 3.修改主题目录下的_config.xml文件

&emsp;&emsp;参照next2文件夹中的_config.yml文件修改next文件夹中的_config.yml文件

```shell themes/next/_config.yml
#主题选择
scheme: Gemini

# 菜单栏
menu:
  home: / || home
  about: /about/ || user
  tags: /tags/ || tags
  categories: /categories/ || th
  archives: /archives/ || archive

# github、Email显示
social:
  GitHub: https://github.com/xxx || github
  E-Mail: xxx@gmail.com || envelope

# 增加评论功能
gitalk:
  enable: true
  github_id: AnchoretY # GitHub repo owner
  repo: AnchoretY.github.io # Repository name to store issues
  client_id: xxx # GitHub Application Client ID
  client_secret: xxx # GitHub Application Client Secret
  admin_user: xxx # GitHub repo owner and collaborators, only these guys can initialize gitHub issues
  distraction_free_mode: true # Facebook-like distraction free mode
  language: zh-CN

# 搜索功能
local_search:
	enable: true 

# 右边栏引用开源协议
creative_commons:
  license: by-nc-sa
  sidebar: true   #边栏
	post: true   #文章底部

  
# 修改左侧目录导航
toc:
  number: false #是否对标题自动编号
  enable: true   #开启目录导航
  wrap: false
  expand_all: true  #默认展开全部内容
  max_depth: 4    # 导航栏最大显示的目录层级

# 修改站角信息显示
footer:  
  since: 2018  # 建站时间
  icon:
    name: heart   # 图标名称
    animated: true   # 开启动画
    color: "#ff0000"   # 图标颜色

  powered:
    enable: true  # 显示由 Hexo 强力驱动
    version: false  # 隐藏 Hexo 版本号

  theme:
    enable: true  # 显示所用的主题名称
    version: false  # 隐藏主题版本号

  
```
自己打开了不蒜子统计，只需要在页面中添加

#### 4.更该页面语言

&emsp;&emsp;在新版本的next主题中将以前的zh-xxx改为了zh-CN,需要将站点目录下的_config.yml文件作相应更改。

~~~diff /_config.xml diff:true
- language: zh-xxx
+ language: zh-CN
~~~

#### 5.字数与阅读时长统计

&emsp;&emsp;首先安装hexo-symbols-count-time插件

~~~shell
npm install hexo-symbols-count-time
~~~

&emsp;&emsp;接下来在站点目录下_config.yml文件中添加：

~~~shell
symbols_count_time:
  symbols: true                # 文章字数统计
  time: true                   # 文章阅读时长
  total_symbols: false          # 站点总字数统计
  total_time: false             # 站点总阅读时长
  exclude_codeblock: false     # 排除代码字数统计
~~~

&emsp;&emsp;最后修改主题目录中的_config.yml文件:

~~~shell themes/next/_config.xml
symbols_count_time:
  separated_meta: true     # 是否另起一行（true的话不和发表时间等同一行）
  item_text_post: true     # 首页文章统计数量前是否显示文字描述（本文字数、阅读时长）
  item_text_total: false   # 页面底部统计数量前是否显示文字描述（站点总字数、站点阅读时长）
  awl: 4                   # Average Word Length
  wpm: 275                 # Words Per Minute（每分钟阅读词数）
  suffix: mins.
~~~

&emsp;&emsp;上面是我的设置，没有要全站的统计，只保留了每篇文章的统计，可以根据自己的喜好来设置显示的类型。

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.qtygh4jpz7.png" alt="image" style="zoom:50%;" />

#### 数学公式

&emsp;&emsp;首先在站点根目录下下载插件

~~~shell
npm install hexo-math --save
~~~

&emsp;&emsp;在站点配置文件 *_config.yml* 中添加：

~~~shell _config.yml
math:
  engine: 'mathjax' # or 'katex'
  mathjax:
    # src: custom_mathjax_source
    config:
      # MathJax config
~~~

&emsp;&emsp;最后在每个页面首部添加公式开启标志，这里采取更改hexo文件生成模板的方式来进行，以后再写博文就不必进行专门的设置了。修改scaffolds文件夹中的post.md文件，在头部区域中添加代码：

~~~diff scaffolds/post.md
	title: {{ title }}
  date: {{ date }}
  tags:
  categories: 
  copyright: true
+ mathjax: true
~~~

#### 流程图

&emsp;&emsp;首先安装流程图模块

~~~shell
npm install hexo-filter-mermaid-diagrams
~~~

&emsp;&emsp;然后在博客根目录下的_config.yml文件中添加下面内容（已经进行过配置的可以跳过）：

~~~shell _config.yml
mermaid: ## mermaid url https://github.com/knsv/mermaid
  enable: true  # default true
  version: "7.1.2" # default v7.1.2
  options:  # find more api options from
  
~~~

&emsp;&emsp;这时mermaid流程图就已经可以正常显示了，但是流程图靠右显示，因此我们还需要进行居中设置。

{%note danger %}

hexo-next主题在更新到7.7之后更改了用户自定义布局的方式，以前是更改*_customs文件夹*中的header.swig内容进行自定义，而7.7以后更改为在主题配置文件themes/next/_config.yml指定自定义链接链接文件，链接文件直接放在根目录下的某个文件夹内，做到不需要直接更改主题布局文件。

{%endnote%}

&emsp;&emsp;首先在网站根目录下创建*source/_data/*文件夹然后创建文件styles.styl,在文件中写入下面内容:

~~~
/*mermaid圖居中*/
  2 .mermaid{
  3   text-align: center;
  4   max-height: 300px;
  5 }
~~~

&emsp;&emsp;然后打开主题配置文件*themes/next/_config.yml*

~~~diff
custom_file_path:
-	# style: source/_data/styles.styl
+ style: source/_data/styles.styl
~~~

&emsp;&emsp;流程图正常居中显示。

