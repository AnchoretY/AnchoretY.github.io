---
title: hexo博客打造评论系统
date: 2020-03-16 16:13:49
tags: [博客]
---

&emsp;&emsp;本文主要讲述使用gitalk和Valine两种方式打造hexo博客评论系统的完整过程，并讲述两种评论系统的使用感受，本人使用它的主题为next，其他主题配置可能略有不同，如果在配置过程中产生任何问题，欢迎留言交流。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.cuoo5vg2tp.png)

<!-- more -->

### 使用gitalk打造评论系统

#### 1. Github注册OAuth应用

&emsp;&emsp;在GitHub上注册新应用，链接：https://github.com/settings/applications/new

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.ue332ss57b.png" alt="image" style="zoom: 40%;" />

**参数说明**：

- Application name： # 应用名称，随意

- Homepage URL： # 网站URL

- Application description # 描述，随意

- Authorization callback URL：# 网站URL

&emsp;&emsp;点击注册后，页面跳转如下，其中Client ID和Client Secret在后面的配置中需要用到，到时复制粘贴即可：

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.ywhey2o6cdd.png" alt="image" style="zoom:30%;" />

#### 2. 新建`gitalk.swig`

&emsp;&emsp;新建`themes/next/layout/_third-party/comments/gitalk.swig`文件，并添加内容：

```shell :themes/next/layout/_third-party/comments/gitalk.swig
{% if page.comments && theme.gitalk.enable %}
  <link rel="stylesheet" href="https://unpkg.com/gitalk/dist/gitalk.css">
  <script src="https://unpkg.com/gitalk/dist/gitalk.min.js"></script>
  <script src="/js/src/md5.min.js"></script>
   <script type="text/javascript">
        var gitalk = new Gitalk({
          clientID: '{{ theme.gitalk.ClientID }}',
          clientSecret: '{{ theme.gitalk.ClientSecret }}',
          repo: '{{ theme.gitalk.repo }}',
          owner: '{{ theme.gitalk.githubID }}',
          admin: ['{{ theme.gitalk.adminUser }}'],
          id: md5(location.pathname), # 使用md5确保id长度不超过50，关键
          distractionFreeMode: '{{ theme.gitalk.distractionFreeMode }}'
        })
        gitalk.render('gitalk-container')
    </script>
{% endif %}
```

&emsp;&emsp;为了配合其中的md5函数能够使用，这里要引入js脚本。将 [md5.min.js](https://github.com/blueimp/JavaScript-MD5/blob/master/js/md5.min.js) 文件下载下来放到 themes/next/source/js/src/ 路径下。

#### 3. 修改`comments.swig`

&emsp;&emsp;修改`themes/next/layout/_partials/comments.swig`，添加内容如下，与前面的`elseif`同一级别上：

```
{% elseif theme.gitalk.enable %}
 <div id="gitalk-container"></div>
```

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.f6pw9frdqo.png" alt="image" style="zoom:50%;" /> 

#### 4. 修改`index.swig`

&emsp;&emsp;修改`themes/next/layout/_third-party/comments/index.swig`，在最后一行添加内容：

```
{% include 'gitalk.swig' %}
```

#### 5. 新建`gitalk.styl`

&emsp;&emsp;新建`/source/css/_common/components/third-party/gitalk.styl`文件，添加内容：

```
.gt-header a, .gt-comments a, .gt-popup a
  border-bottom: none;
.gt-container .gt-popup .gt-action.is--active:before
  top: 0.7em;
```

#### 6. 修改`third-party.styl`

&emsp;&emsp;修改`themes/next/source/css/_common/components/third-party/third-party.styl`，在最后一行上添加内容，引入样式：

```
@import "gitalk";
```

#### 7. 修改`_config.yml`

&emsp;&emsp;在主题配置文件`themes/next/next/_config.yml`中添加如下内容：

```yml themes/next/next/_config.yml
gitalk:
  enable: true
  githubID: github帐号
  repo: 仓库名称
  ClientID: Client ID
  ClientSecret: Client Secret
  adminUser: github帐号 #指定可初始化评论账户
  distractionFreeMode: true
```

#### 8.修改`index.md`文件使about、tag、categories页面不显示评论页面

&emsp;&emsp;上面的操作步骤完成后，tag的页面显示如下,about和categories页面类似:

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.wv7w6b40hv.png" alt="image" style="zoom:30%;" />

​	

&emsp;&emsp;这里只需要在`source/`文件夹下分别新建 `about`、`categories`、`tags` **文件夹**，每个文件夹里面都新建一个 `index.md` 文件（已有的直接添加comment: false），内容为

~~~yml
---
title:  # 标题
type: "about"  # about、categories、tags
comments: false
---
~~~

&emsp;&emsp;就可以正常显示了

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.56bh8f8f7yb.png" alt="image" style="zoom:33%;" /> 





分类页面

最终呈现效果：

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.ysh5txdldwd.png" alt="image" style="zoom:40%;" />

博主使用github账号登录后其他用户可以正常使用评论

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.l9heaiesv2k.png" alt="image" style="zoom:40%;" />

部署

#### 9.文章评论自动初始化

&emsp;&emsp;到上面一步虽然评论系统已经可以使用，但是每篇文章都需要博主去手工进行初始化，非常的繁琐，因此在这里提供一个自动化进行评论初始化的脚本，可实现直接对全部脚本的初始化。

&emsp;&emsp;要使用该脚本首先要在github上穿件一种新的认证方式——[Personal access token](https://github.com/settings/tokens)，点击Generate New token，如下填写

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.ghiqrh9mogv.png" alt="image" style="zoom:30%;" />

&emsp;&emsp;保存Token:

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.kl7xe4s2p5m.png" alt="image" style="zoom:33%;" />

&emsp;&emsp;安装依赖：

~~~shell
sudo gem install faraday activesupport sitemap-parser
~~~

&emsp;&emsp;在根目录下新建gitalk_inti.rb,内容如下：

~~~ruby /gitalk.rb
username = "xxx" # GitHub 用户名
token = "xxx"  # GitHub Token
repo_name = "xxx.github.io" # 存放 issues
sitemap_url = "https://anchorety.github.io/sitemap.xml" # sitemap
kind = "gitalk" # "Gitalk" or "gitment"

require 'open-uri'
require 'faraday'
require 'active_support'
require 'active_support/core_ext'
require 'sitemap-parser'
require 'digest'

puts "正在检索URL"

sitemap = SitemapParser.new sitemap_url
urls = sitemap.to_a

puts "检索到文章共#{urls.count}个"

conn = Faraday.new(:url => "https://api.github.com") do |conn|
  conn.basic_auth(username, token)
  conn.headers['Accept'] = "application/vnd.github.symmetra-preview+json"
  conn.adapter  Faraday.default_adapter
end

commenteds = Array.new
`
  if [ ! -f .commenteds ]; then
    touch .commenteds
  fi
`
File.open(".commenteds", "r") do |file|
  file.each_line do |line|
      commenteds.push line
  end
end

urls.each_with_index do |url, index|
  url.gsub!(/index.html$/, "")

  if commenteds.include?("#{url}\n") == false
    url_key = Digest::MD5.hexdigest(URI.parse(url).path)
    response = conn.get "/search/issues?q=label:#{url_key}+state:open+repo:#{username}/#{repo_name}"

    if JSON.parse(response.body)['total_count'] > 0
      `echo #{url} >> .commenteds`
    else
      puts "正在创建: #{url}"
      title = open(url).read.scan(/<title>(.*?)<\/title>/).first.first.force_encoding('UTF-8')
      response = conn.post("/repos/#{username}/#{repo_name}/issues") do |req|
        req.body = { body: url, labels: [kind, url_key], title: title }.to_json
      end
      if JSON.parse(response.body)['number'] > 0
        `echo #{url} >> .commenteds`
        puts "\t↳ 已创建成功"
      else
        puts "\t↳ #{response.body}"
      end
    end
  end
end
~~~

&emsp;&emsp;最后可直接输入：

~~~shell
ruby gitalk_init.rb
~~~

{% note  success %}

网上大量自动初始化脚本多次提交会出现多次创建issue问题，该脚本不存在多提交多次创建issue的问题，可以放心使用。

{% endnote %}





### 使用Valine打造评论系统

&emsp;&emsp;Valine 是基于 LeanCloud 作为数据存储的，是一款商用的软件，与gitalk相比具有以下优势：

> 1. 稳定性更好
> 2. 不存在初始化问题
> 3. 自带了文章阅读量显示功能
> 4. 具备更加方便的邮箱提醒功能

下面将对使用Valine打造评论系统的详细过程进行介绍：

#### 1.账号注册

&emsp;&emsp;因为是商用软件，因此使用Valine首先要进行的就是登陆[LeanCloud网站](https://www.leancloud.cn)进行账号注册。

{% note  info %}
因为网络安全法的出台，现在LeanCloud账号注册必选要进行实名认证后才能正常使用
{% endnote %}

#### 2.创建应用

&emsp;&emsp;点击创建应用，输入应用名称，选择开发版进行创建。

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.gbbp2qov7xo.png" alt="image" style="zoom:40%;" />

&emsp;&emsp;创建成功后，出现界面

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.7ju3bv7h4c.png" alt="image" style="zoom:50%;" />

#### 3. 打开应用存储，创建存储表

&emsp;&emsp;打开应用，选择左边的存储，进入后点击创建Class，分别创建Counter和Comment两个Class，创建时全向都选择无任何限制，面向全部用户。

{% note  info %}

Counter为文章阅读计数存储表，Comment为评论存储表。

{% endnote %}

&emsp;&emsp;创建成功后如下：

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.70shbacm58k.png" alt="image" style="zoom:50%;" />

#### 4.关闭其他服务，设置Web安全域名

&emsp;&emsp;点击左侧导航栏设置选项，点击安全中心，关闭除数据存储外其他选型，然后在Web安全域名中填写博客地址。

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.9r7z0jskdk.png" alt="image" style="zoom:30%;" />

#### 5.点击应用key，获取应用的AppID、AppKey

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.4mxnnxu5xlj.png" alt="image" style="zoom:30%;" />

#### 6.修改主题配置文件进行配置

&emsp;&emsp;打开主题目录下_config.yml文件搜索valine对其进行配置。

~~~diff themes/next/_config.yml
valine:
  enable: true   
  appid: xxx #上面保存的APPID
  appkey: xxx #上面保存的AppKey
  visitor: true #阅读统计
  guest_info: nick,mail #评论用户信息设置
-  language: zh-CN
+  language: zh-cn
~~~

{% note  danger %}

这里language只能是小写，大写会造成评论区域不显示

{% endnote %}

&emsp;&emsp;博客重新部署后，就可以正常使用Valine评论了。

<img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.1x2lbwouzha.png" alt="image" style="zoom:33%;" />







##### 参考文献

- https://yuanmomo.net/2019/06/20/hexo-add-valine/





