---
title: NLP建模常见处理流程
date: 2019-05-09 10:59:27
tags: [机器学习,面试]
---





#### 1.清洗

​	主要包括**清除掉无关内容和区分出各个部分**。

> 段落的首尾单独区分：这里比较常见的一种却分时将段落的首尾单独区分出来，因为首尾句一般都是更加具有代表性的句子

#### 2.标准化

​	主要包含了**字母小写化**和**标点符号替换**两个步骤

~~~~python
#字母小写化
str.lower()

#标点符号替换为空格
import re
text = re.sub(r"a-zA-Z0-9"," ")
~~~~



#### 3.标记化(分词)

​	标记化是指将目标切分成无法再分符号，一般主要指分词，一般的处理中都会将句子按照" "进行分词。

~~~python
#自写原始分词，进行前面的标准化以后进行
words = text.split()

#使用nltk进行分词,分词会比上面的更加准确，根据标点符号的不同位置进行不同种处理，例如Dr. 中的.不会被处理掉
from nltk.tokenize import word_tokenize
sentence = word_tokenize(text)
words = word_tokenize(sentence)
#nltk提供了多种token方式，包括正则表达式等，按需选择
~~~



#### 4.删除停用词

​	删除停用词是指删除掉哪些去掉哪些和当前任务判断关系不大的词，对于设计到的语料没有具体领域时，可以使用英文常用停用词，其中包括800多个英文的常见停用词。

​	[英文常见停用词标准表](https://github.com/AnchoretY/Short_Text_Auto_Classfication/blob/master/english_stopword.txt)

> 在特定领域时，最好使用专门针对于该领域的停用词表，因为在一个问题中的停用词可能会在另一个问题中肯能就是关键词

~~~python
#去除停用词
def get_stopword(path):
  """
      获取停用词表
      return list
  """
  with open(path) as f:
    stopword = f.read()
  stopword_list = stopword.splitlines()

  return stopword_list

stopwords = get_stopword(path)
words = [word for word in words if word not in stopwords]
~~~



#### 5.词性标注

​		用于标注句子中各个单词分别属于什么词性，更加有助于理解句子的含义，另一方面，词性标注更加有利于后续处理。

> 常见的一种利用词性标注的后续处理步骤就是直接去掉非名词的部分，因为在一个句子中，名词在很大程度就可以表现两个句子的相似度。

~~~python
#使用nltk进行词性标注
from nltk import pos_tag
sentence = word_tokenize("this is a dog")   #分词
pos = pos_tag(sentence)   #标注
~~~



#### 6.命名实体识别

​		命名实体识别指的是识别

​		条件：命名实体识别首先要完成词性标注

​		应用：对新闻文章进行简历索引和搜索

> 实践性能并不是一直都很好，但对大型语料库进行实验确实有效

~~~python
from nltk import pos_tag,ne_chunk
from nltk.tokenize import word_tokenize

ne_chunk(pos_tag(word_tokenize("I live in Beijing University")))
~~~



#### 7.词干化和词型还原

​	**词干提取**是指将词还原成词干或词根的过程

​	方式：利用简单的搜索和替换样式**规则**，例如去除结尾的s、ing，将结尾的ies变为y等规则

​	作用：有助于降低复杂度，同时保留次所含的意义本质

> 还原的词干不一定非常准确，但是只要这个词的所有形式全部都转化成同一个词干就可以了，因为他们都有共同的含义



~~~python
from nltk.stem.porter import PorterStemmer
stemmed = [PoeterStemmer().stem(w) for w in words]

~~~

![](https://github.com/AnchoretY/images/blob/master/blog/词干提取.png?raw=true)



​	**词型还原**是将词还原成标准化形式的另一种技术，利用**字典的方式**将一个词的不同形式映射到其词根

​	方式：字典

​	优点:可以将较大的词型变化很大的正确还原到词根

~~~python
from nltk.stem.wordnet import WordNetLemmater

lemmed = [WordNetLemmater.lemmative(w) for w in words]
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/词型还原1.png?raw=true)

​	这里我们发现只有ones被还原成了one，其他词并没有找到词的原型，这是因为词型转化是针对词型进行的，只会转化指定词型的词，默认只转换名词，因此上面只有ones被转换了，下面我们来指定转换动词：

~~~
from nltk.stem.wordnet import WordNetLemmater

lemmed = [WordNetLemmater.lemmative(w) for w in words]
~~~

![](https://github.com/AnchoretY/images/blob/master/blog/词型转换2.png?raw=true)

#### 8.向量化

​	向量化是将提取好的token转化成向量表示，准备输入到模型中。常见的方式包括词袋模型、tf-idf、Word2vec、doc2vec等



#### 9. 分类模型或聚类模型

​	根据实际情况选用合适的分类模型，聚类模型。



注意:**上面的处理流程并不是全部都一定要进行,**可以根据实际情况进行选择,例如在下一篇文章情感分类中,只是使用了标准化、去停用词、词干提取、向量化、分类等步骤