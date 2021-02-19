---

title: 深度学习——transformer模型
date: 2019-02-28 10:49:15
tags: [机器学习,深度学习,NLP,面试]
---

​	transformer模型来自于Google的经典论文**Attention is all you need**，在这篇论文中作者采用Attention来取代了全部的RNN、CNN，实现效果效率的双丰收。

​	现在transformer在NLP领域已经可以达到全方位吊打CNN、RNN系列的网络，网络处理时间效率高，结果稳定性可靠性都比传统的CNN、RNN以及二者的联合网络更好，因此现在已经呈现出了transformer逐步取代二者的大趋势。

​	下面是三者在下面四个方面的对比试验结果

​		**1.远距离特征提取能力**

​		**2.语义特征提取能力**

​		**3.综合特征提取能力**

​		**4.特征提取效率**



![](https://github.com/AnchoretY/images/blob/master/blog/RNN%E3%80%81CNN%E3%80%81transformer%E9%95%BF%E8%B7%9D%E7%A6%BB%E7%89%B9%E5%BE%81%E6%8D%95%E8%8E%B7%E8%83%BD%E5%8A%9B%E5%AF%B9%E6%AF%94.png?raw=true)

![](https://github.com/AnchoretY/images/blob/master/blog/RNN%E3%80%81CNN%E3%80%81transformer%E8%AF%AD%E4%B9%89%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E8%83%BD%E5%8A%9B%E5%AF%B9%E6%AF%94.png?raw=true)



![](https://github.com/AnchoretY/images/blob/master/blog/RNN%E3%80%81CNN%E3%80%81Transformer%E7%BB%BC%E5%90%88%E7%89%B9%E5%BE%81%E6%8F%90%E5%8F%96%E8%83%BD%E5%8A%9B%E5%AF%B9%E6%AF%94.png?raw=true)

![](https://github.com/AnchoretY/images/blob/master/blog/RNN%E3%80%81CNN%E3%80%81Transformer%E4%B8%89%E8%80%85%E8%AE%A1%E7%AE%97%E6%95%88%E7%8E%87%E5%AF%B9%E6%AF%94.png?raw=true)



下面是从一系列的论文中获取到的RNN、CNN、Transformer三者的对比结论：	

​	**1.从任务综合效果方面来说，Transformer明显优于CNN，CNN略微优于RNN。**

​	**2.速度方面Transformer和CNN明显占优，RNN在这方面劣势非常明显。(主流经验上transformer和CNN速度差别不大，RNN比前两者慢3倍到几十倍)**



### Transformer模型具体细节

​	transformer模型整体结构上主要**Encoder**和**Decoder**两部分组成，Encoder主要用来将数据进行特征提取，而Decoder主要用来实现隐向量解码出新的向量表示(原文中就是新的语言表示)，由于原文是机器翻译问题，而我们要解决的问题是类文本分类问题，因此我们直接减Transformer模型中的Encoder部分来进行特征的提取。其中主要包括下面几个核心技术模块：

​		**1.残差连接**

​		**2.Position-wise前馈网络**

​		**3.多头self-attention**

​		**4.位置编码**



![](https://github.com/AnchoretY/images/blob/master/blog/transformer%E6%A8%A1%E5%9E%8B%E6%95%B4%E4%BD%93%E7%BB%93%E6%9E%84.png?raw=true=20*50)

​	1.采用全连接层进行Embedding （Batch_size,src_vocab_size,model_dim）

​	2.在进行位置编码，位置编码和Embedding的结果进行累加

​	3.进入Encoder_layer进行编码处理(相当于特征提取)

​		(1)

​		

#### 1.位置编码（PositionalEncoding）

​	大部分编码器一般都采用RNN系列模型来提取语义相关信息，但是采用RNN系列的模型来进行语序信息进行提取具有不可并行、提取效率慢等显著缺点，本文采用了一种 Positional Embedding方案来对于语序信息进行编码，主要通过正余弦函数，

​	![image-20190304162008847](https://github.com/AnchoretY/images/blob/master/blog/余弦位置编码.png?raw=true)

**在偶数位置，使用正弦编码;在奇数位置使用余弦进行编码。**

> 为什么要使用三角函数来进行为之编码？
>
> ​	首先在上面的公式中可以看出，给定词语的pos可以很简单其表示为dmodel维的向量，也就是说位置编码的每一个位置每一个维度对应了一个波长从2π到10000*2π的等比数列的正弦曲线，也就是说可以表示各个各个位置的**绝对位置**。
>
> ​	在另一方面，词语间的相对位置也是非常重要的，这也是选用正余弦函数做位置编码的最主要原因。因为
>
> ​	sin(α+β) = sinαcosβ+cosαsinβ
>
> ​	cos(α+β) = cosαcosβ+sinαsinβ
>
> ​	因此对于词汇间位置偏移k，PE(pos+k)可以表示为PE(pos)和PE(k)组合的形式，也就是**具有相对位置表达能力**

~~~python
class PositionalEncoding(nn.Module):
    
    """
        位置编码层
    """
    
    def __init__(self, d_model, max_seq_len):
        """
        初始化
        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()
       
        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
          [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
          for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])
        position_encoding = torch.Tensor(position_encoding)

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.cat((pad_row, position_encoding))
        
        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)
    def forward(self, input_len,max_len):
        """
            神经网络的前向传播。
        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。
          param max_len:数值，表示当前的词的长度

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """
        # 找出这一批序列的最大长度
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
          [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len.tolist()])
        
        return self.position_encoding(input_pos)
~~~



#### 2.scaled Dot-Product Attention

​	**scaled**代表着在原来的dot-product Attention的基础上加入了缩放因子1/sqrt(dk)，dk表示Key的维度，默认用64.

> 为什么要加入缩放因子？
>
> ​	在dk(key的维度)很大时，点积得到的结果维度很大，使的结果处于softmax函数梯度很小的区域，这是后乘以一个缩放因子，可以缓解这种情况的发生。

​	**Dot-Produc**代表乘性attention，attention计算主要分为加性attention和乘性attention两种。加性 Attention 对于输入的隐状态 ht 和输出的隐状态 st直接做 concat 操作，得到 [ht:st] ，乘性 Attention 则是对输入和输出做 dot 操作。

​	**Attention**又是什么呢？通俗的解释Attention机制就是通过query和key的相似度确定value的权重。论文中具体的Attention计算公式为：

![](https://github.com/AnchoretY/images/blob/master/blog/atttion%E8%AE%A1%E7%AE%97%E8%A1%A8%E8%BE%BE%E5%BC%8F.png?raw=true)

​	在这里采用的scaled Dot-Product Attention是self-attention的一种，self-attention是指Q(Query), K(Key), V(Value)三个矩阵均来自同一输入。就是下面来具体说一下K、Q、V具体含义：

> 1. 在一般的Attention模型中，Query代表要进行和其他各个位置的词做点乘运算来计算相关度的节点，Key代表Query亚进行查询的各个节点，每个Query都要遍历全部的K节点，计算点乘计算相关度，然后经过缩放和softmax进行归一化的到当前Query和各个Key的attention score，然后再使用这些attention score与Value相乘得到attention加权向量
> 2. 在self-attention模型中，Key、Query、Value均来自相同的输入
> 3. 在transformer的encoder中的Key、Query、Value都来自encoder上一层的输入，对于第一层encoder layer，他们就是word embedding的输出和positial encoder的加和。

![](https://github.com/AnchoretY/images/blob/master/blog/scaled%20dot-product%20attention.png?raw=true)

> query、key、value来源：
>
> ​	他们三个是由原始的词向量X乘以三个权值不同的嵌入向量Wq、Wk、Wv得到的，三个矩阵尺寸相同
>
> **Attention计算步骤：**
>
> 1. 如上文，将输入单词转化成嵌入向量；
> 2. 根据嵌入向量得到 q、k、v三个向量；
> 3. 为每个向量计算一个score： score = q*k
> 4. 为了梯度的稳定，Transformer使用了score归一化，即除以 sqrt(dk)；
> 5. 对score施以softmax激活函数；
> 6. softmax点乘Value值 v ，得到加权的每个输入向量的评分 v；
> 7. 相加之后得到最终的输出结果Sum(z) ：  。

~~~python
class ScaledDotProductAttention(nn.Module):
    """
        标准的scaled点乘attention层
    """
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)
        
    def forward(self, q, k, v, scale=None, attn_mask=None):
        """
        前向传播.
        Args:
        	q: Queries张量，形状为[B, L_q, D_q]
        	k: Keys张量，形状为[B, L_k, D_k]
        	v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
        	scale: 缩放因子，一个浮点标量
        	attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
        	上下文张量和attention张量
        """
        attention = torch.bmm(q, k.transpose(1, 2))
        
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            # 给需要 mask 的地方设置一个负无穷
            attention = attention.masked_fill(attn_mask,-1e9)
        
        # 计算softmax
        attention = self.softmax(attention)
        # 添加dropout
        attention = self.dropout(attention)
        # 和V做点积
        context = torch.bmm(attention, v)

        return context, attention
~~~



#### 3.多头Attention

​	论文作者发现**将 Q、K、V 通过一个线性映射之后，分成 h 份，对每一份进行 scaled dot-product attention** 效果更好。**然后，把各个部分的结果合并起来，再次经过线性映射，得到最终的输出**。这就是所谓的 multi-head attention。上面的超参数 h 就是 heads 的数量。论文默认是 8。

​	这里采用了四个全连接层+有个dot_product_attention层(也可以说是8个)+layer_norm实现。

>为什么要使用多头Attention？
>
>​	1.”多头机制“能让模型考虑到不同位置的Attention
>
>​	2.”多头“Attention可以在不同的足空间表达不一样的关联

~~~python
class MultiHeadAttention(nn.Module):
    """
        多头Attention层
    """

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads

        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        
        # 残差连接
        residual = query
        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)
        # 线性层 (batch_size,word_nums,model_dim)
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # 将一个切分成多个(batch_size*num_headers,word_nums,word//num_headers)
        """
            这里用到了一个trick就是将key、value、query值要进行切分不需要进行真正的切分，直接将其维度整合到batch_size上，效果等同于真正的切分。过完scaled dot-product attention 再将其维度恢复即可
        """       
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)
        #将mask也复制多份和key、value、query相匹配  （batch_size*num_headers,word_nums_k,word_nums_q）
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)

        # 使用scaled-dot attention来进行向量表达
        #context:(batch_size*num_headers,word_nums,word//num_headers)
        #attention:(batch_size*num_headers,word_nums_k,word_nums_q)
        scale = (key.size(-1)) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, scale, attn_mask)
        
        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # 这里使用了残差连接和LN
        output = self.layer_norm(residual + output)

        return output, attention
~~~





#### 4.残差连接

​	在上面的多头的Attnetion中，还采用了残差连接机制来保证网络深度过深从而导致的精度下降问题。这里的思想主要来源于深度残差网络(ResNet)，残差连接指在模型通过一层将结果输入到下一层时也同时直接将不通过该层的结果一同输入到下一层，从而达到解决网络深度过深时出现精确率不升反降的情况。

![](https://github.com/AnchoretY/images/blob/master/blog/res-net.png?raw=true)

> **为什么残差连接可以在网络很深的时候防止出现加深深度而精确率下降的情况？**
>
> ​	神经网络随着深度的加深会会出现训练集loss逐渐下贱，趋于饱和，然后你再加深网络深度，训练集loss不降反升的情况。这是因为随着网络深度的增加，在深层的有效信息可能变得更加模糊，不利于最终的决策网络做出正确的决策，因此残差网络提出，建立残差连接的方式来将低层的信息也能传到高层，因此这样实现的深层网络至少不会比浅层网络差。



#### 5.Layer normalization

##### Normalization 

​	Normalization 有很多种，但是它们都有一个**共同的目的，那就是把输入转化成均值为 0 方差为 1 的数据**。我们在把数据送入激活函数之前进行 normalization（归一化），**因为我们不希望输入数据落在激活函数的饱和区。**

#####Batch Normalization(BN)

​	应用最广泛的Normalization就是Batch Normalization，其主要思想是:**在每一层的每一批数据上进行归一化**。我们可能会对输入数据进行归一化，但是经过该网络层的作用后，我们的数据已经不再是归一化的了。**随着这种情况的发展，数据的偏差越来越大，我的反向传播需要考虑到这些大的偏差，这就迫使我们只能使用较小的学习率来防止梯度消失或者梯度爆炸。**

![](https://github.com/AnchoretY/images/blob/master/blog/Batch%20normalization%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.png?raw=true)

##### Layer normalization(LN)

​	LN 是**在每一个样本上计算均值和方差，而不是 BN 那种在批方向计算均值和方差**.

>Layer normalization在pytorch 0.4版本以后可以直接使用nn.LayerNorm进行调用

![](https://github.com/AnchoretY/images/blob/master/blog/Batch%20normalization%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.png?raw=true)



#### 6.Mask

​	**mask 表示掩码，它对某些值进行掩盖，使其在参数更新时不产生效果**。Transformer 模型里面涉及两种 mask，分别是 **padding mask** 和 **sequence mask**。

​	在我们使用的Encoder部分，只是用了padding mask因此本文重点介绍padding mask。 

##### padding mask

​	什么是 padding mask 呢？因为每个批次输入序列长度是不一样的也就是说，我们要对输入序列进行对齐。具体来说，就是给**在较短的序列后面填充 0。因为这些填充的位置，其实是没什么意义的，所以我们的 attention 机制不应该把注意力放在这些位置上**，所以我们需要进行一些处理。**具体的做法是，把这些位置的值加上一个非常大的负数(负无穷)，这样的话，经过 softmax，这些位置的概率就会接近0！**而我们的 padding mask 实际上是一个张量，每个值都是一个 Boolean，值为 false 的地方就是我们要进行处理的地方。

~~~python
def padding_mask(seq_k, seq_q):
    """        
        param seq_q:(batch_size,word_nums_q)
        param seq_k:(batch_size,word_nums_k)
        return padding_mask:(batch_size,word_nums_q,word_nums_k)
    """
    
    # seq_k和seq_q 的形状都是 (batch_size,word_nums_k)
    len_q = seq_q.size(1)
    # 找到被pad填充为0的位置(batch_size,word_nums_k)
    pad_mask = seq_k.eq(0)
    #(batch_size,word_nums_q,word_nums_k)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    
    return pad_mask
~~~



#### 3.Position-wise 前馈网络

​	这是一个全连接网络，包含两个线性变换和一个非线性函数(实际上就是 ReLU)

​	![](https://github.com/AnchoretY/images/blob/master/blog/Position-wise%20Feed-Forward%20network%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F.png?raw=true)

**这里实现上用到了两个一维卷积。**

~~~
class PositionalWiseFeedForward(nn.Module):
    """
        前向编码，使用两层一维卷积层实现
    """

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(F.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output
~~~

