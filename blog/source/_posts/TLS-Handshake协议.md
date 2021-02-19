---
title: TLS-Handshake协议
copyright: true
mathjax: true
date: 2021-01-12 20:47:42
tags:
categories:
---

概述：首页描述

![]()

<!--more-->

### 整体通信过程

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.lqonfcv1txg.png)

#### 1. stage 1

&emsp;&emsp;客户端发送Client向服务端通知客户端支持的加密套件情况，发起TLS连接。

#### 2. Stage 2

&emsp;&emsp;服务端向客户端表明其证书以及加密通信的参数。首先通过Server Hello表明服务端选用的加密算法，然后使用Certificate向客户端发送服务端的证书、Server Key Exchange表明ECDiffie-Hellman相关的加密参数，最后使用Server Hello Done表明服务端Hello信息发送完成。

#### 3. Stage3

&emsp;&emsp;客户端向服务端发送ECDiffie-Hellman相关的参数，并通知服务端开始使用加密数据进行通信。首先使用Client Key Exchange发送客户端的ECDiffie-Hellman相关参数值，然后使用Change Cipher Spec通知服务端开始使用加密数据进行通信，最后Finished表明TLS握手客户端部分完成。

#### 4. Stage 4

&emsp;&emsp;服务端通知客户开始使用加密数据进行数据通信，完成TLS握手服务端部分。



###  客户端与服务器处理过程

&emsp;&emsp;上面部分讲解了TLS在网络通信上的整体流程，那么在进行网络通信的过程中客户端与服务器分别做了哪些操作呢？下图很好的表示二者的在TLS通信过程中全部处理过程。

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.lh7dg7hje6k.png)

#### 1. Client deal1

&emsp;&emsp;在客户端发起TLS连接请求之前生成Client Random、Cipher Suites、Extensions等，在Stage1会发送给服务端。

#### 2. Sever deal1

1. 在服务端提供的Cipher Suites、Extensions中选择服务端支持的
2. 校验SessionId是否已经存在，存在则使用已经存在连接，不再继续进行TLS握手
3. 生成Server Random
4. 生成Premaster secret服务端参数

#### 3. Client deal2

1. 根据客户端提供的加密参数和自身加密参数计算Premaster secret（RSA与DH交换的参数不同）
2. 使用Premaster secret、Server Random、Client Random生成master key
3. 使用master key采用对称加密算法对之前全部的全部的握手消息计算HMac

#### 4. Sever deal2

1. 根据服务端提供的加密参数和自身加密参数计算Premaster secret（RSA与DH交换的参数不同）
2. 使用Premaster secret、Server Random、Client Random生成master key
3. 使用master key对之前的握手消息（不包含stage3消息）进行加密，与finished消息内容进行对比，验证消息的正确性以确认客户端身份。
4. 使用master key采用对称加密算法对之前全部的握手消息（包含stage3消息）计算HMac，发送给客户端。

#### 5. Client deal2

&emsp;&emsp;使用master key对之前的握手消息（不包含stage3消息）进行加密，与finished消息内容进行对比，验证消息的正确性，以验证服务端身份。



### 关键问题

#### 1. 密钥交换方式

&emsp;&emsp;秘钥交换算法主要涉及到Premaster secret生成的方式，以及对称秘钥生成过程中相关参数的交换。计算方式主要有RSA和Diffe-Hellan两种。

- **RSA**：客户端使用2Bytes的协议版本号和46 Bytes的随机数组合生成Premaster secert，生成后使用Server 证书中的公钥通过Client Key Exchange发送给Server。Premaster secert生成过程不需要任何Server端加密参数，因此使用RSA加密套件时Server端不需要发送Server Key Exchange消息。

  <img src="https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.n47h006c4p.png" alt="image" style="zoom:60%;" />

- **Diffe-Hellan**：双方通过Server Exchange secert和Client Exchange secert交换DH算法计算对称秘钥的参数，各自对方发送的参数以及自己生成随机大数使用加密算法生成Premaster secert。

  {% note  warning %}
  这里使用的随机大并不是Client Hello和Server Hello部分的Random，而是专用于该算法生成的随机大数。

  {% endnote %}

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.5myigboujsr.png)

{% note  default %}
**Premaster secert是TLS中通信能能否破解最为关键的环节**。

{% endnote %}

#### 2. 会话秘钥的生成

&emsp;&emsp;会话秘钥的生成使用之前握手过程中获得的Server Random、Client Random、Premaster secert计算得出，用于将之前所有握手消息采用会话秘钥加密，然后进行HMAC计算，最后使用Finished消息发送给服务端，**以验证密钥交换和身份验证过程是否成功。**

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.najd2pi1lbh.png)

#### 3. Premaster Secret、master secret、证书的作用

- **证书**: 验证服务器身份，确认Server公钥的正确性

- **Premaster secret**：生成Master Secret

- **Master Secret** : 加密通信





### 通信过程

#### 1.Client Hello

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.4twfklm0r3a.png)

&emsp;&emsp;客户端发起握手请求，向服务器发送Client Hello消息，消息中主要包含：

- **支持的TLS/SSL版本**

- **Cipher Suites加密算法列表**：告知Server端Client支持的加密套件都有哪些，用于服务端加密套件选择。

- **SessionID**：用于恢复会话。如果客户在几秒钟之前登陆过这个服务器，就可以直接使用SessionID值恢复之前的会话，而不再需要一个完整的握手过程。

- **Random（Server）**：为后面生成会话秘钥做准备

- **Extension(Client)**:客户端使用的拓展

  &emsp;&emsp;这里最常用的客户端拓展就是Server Name Idication Extension,简称**SNI**，其中指明server name**表明客户端想要请求进行通信的网站**，一般是一个域名。

> 注：Cipher Suite格式
>
> （1）秘钥交换算法: 秘钥交换以及计算的方式，主要影响Server Key Exchange、Client Key Exchange阶段传输参数的内容以及传输的方式。可选包括：RSA, DH, ECDH, ECDHE
>
> （2）加密算法：对称加密算法，
>
> （3）报文认证信息码（MAC）算法：用于创建报文摘要，确保报文完整性，常见包括MD5、SHA等
>
> （4）PRF（伪随机数函数）：用于生成“Master secret”
>
> ![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.w4mjc1lawsc.png)
>
> ​	WITH是一个分隔单次，**WITH前面的表示的是握手过程所使用的非对称加密方法**，**WITH后面的表示的是加密信道的对称加密方法和用于数据完整性检查的哈希方法**。**WITH前面通常有两个单词，第一个单次是约定密钥交换的协议，第二个单次是约定证书的验证算法**。要区别这两个域，必须要首先明白，两个节点之间交换信息和证书本身是两个不同的独立的功能。两个功能都需要使用非对称加密算法。交换信息使用的非对称加密算法是第一个单词，证书使用的非对称加密算法是第二个。有的证书套件，例如TLS_RSA_WITH_AES_256_CBC_SHA，**WITH单词前面只有一个RSA单词，这时就表示交换算法和证书算法都是使用的RSA**，所以只指定一次即可。可选的主要的密钥交换算法包括: RSA, DH, ECDH, ECDHE。可选的主要的证书算法包括：RSA, DSA, ECDSA。两者可以独立选择，并不冲突。AES_256_CBC指的是AES这种对称加密算法的256位算法的CBC模式，AES本身是一类对称加密算法的统称，实际的使用时要指定位数和计算模式，CBC就是一种基于块的计算模式。最后一个SHA就是代码计算一个消息完整性的哈希算法。

​							

#### 2. Server Hello

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.91lsazkj1eq.png)

&emsp;&emsp;服务端根据客户端支持发送的Hello信息回复，选择秘钥并确认是否存在已有会话、并提供Server端Random。消息中包含：

- **选择的Cipher Suite套件**：Server端根据自身情况在Client端提供的Cipher Suites中选择一个作为二者后续进行加密通信要使用的加密套件。
- **SessionID**：如果服务端保存有二者之间的SessionID，那么返回SessionID，不用在进行后续的握手，直接使用先前会话的证书、秘钥等进行通信。
- **Random(Server)**：服务端随机数，为后面生成会话秘钥做准备。
- **Extension(Server)**: 服务端使用的拓展



#### 3.Certificate

&emsp;&emsp;服务端发送服务端证书给客户端，**服务端证书主要用于用于确认服务端身份，使Clinet确认服务端公钥**。

#### （1）证书结构

&emsp;&emsp;数字证书由CA（Certificate Authority）机构进行签发，关键内容包括：

- **证书颁发者（issuer）**:

- **证书持有者（Subject）**:

- **证书有效期:**

- **证书持有者公钥：**

- **证书持有者域名（DN）**：

- 证书颁发者的数字签名:已签名的数字证书采用，未签名的数字证书只有上面的内容。

  

#### （2）数字签名

&emsp;&emsp; 证书的签发过程通俗的说**就是数字签名证书签发机构对证书进行数字签名的过程**。数字签名包括两个过程：**签发证书（Signing）** 和 **验证证书（Verification）**

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.2r77bwtklam.png)



#### （3）证书签发与验证

##### 签发证书的过程

1. 撰写证书元数据：包括 证书结构中**除数字签名以外的全部数据作为元数据**，即未签名证书，进行数字签名。
2. 使用通用的 Hash 算法（如SHA-256）对证书元数据计算生成 **数字摘要**
3. 使用 Issuer 的私钥对该数字摘要进行加密，生成一个加密的数字摘要，也就是Issuer的 **数字签名**
4. 将数字签名附加到数字证书上，变成一个 **签过名的数字证书**
5. 将签过名的数字证书与 **Issuer 的公钥**，一同发给证书使用者（注意，将公钥主动发给使用者是一个形象的说法，只是为了表达使用者最终获取到了 Issuer 的公钥）

##### 验证证书的过程

1. 证书使用者获通过某种途径（如浏览器访问）获取到该数字证书，解压后分别获得 **证书元数据** 和 **数字签名**
2. 使用同样的Hash算法计算证书元数据的 **数字摘要**
3. 使用 **Issuer 的公钥** 对数字签名进行解密，得到 **解密后的数字摘要**
4. 对比 2 和 3 两个步骤得到的数字摘要值，如果相同，则说明这个数字证书确实是被 Issuer 验证过合法证书，证书中的信息（最主要的是 Owner 的公钥）是可信的

{% note  default %}
这里我们可以注意到证书签发者公钥和证书拥有者公钥具有完全不同的作用。

- 证书签发者公钥用于验证证书是否真的由证书签发机构签发。
- 证书拥有者公钥包含在证书元数据中进行数字签名，确保公钥为持有着所有，后续用于加密通信。

{% endnote %}

#### （4）证书链

&emsp;&emsp;从上面的例子中可以看出，“签发证书”与“验证证书”两个过程，Issuer（CA）使用 **Issuer 的私钥** 对签发的证书进行数字签名，证书使用者使用 **Issuser 的公钥** 对证书进行校验，如果校验通过，说明该证书可信。由此看出，**校验的关键**是 **Issuer 的公钥**，使用者获取不到 Issuer 的私钥，只能获取到 Issuer 的公钥，如果 Issuer 是一个坏家伙，谁来证明 **Issuer 的身份** 是可信的这就**需要靠证书链来进行保证Issuer身份的可信**。

&emsp;&emsp;还是以百度为例，在浏览器上访问 “[www.baidu.com](http://www.baidu.com/)” 域名，地址连左侧有一个小锁的标志，点击就能查看百度的数字证书，如下图所示（使用的是Edge浏览器）

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.6m0qdq12ql.png)

&emsp;&emsp;在图片的顶部，我们看到这样一个层次关系：

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;GlobalSign Root CA -> GlobalSign Organization Validation CA -> baidu.com

&emsp;&emsp;这个层次可以抽象为三个级别：

1. end-user：即 baidu.com，该证书包含百度的公钥，访问者就是使用该公钥将数据加密后再传输给百度，即在 HTTPS 中使用的证书
2. intermediates：即上文提到的 **签发人 Issuer**，用来认证公钥持有者身份的证书，负责确认 HTTPS 使用的 end-user 证书确实是来源于百度。这类 intermediates 证书可以有很多级，也就是说 **签发人 Issuer 可能会有有很多级**
3. root：可以理解为 **最高级别的签发人 Issuer**，负责认证 intermediates 身份的合法性

&emsp;&emsp;这其实代表了一个信任链条，**最终的目的就是为了保证 end-user 证书是可信的，该证书的公钥也就是可信的。**



![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.pl7h8zuiuva.png)

结合实际的使用场景对证书链进行一个归纳：

1. 为了获取 end-user 的公钥，需要获取 end-user 的证书，因为公钥就保存在该证书中
2. 为了证明获取到的 end-user 证书是可信的，就要看该证书是否被 intermediate 权威机构认证，等价于是否有权威机构的数字签名
3. 有了权威机构的数字签名，而权威机构就是可信的吗？需要继续往上验证，即查看是否存在上一级权威认证机构的数字签名
4. 信任链条的最终是Root CA，他采用自签名，对他的签名只能无条件的信任

{% note  default %}
Root CA浏览器中已经内置，直接信任，这就使为什么有些网络会被HTTPS认证绿锁

{% endnote %}



#### 4.Server Key Exchange

&emsp;&emsp;该消息主要用于发送Server端 ECDiffie-Hellman等加密算法相关参数。加密算法由Server Hello报文进行选择，当选择的报文为DHE、DH_ano等加密算法组时才会有Server Key Exchange报文。

**Server端加密算法相关参数作用**：发送给Client，Client根据对方参数和自身参数计算出Premaster srcert。

#### 5. Server Hello Done

&emsp;&emsp;Server端向Client发送Server Hello Done消息，表明服务端握手已经加送完成。



{% note  default %}
Certificate、Server Key Exchange、Server Hello Done三个消息经常使用一个报文进行发送。

{% endnote %}

##### 6. Client Key Exchange

&emsp;&emsp;客户端收到Server端发来的证书，进行证书验证，确认证书可信后，会向Server端发送Client Key Exchange消息，其中包含了Premaster秘钥相关的信息。

> Client key Exchange是无论使用什么秘钥交换算法都需要发送的消息。
>
> &emsp;&emsp;RSA：使用Client公钥加密后的Premaster secert秘钥。
>
> &emsp;&emsp;DH:  Pa



#### 7. Change Cipher Spec

&emsp;&emsp;客户端发送Change Cipher Spec消息来通知Server端开始使用加密的方式来进行通信。



#### 8.Finished

&emsp;&emsp;客户端使用之前握手过程中获得的Server Random、Client Random、Premaster secert计算master secert(会话秘钥)，然后使用会话秘钥采用加密算法使用master secret对（对称算法，加密套件中的第二部分）之前所有握手消息进行HMAC计算，然后使用Finished消息发送给服务端，**用于验证密钥交换和身份验证过程是否成功。**

![image](https://raw.githubusercontent.com/AnchoretY/images/master/blog/image.najd2pi1lbh.png)



#### 9. Change Cipher Spec

&emsp;&emsp;服务端收到客户端加密的的Finished消息后，服务器采用完全相同的方法来计算Hash和MAC值，相同则认为身份验证成功，Server端接受这个Master secret作为后续通信的秘钥，然后使用Change Cipher Spec来通知Client端开始使用加密的方式来进行通信。



#### 10. Finished

&emsp;&emsp;服务端使用与客户端完全相同的方式对以往全部信息的进行MAC和Hash运算，然后将其使用使用master secret加密后发给客户端，客户端验证成功后则认为会话秘钥协商成功，后续开始正式加密通信。



{% note  default %}
服务端发进行Hash和Mac运算的消息比Client端进行Hash和Mac运算的消息要多出客户端的 Change Cipher Spec和Finished这两个消息。

{% endnote %}













##### 参考文献

- https://www.jianshu.com/p/fcd0572c4765
- https://halfrost.com/https-key-cipher/