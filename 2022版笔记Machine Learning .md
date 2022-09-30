<a name="QpRCj"></a>
# 引言(Introduction)
<a name="rRJTy"></a>
## 机器学习是什么？

机器学习是什么？实际上，即使是在机器学习的专业人士中，也不存在一个被广泛认可的定义来准确定义机器学习是什么或不是什么，现在我将告诉你一些人们尝试定义的示例。

第一个机器学习的定义来自于Arthur Samuel。他定义机器学习为，在进行特定编程的情况下，给予计算机学习能力的领域。Samuel的定义可以回溯到50年代，他编写了一个西洋棋程序。这程序神奇之处在于，编程者自己并不是个下棋高手。但因为他太菜了，于是就通过编程，让西洋棋程序自己跟自己下了上万盘棋。通过观察哪种布局（棋盘位置）会赢，哪种布局会输，久而久之，这西洋棋程序明白了什么是好的布局，什么样是坏的布局。然后就牛逼大发了，程序通过学习后，玩西洋棋的水平超过了Samuel。这绝对是令人注目的成果。

另一个年代近一点的定义，由Tom Mitchell提出，来自卡内基梅隆大学，Tom定义的机器学习是，一个好的学习问题定义如下，他说：一个程序被认为能从经验E中学习，解决任务T，达到性能度量值P，当且仅当有了经验E后，经过P评判，程序在处理T时的性能有所提升。我认为经验E 就是程序上万次的自我练习的经验而任务T 就是下棋。性能度量值P呢，就是它在与一些新的对手比赛时，赢得比赛的概率。

---

<a name="anUeE"></a>
## 监督学习

:::info
定义：监督学习 (supervised learning) 是指从标注数据中学习预测模型的机器学习问题。标注数据表示输入输出的对应关系，预测模型对给定的输入产生相应的输出。监 督学习的本质是学习输入到输出的映射的统计规律。
:::

:::info
监督学习的目的在于学习某个由输入到输出的映射，这一映射由模型来表示。换句话说，学习的目的就在于找到最好的这样的模型。模型属于由输入空间到输出空间的映射的集合，这个集合就是假设空间 (hypothesis space)
:::

例：一个学生收集了一些房价的数据。这些数据画出来：横轴表示房子的面积，单位是平方英尺，纵轴表示房价，单位是千美元。那基于这组数据，假如有一套750平方英尺房子，现在希望把房子卖掉，想知道这房子能卖多少钱。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654155544482-ac1cc23b-3654-4601-a9c1-0978392518c5.png#clientId=u70f7583f-91de-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=245&id=ub97eaf80&name=image.png&originHeight=337&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=75416&status=done&style=none&taskId=u9918ed7e-6b26-4096-8b0c-fb692bd4a8e&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=270&id=etABB&originHeight=337&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=481)

我们应用学习算法，可以在这组数据中画一条直线，或者换句话说，拟合一条直线，根据这条线我们可以推测出，这套房子可能卖$150,000，当然这不是唯一的算法。<br />比如用二次方程去拟合可能效果会更好。根据二次方程的曲线，我们可以从这个点推测出，这套房子能卖接近$200,000。

可以看出，监督学习指的就是我们给学习算法一个数据集。这个数据集由“正确答案”组成。在房价的例子中，我们给了一系列房子的数据，我们给定数据集中每个样本的正确价格，即它们实际的售价。然后运用学习算法，算出更多的正确答案。比如那个新房子的价格。

用术语来讲，这叫做**回归问题**。我们试着推测出一个连续值的结果，即房子的价格。一般房子的价格会记到美分，所以房价实际上是一系列离散的值，但是我们通常又把房价看成实数，看成是标量，所以又把它看成一个连续的数值。

**若欲预测的是连续值，则此类学习任务称为"回归" (regression)**

我再举另外一个监督学习的例子。假设你想通过查看病历来推测乳腺癌良性与否。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654155563117-0704eada-4304-45ec-8429-60c98742fdc4.png#clientId=u70f7583f-91de-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=209&id=udce3674c&name=image.png&originHeight=288&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=68126&status=done&style=none&taskId=u96156aa1-eba3-4246-b92a-1a2d7894aec&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=227&id=S0Nrh&originHeight=288&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=472)<br />	让我们来看一组数据：在这个数据集中，横轴表示肿瘤的大小，纵轴上，我标出1和0表示是或者不是恶性肿瘤。我们之前见过的肿瘤，如果是恶性则记为1，不是恶性，或者说良性记为0。

我有5个良性肿瘤样本，在1的位置有5个恶性肿瘤样本。现在我们有一个朋友很不幸检查出乳腺肿瘤。假设说她的肿瘤大概这么大，那么机器学习的问题就在于，你能否估算出肿瘤是恶性的或是良性的概率。用术语来讲，这是一个**分类问题**。

**若我们欲预测的是离散值，则此类学习任务称为 "分类" (classification);**

我们试着预测出离散的输出值：0或1良性或恶性，而事实上在分类问题中，输出可能不止两个值。比如说可能有三种乳腺癌，所以你希望预测离散输出0、1、2、3。0 代表良性，1 表示第1类乳腺癌，2表示第2类癌症，3表示第3类，但这也是分类问题。

现在我用不同的符号来表示这些数据。既然我们把肿瘤的尺寸看做区分恶性或良性的特征，那么可以用不同的符号来表示良性和恶性肿瘤，比如良性的肿瘤改成用 **O** 表示，恶性的继续用 **X** 表示。

在其它一些机器学习问题中，可能会遇到不止一种特征。举个例子，我们不仅知道肿瘤的尺寸，还知道对应患者的年龄。在其他机器学习问题中，我们通常有更多的特征。有算法不仅能处理2种3种或5种特征，即使有无限多种特征都可以处理。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654155573500-b26ea970-1a1e-4a09-b183-6e3cb3cd4fd7.png#clientId=u70f7583f-91de-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=199&id=u003bcb33&name=image.png&originHeight=274&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=50214&status=done&style=none&taskId=u6802ce6b-916f-4911-bece-9a0f1f14b71&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=206&id=Owtho&originHeight=274&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=451)<br />	上图中，列举了总共5种不同的特征，坐标轴上的两种和右边的3种，但是在一些学习问题中，你希望不只用3种或5种特征。相反，你想用无限多种特征，好让你的算法可以利用大量的特征，或者说线索来做推测。**可以采用支持向量机算法，让计算机处理无限多个特征。**

现在来回顾一下监督学习。**其基本思想是，我们数据集中的每个样本都有相应的“正确答案”，再根据这些样本作出预测。**我们还介绍了回归问题，即通过回归来推出一个连续的输出，之后我们介绍了分类问题，其目标是推出一组离散的结果。

:::success
**测验：假设你经营着一家公司，你想开发学习算法来处理这两个问题：**

1. ** 你有一大批同样的货物，这时你想预测接下来的三个月能卖多少件？ **
2. ** 你有许多客户，这时你想写一个软件来检验每一个用户的账户。对于每一个账户，你要判断它们是否曾经被盗过？ **

**那这两个问题，它们属于分类问题、还是回归问题?**<br />解答：<br />问题一是一个回归问题，因为你知道，如果我有数千件货物，我会把它看成一个实数，一个连续的值。因此卖出的物品数，也是一个连续的值。<br />问题二是一个分类问题，因为我会把预测的值，用 0 来表示账户未被盗，用 1 表示账户曾经被盗过。所以我们根据账号是否被盗过，把它们定为0 或 1，然后用算法推测一个账号是 0 还是 1，因为只有少数的离散值，所以我把它归为分类问题。
:::

---

<a name="L1gk7"></a>
## 无监督学习

:::info
定义：无监督学习是从无标注的数据中学习数据的统计规律或者说内在结构的机器学习，主要包括聚类、降维、概率估计。无监督学习可以用于数据分析或者监督学习的前处理。无监督学习的基本想法是对给定数据(矩阵数据)进行某种"压缩"，从而找到数据的潜在结构。
:::

:::info
定义：聚类(clustering )是将样本集合中相似的样本(实例)分配到相同的类，不相似的样本分配到不同的类。聚类时，样本通常是欧氏空间中的向量，类别不是事先给定， 而是从数据中自动发现，但类别的个数通常是事先给定的。
:::

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654155582523-34f10818-ec92-4905-926b-2bafd438d49e.png#clientId=u70f7583f-91de-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=143&id=u2004cd84&name=image.png&originHeight=197&originWidth=200&originalType=binary&ratio=1&rotation=0&showTitle=false&size=12172&status=done&style=none&taskId=u8f6dc829-733c-41ec-803e-73d211d7e7f&title=&width=145.45454545454547#crop=0&crop=0&crop=1&crop=1&id=nI3uF&originHeight=197&originWidth=200&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654155588859-b194c62b-9846-4b83-a91e-a97c6e1744f6.png#clientId=u70f7583f-91de-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=138&id=ub9f566fa&name=image.png&originHeight=190&originWidth=200&originalType=binary&ratio=1&rotation=0&showTitle=false&size=17629&status=done&style=none&taskId=u0979fbc9-0f69-45c4-8d26-dc615bbb245&title=&width=145.45454545454547#crop=0&crop=0&crop=1&crop=1&id=WKXe1&originHeight=190&originWidth=200&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)<br />	在无监督学习中，我们已知的数据看上去有点不一样，不同于监督学习的数据的样子，无监督学习中没有任何的标签或者是有相同的标签或者就是没标签。所以我们已知数据集，但不知道每个数据是什么东西，也不知如何处理，只有一个数据集。<br />针对数据集，无监督学习能判断出数据有两个不同的聚集簇。无监督学习算法可能会把这些数据分成两个不同的簇。所以叫做**聚类算法**。

聚类应用的一个例子就是在谷歌新闻中。谷歌新闻每天都在，收集非常多，非常多的网络的新闻内容。它再将这些新闻分组，组成有关联的新闻。所以谷歌新闻做的就是搜索非常多的新闻事件，自动地把它们聚类到一起。所以，这些新闻事件全是同一主题的，所以显示到一起。

聚类算法和无监督学习算法同样还用在很多其它的问题上。因为我们没有提前告知算法一些信息，比如，这是第一类的人，那些是第二类的人，还有第三类等等。我们只是说这有一堆数据，但我不知道数据里面有什么。我不知道谁是什么类型。我甚至不知道人们有哪些不同的类型，这些类型又是什么。但算法能自动地聚类那些个体到各个类。我没法提前知道哪个是哪类，因为我们没有给算法正确答案来回应数据集中的数据，所以这就是无监督学习。

无监督学习或聚集有着大量的应用，比如市场分割。许多公司有大型的数据库，存储消费者信息。所以，你能检索这些顾客数据集，自动地发现市场分类，并自动地把顾客划分到不同的细分市场中，你才能自动并更有效地销售或不同的细分市场一起进行销售。因为我们拥有所有的顾客数据，但我们没有提前知道是什么的细分市场，以及分别有哪些我们数据集中的顾客。我们不知道谁是在一号细分市场，谁在二号市场，等等。那我们就必须让算法从数据中发现这一切。<br />最后，无监督学习也可用于天文数据分析，这些聚类算法给出了令人惊讶、有趣、有用的理论，解释了星系是如何诞生的。这些都是聚类的例子，聚类只是无监督学习中的一种。

我们介绍了无监督学习，它是学习策略，交给算法大量的数据，并让算法为我们从数据中找出某种结构。

**垃圾邮件问题**的例子。如果你有标记好的数据，区别好是垃圾还是非垃圾邮件，我们把这个当作**监督学习问题**。

**新闻事件分类**的例子，可以用一个聚类算法来聚类这些文章到一起，所以是**无监督学习**。

**细分市场**的例子，你可以当作**无监督学习**问题，因为我只是拿到算法数据，再让算法去自动地发现细分市场。

接下来我们将深入探究特定的学习算法，开始介绍这些算法是如何工作的，和我们还有你如何来实现它们。

---

<a name="y5z6P"></a>
# 单变量线性回归(Linear Regression with One Variable)

<a name="nQbg5"></a>
## 通用模型表示

:::info
给定由$d$个属性描述的示例$x=(x_{1};x_{2};...;x_{d})$其中均是 在第$d$个属性上的取值，线性模型(linear model) 试图学得一个通过属性的线性组合来进行预测的函数，即 

$f(x)=\theta_{1}x_{1}+\theta _{2}x_{2}+...+\theta_{d}x_{d}+b$

一般用向量形式表示 $f(x)=\theta ^{T}x+b$ ，其中$\theta ^{T}=(\theta _{1};\theta _{2};...;\theta _{d})$.$\theta , d$学得之后，模型也就确定。<br />**"线性回归" (linear regression) 试图学得一个线性模型以尽可能准确地预测实值输出标记。**
:::

例：我们要使用一个数据集，数据集包含俄勒冈州波特兰市的住房价格。在这里，根据不同房屋尺寸所售出的价格，画出数据集。比方说，如果你朋友的房子是1250平方尺大小，你要告诉他们这房子能卖多少钱。那么，你可以做的一件事就是构建一个模型，也许是条直线，从这个数据模型上来看，也许你可以告诉你的朋友，他能以大约220,000(美元)左右的价格卖掉这个房子。这就是监督学习算法的一个例子。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654161228431-0452df36-8fcd-409b-ba5f-00e288ef855f.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=233&id=uba39af81&name=image.png&originHeight=321&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=115686&status=done&style=none&taskId=u52d0bde3-6475-41b5-86cf-bd4f73bbef9&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=230&id=jsEyj&originHeight=321&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=429)

它被称作监督学习是因为对于每个数据来说，我们给出了“正确的答案”，即告诉我们：根据我们的数据来说，房子实际的价格是多少，而且，更具体来说，这是一个回归问题。回归一词指的是，我们根据之前的数据预测出一个准确的输出值，对于这个例子就是价格，同时，还有另一种最常见的监督学习方式，叫做分类问题，当我们想要预测离散的输出值，例如，我们正在寻找癌症肿瘤，并想要确定肿瘤是良性的还是恶性的，这就是0/1离散输出的问题。更进一步来说，在监督学习中我们有一个数据集，这个数据集被称训练集。

**我将在整个课程中用小写的**$m$**来表示训练样本的数目。**

以之前的房屋交易问题为例，假使我们回归问题的训练集（Training Set）如下表所示：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654161238945-155f0a7b-834f-482b-bafd-2843b5a2e273.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=137&id=u73e30bc5&name=image.png&originHeight=188&originWidth=411&originalType=binary&ratio=1&rotation=0&showTitle=false&size=5692&status=done&style=none&taskId=u320b3ad2-7ecd-4cfd-bfe6-2dada759366&title=&width=298.90909090909093#crop=0&crop=0&crop=1&crop=1&id=Mf0p0&originHeight=188&originWidth=411&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

我们将要用来描述这个回归问题的标记如下:

:::info
$m$：代表训练集中实例的数量<br />$x$ ： 代表特征/输入变量<br />$y$ ：代表目标变量/输出变量<br />$\left( x,y \right)$ ：代表训练集中的实例<br />$({{x}^{(i)}},{{y}^{(i)}})$ ：代表第$i$个观察实例<br />$h$  ：代表学习算法的解决方案或函数也称为假设（**hypothesis**）
:::

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654161250365-a6876e25-daeb-4389-bbbc-57ad2bff215b.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=197&id=u29ff9950&name=image.png&originHeight=271&originWidth=339&originalType=binary&ratio=1&rotation=0&showTitle=false&size=8113&status=done&style=none&taskId=u7a4e35e5-ca8d-4b42-9f77-23ca7cbce48&title=&width=246.54545454545453#crop=0&crop=0&crop=1&crop=1&height=210&id=EWBbg&originHeight=271&originWidth=339&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=263)<br />	这就是一个监督学习算法的工作方式，我们可以看到这里有我们的训练集里房屋价格，我们把它喂给我们的学习算法，然后输出一个函数，通常用$h$表示。$h$意为hypothesis(假设)，表示一个函数，输入$x$是房屋尺寸大小，$y$ 值对应房子的价格 ，因此$h$根据输入的$x$值来得出$y$值，即$h$是一个从$x$ 到$y$的函数映射。

$h$代表hypothesis因而，我们实际上是要将训练集“喂”给我们的学习算法，进而学习得到一个假设$h$，然后将我们要预测的房屋的尺寸作为输入变量输入给$h$，预测出该房屋的交易价格作为输出变量输出为结果。那么，对于我们的房价预测问题，我们该如何表达$h$？

一种可能的表达方式为：$h_\theta \left( x \right)=\theta_{0} + \theta_{1}x$，因为只含有一个特征/输入变量，因此这样的问题叫作单变量线性回归问题。

---


<a name="a9Hwt"></a>
## 代价函数

如何选择决策函数$h_{\theta}(x)$的参数$\theta_{i}$<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654161258688-dc35e3d3-71f2-40f2-95d0-21458c1c734a.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=192&id=u085bc8ed&name=image.png&originHeight=215&originWidth=400&originalType=binary&ratio=1&rotation=0&showTitle=false&size=51064&status=done&style=none&taskId=u11eb00f9-0510-41dd-bf7c-d74417c8650&title=&width=357.9090881347656#crop=0&crop=0&crop=1&crop=1&id=gEyMe&originHeight=215&originWidth=400&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

在线性回归中我们有一个像这样的训练集，$m$代表了训练样本的数量，比如 $m = 47$。而我们的假设函数，也就是用来进行预测的函数，是这样的：$h_\theta \left( x \right)=\theta_{0}+\theta_{1}x$。

现在要做的便是为模型选择合适的**参数**（**parameters**）$\theta_{0}$ 和 $\theta_{1}$。我们选择的参数决定了我们得到的直线相对于我们的训练集的准确程度，模型所预测的值与训练集中实际值之间的差距（下图中蓝线所指）就是**建模误差**（**modeling error**）。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654161269581-67e5f92e-2574-4fb3-b6ed-9beed1abc671.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=151&id=ua4cfffdf&name=image.png&originHeight=207&originWidth=262&originalType=binary&ratio=1&rotation=0&showTitle=false&size=8002&status=done&style=none&taskId=u3010a761-1e37-4e5e-8cef-0b4020c0e19&title=&width=190.54545454545453#crop=0&crop=0&crop=1&crop=1&id=OAxvl&originHeight=207&originWidth=262&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

我们的目标便是选择出可以使得建模误差的平方和能够最小的模型参数。 即使得均方代价函数 

$J \left( \theta_0, \theta_1 \right) = \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right)^{2}$最小。基于均方误差最小化来进行模型求解模型的方法称为**“最小二乘法”**

常数$\frac{1}{2}$不会带来本质差别，但可使损失函数求导后常数系数为1。平方误差代价函数可能是解决回归问题最常用的性能度量手段。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654174571837-78e15646-c57d-45b6-b94f-f649a973689e.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=219&id=uc8fa3b79&name=image.png&originHeight=381&originWidth=875&originalType=binary&ratio=1&rotation=0&showTitle=false&size=91244&status=done&style=none&taskId=ud321c373-1a13-4efe-ab2a-6e22ea6c72d&title=&width=503.3636474609375#crop=0&crop=0&crop=1&crop=1&height=161&id=BSVfD&originHeight=381&originWidth=875&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=370)


接下来让我们通过一些例子来获取一些直观的感受，看看代价函数到底是在干什么。

注意：h是θ确定后关于x的函数，J是关于θ的函数<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654161298175-680a4fc2-6e9b-4cc4-bd6a-82d1f7c02615.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=246&id=u54833861&name=image.png&originHeight=338&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=160289&status=done&style=none&taskId=u9d49e83d-6407-40e8-8bd0-7adeb413e59&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=290&id=tcULJ&originHeight=338&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=514)

我们绘制一个等高线图，三个坐标分别为$\theta_{0}$和$\theta_{1}$ 和$J(\theta_{0}, \theta_{1})$，则可以看出在三维空间中存在一个使得$J(\theta_{0}, \theta_{1})$最小的点。代价函数的样子，等高线图，则可以看出在三维空间中存在一个使得$J(\theta_{0}, \theta_{1})$最小的点。线性回归的代价函数总是一个**凸函数**，无局部最优，只有一个全局最优解。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654161308316-67f94792-78b4-40c3-9391-d8901e7a03d1.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=184&id=eJpao&name=image.png&originHeight=253&originWidth=400&originalType=binary&ratio=1&rotation=0&showTitle=false&size=102276&status=done&style=none&taskId=u58362e4d-e11c-437c-8425-49630ee8c2c&title=&width=290.90909090909093#crop=0&crop=0&crop=1&crop=1&id=HyxnR&originHeight=253&originWidth=400&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654161317578-f7fa27f3-2291-4340-9609-405294ca103d.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=212&id=u8d5ab539&name=image.png&originHeight=291&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=172329&status=done&style=none&taskId=u1ebafcb2-1147-4884-bd61-3f2a7b052a9&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=259&id=p2aJm&originHeight=291&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=535)

通过这些图形能更好地理解这些代价函数$J$所表达的值是什么样的，它们对应的假设是什么样的，以及什么样的假设对应的点，更接近于代价函数最小值。

当然，我们真正需要的是一种有效的算法，能够自动地找出这些使代价函数$J$取最小值的参数$\theta_{0}$和$\theta_{1}$来。我们也会遇到更复杂、更高维度、更多参数的情况，而这些情况是很难画出图的，因此更无法将其可视化，因此我们真正需要的是编写程序来找出这些最小化代价函数的$\theta_{0}$和$\theta_{1}$的值。在下一节，将介绍一种算法能够自动地找出能使代价函数$J$最小化的参数$\theta_{0}$和$\theta_{1}$的值。

---

<a name="IfzcB"></a>
## 梯度下降

梯度下降是一个用来求函数最小值的算法，我们将使用梯度下降算法来求出代价函数$J(\theta_{0}, \theta_{1})$ 的最小值。

**梯度下降背后的思想是：开始时我们随机选择一个参数的组合**$\left( {\theta_{0}},{\theta_{1}},......,{\theta_{n}} \right)$**，计算代价函数，然后我们寻找下一个能让代价函数值下降最多的参数组合。我们持续这么做直到找到一个局部最小值（local minimum）**，因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是全局最小值（global minimum），选择不同的初始参数组合，可能会找到不同的局部最小值。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654161328068-78cbe18b-a81b-48f5-a96f-27ae7f96eb3e.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=180&id=u18940e85&name=image.png&originHeight=248&originWidth=500&originalType=binary&ratio=1&rotation=0&showTitle=false&size=156240&status=done&style=none&taskId=uac9f404f-54fb-404b-bc0a-3ad03717f1d&title=&width=363.6363636363636#crop=0&crop=0&crop=1&crop=1&height=221&id=paOZ6&originHeight=248&originWidth=500&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=446)

想象一下你正站立在山的这一点上，站立在你想象的公园这座红色山上，在梯度下降算法中，我们要做的环看周围，并问自己要在某个方向上，用小碎步尽快下山。这些小碎步需要朝什么方向？如果我们站在山坡上的这一点，你看一下周围，你会发现最佳的下山方向，你再看看周围，然后再一次想想，我应该从什么方向迈着小碎步下山？然后你按照自己的判断又迈出一步，重复上面的步骤，从这个新的点，你环顾四周，并决定从什么方向将会最快下山，然后又迈进了一小步，并依此类推，直到你接近局部最低点的位置。

:::info
**定义：**梯度（gradient）向量，即⼀个多元函数对其所有变量的偏导数。表示某一函数在该点处的方向导数沿着该方向取得最大值，即函数在该点处沿着该方向（此梯度的方向）变化最快，变化率最大（为该梯度的模）
:::

**梯度下降**（**gradient descent**）算法的公式为：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654178078626-8c5f1f96-7761-4a8d-8fc2-d2009ea20e67.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=242&id=ue9a4efce&name=image.png&originHeight=469&originWidth=711&originalType=binary&ratio=1&rotation=0&showTitle=false&size=111085&status=done&style=none&taskId=u57f0c017-45a6-47f9-8a24-aa14915936b&title=&width=367.0909118652344#crop=0&crop=0&crop=1&crop=1&height=280&id=KjG11&originHeight=469&originWidth=711&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=425)<br />即$\theta=\theta -\alpha\nabla\ _{\theta}J(\theta)$（$\nabla$：哈密顿算子，$\nabla\ _{\theta}J(\theta)$即为$J(\theta)$相对于$\theta$的梯度）

其中$\bold{\alpha}$是学习率（learning rate），它决定了我们沿着能让代价函数下降程度最大的方向向下的步长，在批量梯度下降中，我们每一次都同时让所有的参数减去学习速率乘以代价函数对相应参数的偏导数

---

<a name="Uq5oA"></a>
## 梯度下降的直观理解

在之前的视频中，我们给出了一个数学上关于梯度下降的定义，本次视频我们更深入研究一下，更直观地感受一下这个算法是做什么的，以及梯度下降算法的更新过程有什么意义。梯度下降算法如下：

${\theta_{j}}:={\theta_{j}}-\alpha \frac{\partial }{\partial {\theta_{j}}}J\left(\theta \right)$

描述：对 赋值，使得按梯度下降最快方向进行，迭代下去直到得到局部最小值。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654181513612-b0ef30e0-cf83-4c3f-9de1-b0bb60954f1b.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=209&id=u250b05ca&name=image.png&originHeight=288&originWidth=405&originalType=binary&ratio=1&rotation=0&showTitle=false&size=12254&status=done&style=none&taskId=u54d0a004-cde9-4eeb-b557-8c2eeeae0ee&title=&width=294.54545454545456#crop=0&crop=0&crop=1&crop=1&height=191&id=B0tKo&originHeight=288&originWidth=405&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=268)<br />对于这个问题，求导的目的，基本上可以说取这个红点的切线，这条红色直线的斜率正好是这个三角形的高度除以这个水平长度，现在这条线有一个正斜率，也就是说它有正导数，因此，我们得到新的${\theta_{1}}$：即${\theta_{1}}$减去$J(\theta)$对$\theta_{1}$的偏导乘以$\alpha$。

这就是我梯度下降法的更新规则：${\theta_{j}}:={\theta_{j}}-\alpha \frac{\partial }{\partial {\theta_{j}}}J\left( \theta  \right)$

**让我们来看看如果**$\alpha$**太小或**$\alpha$**太大会出现什么情况：**<br />**如果**$\alpha$**太小了，即我的学习速率太小，这样就需要很多步才能到达最低点，收敛可能会很慢**，因为它会一点点挪动，它会需要很多步才能到达全局最低点。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654179587164-7fbe8a5c-a036-4ed4-8dea-37b9865845b9.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=207&id=uf7c65a59&name=image.png&originHeight=289&originWidth=413&originalType=binary&ratio=1&rotation=0&showTitle=false&size=12398&status=done&style=none&taskId=uf498f1cc-4dee-4770-be84-5d462c00fdc&title=&width=295.3636474609375#crop=0&crop=0&crop=1&crop=1&height=220&id=D5gK8&originHeight=289&originWidth=413&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=314)<br />**如果**$\alpha$**太大，那么梯度下降法可能会越过最低点，甚至可能无法收敛**，下一次迭代又移动了一大步，越过一次，又越过一次，一次次越过最低点，直到你发现实际上离最低点越来越远，所以，如果$\alpha$太大，它会导致无法收敛，甚至发散。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654179577253-e530c5ea-8ba6-4b72-b390-1f2def460a67.png#clientId=ue15db1c4-cec8-4&crop=0.0252&crop=0.032&crop=1&crop=1&from=paste&height=214&id=u4d9b371f&name=image.png&originHeight=301&originWidth=437&originalType=binary&ratio=1&rotation=0&showTitle=false&size=16271&status=done&style=none&taskId=ufd592aa7-0c27-482b-9355-f314932bd87&title=&width=310#crop=0&crop=0&crop=1&crop=1&height=227&id=O9zMs&originHeight=301&originWidth=437&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=330)<br />如果我们预先把${\theta_{1}}$放在一个局部的最低点，你认为下一步梯度下降法会怎样工作？

假设你将${\theta_{1}}$初始化在一个局部的最优处或局部最低点。结果是局部最优点的导数将等于零，因为它是那条切线的斜率。这意味着你已经在局部最优点，它使得${\theta_{1}}$不再改变，也就是新的${\theta_{1}}$等于原来的${\theta_{1}}$，因此，如果你的参数已经处于局部最低点，那么梯度下降法更新其实什么都没做，它不会改变参数的值。

在梯度下降法中，当我们接近局部最低点时，梯度下降法会自动采取更小的幅度。这是因为局部最低时导数等于零，所以当我们接近局部最低点时，导数值会自动变得越来越小，所以梯度下降将自动采取较小的幅度，直到$J$收敛于局部极小值，这就是梯度下降的做法。所以实际上**没有必要再另外减小**$\alpha$。这也解释了为什么即使学习速率$\alpha$保持不变时，梯度下降也可以收敛到局部最低点。

这就是梯度下降算法，你可以用它来最小化任何代价函数$J$，不只是线性回归中的代价函数$J$。

在接下来的视频中，我们要用代价函数$J$，回到它的本质，线性回归中的代价函数。也就是我们前面得出的平方误差函数，结合梯度下降法，以及平方代价函数，我们会得出第一个机器学习算法，即线性回归算法。

---

<a name="Nvoso"></a>
## 梯度下降的线性回归

本节，我们要将梯度下降和代价函数结合，将其应用于具体的拟合直线的线性回归算法里。

梯度下降算法和线性回归算法比较如图：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654184337645-5f6bf289-3867-4847-a766-89263275bf3e.png#clientId=ue15db1c4-cec8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=179&id=u929807cb&name=image.png&originHeight=412&originWidth=1226&originalType=binary&ratio=1&rotation=0&showTitle=false&size=148080&status=done&style=none&taskId=ube8e51a1-366d-4d5d-99bc-48a8c2388f6&title=&width=532#crop=0&crop=0&crop=1&crop=1&height=164&id=ZFv1g&originHeight=412&originWidth=1226&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=488)

对我们之前的线性回归问题运用梯度下降法，关键在于求出代价函数的导数，即：(很重要！)<br />$\frac{\partial }{\partial {{\theta }_{j}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{\partial }{\partial {{\theta }_{j}}}\frac{1}{2m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}^{2}}$

$j=0$  时：$\frac{\partial }{\partial {{\theta }_{0}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{1}{m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}}$<br />$j=1$  时：$\frac{\partial }{\partial {{\theta }_{1}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

$\begin{aligned}
&\frac{\partial }{\partial {{\theta }_{j}}}J({{\theta }_{0}},{{\theta }_{1}})=\frac{\partial }{\partial {{\theta }_{j}}}\frac{1}{2m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}^{2}}=\left\{\begin{matrix}
 \frac{1}{m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}} & ,j=0\\
  \frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}&,j=1
\end{matrix}\right.
\end{aligned}$

则算法改写成：

**Repeat {**<br />                ${\theta_{0}}:={\theta_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{ \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}$<br />                ${\theta_{1}}:={\theta_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$<br />               **}**

我们刚刚使用的算法，有时也称为**“批量梯度下降（Batch Gradient Descent）”**。指的是在梯度下降的每一步中，我们都用到了所有的训练样本。事实上，有时也有其他类型的梯度下降法，不是这种"批量"型的，不考虑整个的训练集，而是每次只关注训练集中的一些小的子集。

有一种计算代价函数$J$最小值的数值解法，不需要梯度下降这种迭代算法。在后面的课程中，我们也会谈到这个方法，它可以在不需要多步梯度下降的情况下，也能解出代价函数$J$的最小值，这是另一种称为正规方程(normal equations)的方法。实际上在数据量较大的情况下，梯度下降法比正规方程要更适用一些。

---

<a name="ZxCkI"></a>
## 多元线性回归(Linear Regression with Multiple Variables)

目前为止，我们探讨了单变量/特征的回归模型，现在我们对房价模型增加更多的特征变为“**多维特征”**，例如房间数楼层等，构成一个含有多个变量的模型，模型中的特征为$\left( {x_{1}},{x_{2}},...,{x_{n}} \right)$。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654239373128-497b59a6-3472-40ee-9bad-85ebe3b7ac87.png#clientId=ub5ad8871-3d34-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=180&id=u9dc010fa&name=image.png&originHeight=247&originWidth=591&originalType=binary&ratio=1&rotation=0&showTitle=false&size=20710&status=done&style=none&taskId=u077a78f4-34cf-4dec-8f02-4da4acccc1d&title=&width=429.8181818181818#crop=0&crop=0&crop=1&crop=1&height=177&id=aU0oQ&originHeight=247&originWidth=591&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=424)<br />增添更多特征后，我们引入一系列新的注释：

:::info
$n$ 代表特征的数量<br />${x^{\left( i \right)}}$**代表第 **$i$** 个训练实例，即特征矩阵中的第**$i$**行，是一个向量（vector）。**如上图${x}^{(2)}\text{=}\begin{bmatrix} 1416\\\ 3\\\ 2\\\ 40 \end{bmatrix}$<br />${x}_{j}^{\left( i \right)}$代表特征矩阵中第 $i$ 行的第 $j$ 个特征，如上图：$x_{2}^{\left( 2 \right)}=3,x_{3}^{\left( 2 \right)}=2$

多元假设 $h$ 表示为：$h_{\theta}\left( x^{(i)} \right)={\theta_{0}}+{\theta_{1}}{x^{(i)}_{1}}+{\theta_{2}}{x^{(i)}_{2}}+...+{\theta_{n}}{x^{(i)}_{n}}$
:::

这个公式中有$n+1$个参数和$n$个变量，为了使得公式能够简化一些，引入$x_{0}=1$，则公式转化为：

$h_{\theta} \left( x^{(i)} \right)={\theta_{0}}{x^{(i)}_{0}}+{\theta_{1}}{x^{(i)}_{1}}+{\theta_{2}}{x^{(i)}_{2}}+...+{\theta_{n}}{x^{(i)}_{n}}$

此时模型中的参数是一个$n+1$维的向量，任何一个训练实例也都是$n+1$维的向量，特征矩阵$X$的维度是 $m·(n+1)$。 因此公式可以简化为：<br />$h_{\theta} \left( x \right)={\theta^{T}}X$

---

<a name="lUsMr"></a>
## 多元梯度下降

与单变量线性回归类似，在多变量线性回归中，我们也构建一个代价函数，则这个代价函数是所有建模误差的平方和，即：<br />$J\left( {\theta_{0}},{\theta_{1}}...{\theta_{n}} \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( h_{\theta} \left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}}$

其中：$h_{\theta}\left( x \right)=\theta^{T}X={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$ ，

我们的目标和单变量线性回归问题中一样，是要找出使得代价函数最小的一系列参数，即：<br />$\theta=\mathrm{arg} \min_{\theta}(Y-X\theta)^{\mathrm{T}}(Y-X\theta)$

<br />**多变量线性回归的批量梯度下降算法为：**

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654239506434-fa4d85bb-7858-4084-b782-7444dce08c23.png#clientId=ub5ad8871-3d34-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=87&id=uaa7a16a1&name=image.png&originHeight=120&originWidth=307&originalType=binary&ratio=1&rotation=0&showTitle=false&size=7697&status=done&style=none&taskId=u9751775f-e896-4b10-9d78-0a2da0b9ce9&title=&width=223.27272727272728#crop=0&crop=0&crop=1&crop=1&height=91&id=XxT0o&originHeight=120&originWidth=307&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=233)

即：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654240154295-47f8b806-1e37-4147-9c06-0a513425ebb4.png#clientId=ub5ad8871-3d34-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=189&id=HNJim&name=image.png&originHeight=260&originWidth=442&originalType=binary&ratio=1&rotation=0&showTitle=false&size=49876&status=done&style=none&taskId=ube95d591-8ada-46f0-9255-a5d3f14063c&title=&width=321.45454545454544#crop=0&crop=0&crop=1&crop=1&height=168&id=qf8fX&originHeight=260&originWidth=442&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=286)<br />当$n>=1$时，<br />${{\theta }_{0}}:={{\theta }_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{0}^{(i)}$<br />${{\theta }_{1}}:={{\theta }_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{1}^{(i)}$<br />${{\theta }_{2}}:={{\theta }_{2}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}})}x_{2}^{(i)}$

与之前一样，开始时我们随机选择一系列的参数值，计算所有的预测结果后，再更新所有的参数的值，如此循环直到收敛。

---

<a name="sz9nz"></a>
## 梯度下降法实践1-特征缩放

在我们面对多维特征问题的时候，我们要保证这些特征都具有相近的尺度，这将帮助梯度下降算法更快地收敛。

以房价问题为例，假设我们使用两个特征，房屋的尺寸和房间的数量，尺寸的值为 0-2000平方英尺，而房间数量的值则是0-5，以两个参数分别为横纵坐标绘制代价函数的等高线图像，可以看出图像会显得很扁，梯度下降算法需要非常多次的迭代才能收敛。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654241095172-97bc16bf-9324-4ad7-8cb8-0f154be86110.png#clientId=ub5ad8871-3d34-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=258&id=udecb859b&name=image.png&originHeight=530&originWidth=453&originalType=binary&ratio=1&rotation=0&showTitle=false&size=46249&status=done&style=none&taskId=u0063b291-0182-4447-aeea-eb3f455e0c7&title=&width=220.45455932617188#crop=0&crop=0&crop=1&crop=1&height=256&id=HLyF7&originHeight=530&originWidth=453&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=219)

解决的方法是尝试将所有特征的尺度都尽量缩放到-1到1之间。如图：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654239882800-7d763f18-5eb2-44a2-9aaf-e99e3bc62a8f.png#clientId=ub5ad8871-3d34-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=249&id=ua6b79b31&name=image.png&originHeight=342&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=115141&status=done&style=none&taskId=u1183595b-3390-46fc-82a5-18f2114420d&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=276&id=ilvlM&originHeight=342&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=484)

最简单的方法是令：$\bold{{{x}_{n}^{*}}=\frac{{{x}_{n}}-{{\mu}_{n}}}{{{s}_{n}}}}$，即概率论中的标准化随机变量，${\mu_{n}}$是平均值，${s_{n}}$是标准差或者为变量的范围。

---

<a name="TzcYj"></a>
## 梯度下降法实践2-学习率

梯度下降算法收敛所需要的迭代次数根据模型的不同而不同，我们不能提前预知，我们可以绘制迭代次数和代价函数的图表来观测算法在何时趋于收敛。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654242515529-e95e724e-2f7a-42f1-b893-2d83dc73cf53.png#clientId=ub5ad8871-3d34-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=291&id=u726a372d&name=image.png&originHeight=594&originWidth=1119&originalType=binary&ratio=1&rotation=0&showTitle=false&size=99162&status=done&style=none&taskId=u9ff002a2-4422-42ef-a87f-8815ac93148&title=&width=549#crop=0&crop=0&crop=1&crop=1&height=235&id=Hs8Uc&originHeight=594&originWidth=1119&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=443)

有一些自动测试是否收敛的方法，例如将代价函数的变化值与某个阀值$\epsilon$（例如0.001）进行比较，但$\epsilon$的选取通常比较困难，因此通常看图表更加直观。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654242784979-fbbf1c36-004f-4804-abf7-b91ac03e3957.png#clientId=ub5ad8871-3d34-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=284&id=u44e31a29&name=image.png&originHeight=603&originWidth=1130&originalType=binary&ratio=1&rotation=0&showTitle=false&size=101536&status=done&style=none&taskId=u1b751b2f-ca22-45f2-8f38-db50be1f1fb&title=&width=533#crop=0&crop=0&crop=1&crop=1&height=248&id=Y6pEz&originHeight=603&originWidth=1130&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=464)<br />	如上图所示，学习率$\alpha$过大，可能会出现图示曲线形状，即每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛。

梯度下降算法的每次迭代受到学习率的影响。数学上可以证明，$\alpha$足够小，每次迭代后$J(\theta)$都会下降。但如果学习率$a$过小，则达到收敛所需的迭代次数会非常高。

通常可以考虑尝试些学习率：<br />$\alpha=0.01，0.03，0.1，0.3，1，3，10$（大致按3倍增加）。<br />先选取一系列学习率，然后找到一个合适的范围，最后选取范围内较大的学习率。

---

<a name="jLOIC"></a>
## 特征和多项式回归

例：房价预测问题<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654243241847-3347e204-ada5-47f5-9224-1c8467207b94.png#clientId=ub5ad8871-3d34-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=218&id=ufe67de2c&name=image.png&originHeight=324&originWidth=434&originalType=binary&ratio=1&rotation=0&showTitle=false&size=102325&status=done&style=none&taskId=u88305f23-977a-45b7-9046-57ab4918160&title=&width=292.6363830566406#crop=0&crop=0&crop=1&crop=1&height=222&id=zQF6O&originHeight=324&originWidth=434&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=298)

我们可以选取两个特征：${x_{1}}=frontage$（临街宽度），${x_{2}}=depth$（纵向深度），由此得到假设1<br />$h_{\theta }\left( x \right)={\theta_{0}}+{\theta_{1}}\times{frontage}+{\theta_{2}}\times{depth}$<br />我们还可以选取特征：$x=frontage*depth=area$（面积），由此得到假设2<br />${h_{\theta}}\left( x \right)={\theta_{0}}+{\theta_{1}}x$（减少了一个特征）

**有时候对同一个问题的不同特征选取可能会得到一个更好的模型。**

线性回归并不适用于所有数据，有时需要曲线来适应数据。通常我们需要先观察数据然后再决定准备尝试怎样的模型。比如一个二次模型$h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x}+{\theta_{2}}{x^{2}}$或者三次模型： $h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x}+{\theta_{2}}{x^2}+{\theta_{3}}{x^3}$

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654243692481-1c8c249b-4265-44ab-9320-43c5b81ef7fe.png#clientId=ub5ad8871-3d34-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=263&id=u07433cfe&name=image.png&originHeight=489&originWidth=924&originalType=binary&ratio=1&rotation=0&showTitle=false&size=215969&status=done&style=none&taskId=u8cfef943-4c14-43a9-b490-809a29e38db&title=&width=497#crop=0&crop=0&crop=1&crop=1&height=274&id=iJHDn&originHeight=489&originWidth=924&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=518)

根据图像我们可以令：$x_{1}=size,x_{2}=size^{2}$,从而得到二次假设：${{{h}}_{\theta}}(x)={{\theta }_{0}}\text{+}{{\theta }_{1}}(size)+{{\theta}_{2}}{{(size)}^{2}}$

由于二次函数先增后减，不能很好的拟合数据，所以还可以令<br />$x_{1}=size,x_{2}=size^{2},x_{3}=size^{3}$，则有：${{{h}}_{\theta}}(x)={{\theta }_{0}}\text{+}{{\theta }_{1}}(size)+{{\theta}_{2}}{{(size)}^{2}+{{\theta}_{3}}{{(size)}^{3}}}$

上述两种方法均**将模型转化为线性回归模型**。

根据函数图形特性，我们还可以使：${{{h}}_{\theta}}(x)={{\theta }_{0}}\text{+}{{\theta }_{1}}(size)+{{\theta }_{2}}\sqrt{size}$

注意，如果我们采用多项式回归模型，在运行梯度下降算法前特征缩放非常有必要。

---

<a name="cxGKM"></a>
## 正规方程

到目前为止，我们都在使用梯度下降算法，但是对于某些线性回归问题，正规方程方法是更好的解决方案。如：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654247019223-b1ec07d3-2972-4f7a-8e4d-3e34d5c45684.png#clientId=uecb62040-5a91-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=286&id=ub65b4938&name=image.png&originHeight=635&originWidth=1182&originalType=binary&ratio=1&rotation=0&showTitle=false&size=205997&status=done&style=none&taskId=ubad198ea-1706-4e0a-a43b-636194e9e35&title=&width=533#crop=0&crop=0&crop=1&crop=1&height=265&id=IOeB0&originHeight=635&originWidth=1182&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=493)

正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：$\frac{\partial}{\partial{\theta_{j}}}J\left( {\theta_{j}} \right)=0$ 。<br />假设我们的训练集特征矩阵为 $X=\begin{bmatrix}
 (x^{(1)})^{T}
\\(x^{(2)})^{T}
\\...
\\(x^{(m)})^{T}
\end{bmatrix}$（包含了 ${{x}_{0}}=1$）并且我们的训练集结果为向量 $y$，有$h_{\theta}(x)=X\theta$，$J(\theta)=\frac{1}{2}(X\theta-y)^{T}(X\theta-y)$，则有：

$\begin{aligned}
&\frac{\partial J\left( \theta  \right)}{\partial \theta }={{X}^{T}}X\theta -{{X}^{T}}y\\&\\
&\bold{\theta ={{\left( {X^T}X \right)}^{-1}}{X^{T}}y}
\end{aligned}$

:::info
$\theta ={{\left( {X^{T}}X \right)}^{-1}}{X^{T}}y$** 的推导过程：**<br />$J\left( \theta  \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( {h_{\theta}}\left( {x^{(i)}} \right)-{y^{(i)}} \right)}^{2}}}$，其中：${h_{\theta}}\left( x \right)=X\theta$，$X=\begin{bmatrix} (x^{(1)})^{T}\\(x^{(2)})^{T}\\...\\(x^{(m)})^{T}\end{bmatrix}$则有：<br />$J(\theta )=\frac{1}{2}{{\left( X\theta -y\right)}^{T}}\left( X\theta -y \right)$

$=\frac{1}{2}\left( {{\theta }^{T}}{{X}^{T}}-{{y}^{T}} \right)\left(X\theta -y \right)$

$=\frac{1}{2}\left( {{\theta }^{T}}{{X}^{T}}X\theta -{{\theta}^{T}}{{X}^{T}}y-{{y}^{T}}X\theta -{{y}^{T}}y \right)$

接下来对$J(\theta )$偏导，需要用到以下几个矩阵的求导法则:<br />$\frac{dAB}{dB}={{A}^{T}}$，$\frac{d{{X}^{T}}AX}{dX}=2AX$

故：$\frac{\partial J\left( \theta  \right)}{\partial \theta }=\frac{1}{2}\left(2{{X}^{T}}X\theta -{{X}^{T}}y -{}({{y}^{T}}X )^{T}-0 \right)$

$=\frac{1}{2}\left(2{{X}^{T}}X\theta -{{X}^{T}}y -{{X}^{T}}y -0 \right)$

$={{X}^{T}}X\theta -{{X}^{T}}y$

令$\frac{\partial J\left( \theta  \right)}{\partial \theta }=0$，则有$\theta ={{\left( {X^{T}}X \right)}^{-1}}{X^{T}}y$
:::

以下表示数据为例：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654244966294-049237f4-c511-41b1-81d2-68b949d0ef04.png#clientId=u23f39f1f-571d-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=324&id=u72d98fb0&name=image.png&originHeight=608&originWidth=1033&originalType=binary&ratio=1&rotation=0&showTitle=false&size=91390&status=done&style=none&taskId=u059770f8-043d-48b8-b929-0fd95c7567f&title=&width=550#crop=0&crop=0&crop=1&crop=1&height=284&id=XLUmL&originHeight=608&originWidth=1033&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=483)


**梯度下降与正规方程的比较：**

| **梯度下降** | **正规方程** |
| --- | --- |
| 需要选择学习率$\alpha$ | 不需要 |
| 需要多次迭代 | 一次运算得出 |
| 当特征数量$n$大时也能较好适用 | 需要计算${{\left( {{X}^{T}}X \right)}^{-1}}$ 如果特征数量n较大则运算代价大，因为矩阵逆的计算时间复杂度为$O\left( {{n}^{3}} \right)$，通常来说当$n$<10000 时还是可以接受的 |
| 适用于各种类型的模型 | 只适用于线性模型，不适合逻辑回归模型等其他模型 |


总结：只要特征变量的数目并不大，正规方程是一个很好的计算参数$\theta$的方法。具体地说，只要特征变量数量小于10,000 通常使用标准方程法，而不使用梯度下降法。

随着我们要讲的学习算法越来越复杂，例如，当我们讲到分类算法，像逻辑回归算法，我们会看到，实际上对于那些算法，并不能使用正规方程法。对于那些更复杂的学习算法，我们将不得不仍然使用梯度下降法。但对于这个特定的线性回归模型，正规方程法是一个比梯度下降法更快的替代算法。所以，根据具体的问题，以及你的特征变量的数量，这两种算法都是值得学习的。


当计算 $\theta ={{\left( {X^T}X \right)}^{-1}}{X^{T}}y$ ，对于矩阵$X^{T}X$是**不可逆**的情况怎么办?

实际上$X^{T}X$的不可逆的问题很少发生，在**Octave**里，如果你用它来实现$\theta$的计算，你将会得到一个正常的解。在**Octave**里，有两个函数可以求解矩阵的逆，一个被称为`pinv()`，另一个是`inv()`，这两者之间的差异是些许计算过程上的，一个是所谓的**伪逆**，另一个被称为逆。使用`pinv()` 函数，即便矩阵$X^{T}X$是不可逆的，也可以计算出$\theta$的值。

通常，我们会使用一种叫做正则化的线性代数方法，通过删除某些特征或者是使用某些技术，来解决当$m$比$n$小的时候的问题。即使你有一个相对较小的训练集，也可使用很多的特征来找到很多合适的参数。总之，当你发现的矩阵$X^{T}X$的结果是奇异矩阵，或者找到的其它矩阵是不可逆的，我会建议你这么做。

首先，看特征值里是否有一些多余的特征，像这些${x_{1}}$和${x_{2}}$是线性相关的，互为线性函数。同时，当有一些多余的特征时，可以删除这两个重复特征里的其中一个，无须两个特征同时保留，可以解决不可逆性的问题。如果特征数量实在太多，我会删除某些使用较少的特征来反映尽可能多内容，或者考虑使用正规化方法。<br />	如果矩阵$X^{T}X$是不可逆的，如果在**Octave**里，可以用伪逆函数`pinv()` 来求解。即使$X'X$的结果是不可逆的，但算法执行的流程是正确的。总之，出现不可逆矩阵的情况极少发生，所以在大多数实现线性回归中，出现不可逆的问题不应该过多的关注${X^{T}}X$是不可逆的。

---

<a name="LhjGj"></a>
## 本章相关代码
[sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linearregression#sklearn.linear_model.LinearRegression)<br />[sklearn.linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html?highlight=sgdregressor#sklearn.linear_model.SGDRegressor)<br />[sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html?highlight=polynomialfeatures#sklearn.preprocessing.PolynomialFeatures)
```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression() #LinearRegression()基于scipy.linalg.lstsq()
# scipy.linalg.lstsq() 基于伪逆计算np.linalg.pinv()
lin_reg.fit(X, y)
print(lin_reg.predict(x_new)) # 输出预测结果
lin_reg.coef_，lin_reg.intercept_ # 查看截距和斜率
```
```python
from sklearn.linear_model import SGDRegressor

# 最多运行1000轮，或者一个轮次内损失下降小于1e-3，默认以0.1的学习率开始
sgd_reg = SGDRegressor(max_iter=1000, tol=1e-3, penalty=None, eta0=0.1, random_state=42)
sgd_reg.fit(X, y.ravel()) # y.ravel()将y转为一维

sgd_reg.intercept_, sgd_reg.coef_ # 查看截距和斜率 
```
```python
import numpy as np
    
#theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
theta = np.linalg.inv(X.T@X)@X.T@y # X.T@X等价于X.T.dot(X)
```
```python
import numpy as np

x=2*random.rand(100,1) # 随机生成[0,2)间的数
y=4+3*x+np.random.randn100,1() # 生成一些线性数据
m = len(X_b) # 样本特征个数

alpha = 0.1 #学习率
n_iterations= 1000 # 迭代次数

X_b = np.c_[np.ones((100, 1)), X]  #  给每一个样本数据添加 x0 = 1
theta = np.random.randn(2,1) # 随机初始化

for iteration in range(n_iterations):
    gradients = 1/m * X_b.T.dpt(X_b.dot(theta)-y) 
    theta = theta - alpha * gradients # 更新参数
```
（随机梯度下降和小批量随机梯度下降在之后的章节）
```python
import numpy as np

m = 100 # 特征维度
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree=2, include_bias=False)
# 使用PolynomialFeatures将多项中每一个二次型替代为一个新特征
# PolynomialFeatures还可以将特征的所有组合添加到给定的多项式阶数(degree)

X_poly = poly_features.fit_transform(X) #X_poly包含X的原始特征和该特征的平方

lin_reg = LinearRegression() # 对新扩展数据集进行线性拟合即可
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
```

---

<a name="ULgQs"></a>
# 逻辑回归(Logistic Regression)

<a name="eikux"></a>
## 分类问题

在分类问题中，要预测的变量 $y$ 是离散的值，我们将学习一种叫做逻辑回归 (**Logistic Regression**) 的算法，这是目前最流行使用最广泛的一种学习算法。

在分类问题中，我们尝试预测的是结果是否属于某一个类（例如正确或错误）。分类问题的例子有：判断一封电子邮件是否是垃圾邮件；判断一次金融交易是否是欺诈；判断一个肿瘤是恶性的还是良性的。

我们从二元的分类问题开始讨论。

我们将因变量(dependent variable)可能属于的两个类分别称为负向类（negative class）和正向类（positive class），则因变量$y\in { \{0,1\} \\}$ ，其中 0 表示负向类，1 表示正向类。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654272924488-f52b2222-525d-4bc5-939a-9d7a3f0291aa.png#clientId=u050ebd2a-4af4-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=268&id=u68a10fd7&name=image.png&originHeight=368&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=161328&status=done&style=none&taskId=ua5007ac0-4863-4d41-91f2-9ae7a17a205&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=282&id=XYsAN&originHeight=368&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=460)<br />如果我们要用线性回归算法来解决一个分类问题，$y$ 取值必须为 0 或者1，但实际假设函数的输出值可能远大于 1，或者远小于0。所以我们在接下来的要研究的算法就叫做逻辑回归算法，这个算法的性质是：它的输出值永远在0到 1 之间。

顺便说一下，逻辑回归算法是分类算法，有时候可能因为这个算法的名字中出现了“回归”使你感到困惑，但逻辑回归算法实际上是一种分类算法，它适用于标签  $y$ 取值离散的情况。

---

<a name="ae20g"></a>
##  假设表示

我们希望找出一个满足预测值要在0和1之间的假设函数。回顾在一开始提到的乳腺癌分类问题，我们可以用线性回归的方法求出适合数据的一条直线：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654273122852-2444d3cf-d291-4e75-89b8-30e83c93b9d6.png#clientId=u050ebd2a-4af4-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=116&id=u943ea2b9&name=image.png&originHeight=159&originWidth=433&originalType=binary&ratio=1&rotation=0&showTitle=false&size=57370&status=done&style=none&taskId=u7f3c6be2-75fa-4401-a587-4d1da85a23b&title=&width=314.90909090909093#crop=0&crop=0&crop=1&crop=1&height=141&id=duVxz&originHeight=159&originWidth=433&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=385)

根据线性回归模型我们只能预测连续的值，然而对于分类问题，我们需要输出0或1，我们可以预测：

当${h_\theta}\left( x \right)>=0.5$时，预测 $y=1$。

当${h_\theta}\left( x \right)<0.5$时，预测 $y=0$ 。

对于上图所示的数据，这样的一个线性模型似乎能很好地完成分类任务。假使我们又观测到一个非常大尺寸的恶性肿瘤，将其作为实例加入到我们的训练集中来，这将使得我们获得一条新的直线。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654273147754-2bb4d6de-8b30-4929-bd47-6dc82747ccfd.png#clientId=u050ebd2a-4af4-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=111&id=u146d138f&name=image.png&originHeight=153&originWidth=497&originalType=binary&ratio=1&rotation=0&showTitle=false&size=76082&status=done&style=none&taskId=udeae6769-62fb-4a72-a8f6-ee50d3367d4&title=&width=361.45454545454544#crop=0&crop=0&crop=1&crop=1&height=136&id=xzumC&originHeight=153&originWidth=497&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=443)

这时，再使用0.5作为阀值来预测肿瘤是良性还是恶性便不合适了。可以看出，线性回归模型，因为其预测的值可以超越[0,1]的范围，所以并不适合解决这个分类问题。

对于二分类任务，最理想的是"单位阶跃函数" (unit-step function) ，即<br />$y=\left\{\begin{matrix}
  0&z<0\\
  0.5&z=0 \\
  1&z>0
\end{matrix}\right.$

但单位阶跃函数不连续，因此不能作为假设函数。于是我们希望找到能在一定程度上近似单位阶跃函数的"替 <br />代函数" (surrogate function) ，并希望它单调可微。

对数几率函数(logistic function) 正是这样一个常用的替代函数：$\bold{g\left( z \right)=\frac{1}{1+{{e}^{-z}}}}$

我们引入这个新的模型即逻辑回归，**逻辑回归模型的假设是： **$\bold{h_\theta \left( x \right)=g\left(\theta^{T}x \right)}$<br />其中：
:::info
$x$：特征向量<br />$g$ ：逻辑函数（logistic function)是一个常用的S形函数（Sigmoid function）公式：$g\left( z \right)=\frac{1}{1+{{e}^{-z}}}$
:::

单位阶跃函数与对数几率函数图像<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654273819996-e003eee9-a28d-41d4-9ac3-b6dfb7f2d3ba.png#clientId=u050ebd2a-4af4-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=229&id=u7be238a7&name=image.png&originHeight=291&originWidth=540&originalType=binary&ratio=1&rotation=0&showTitle=false&size=43911&status=done&style=none&taskId=u5407600d-867b-42a8-bccd-62134e8bbd9&title=&width=424.727294921875#crop=0&crop=0&crop=1&crop=1&height=249&id=CSpp5&originHeight=291&originWidth=540&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=462)

**python**代码实现：

```python
import numpy as np
    
def sigmoid(z):
   return 1 / (1 + np.exp(-z))
```

$h_\theta \left( x \right)$的作用是，对于给定的输入变量，根据选择的参数计算输出变量=1的可能性（estimated probablity）即$h_\theta \left( x \right)=P\left( y=1|x;\theta \right)$<br />例如，如果对于给定的$x$，通过已经确定的参数计算得出$h_\theta \left( x \right)=0.7$，则表示有70%的几率$y$为正向类，相应地$y$为负向类的几率为1-0.7=0.3。

---

<a name="cdamJ"></a>
## 决策边界

决策边界(decision boundary)的概念能更好地帮助我们理解逻辑回归的假设函数在计算什么。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654274337462-9817743c-64ae-476e-984a-476db1442133.png#clientId=u050ebd2a-4af4-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=97&id=ubcb5c64a&name=image.png&originHeight=134&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=39028&status=done&style=none&taskId=uff0cd3be-7425-4b93-a547-cd60e4e4a84&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=104&id=W5PT2&originHeight=134&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=465)<br />在逻辑回归中，我们规定：

当${h_\theta}\left( x \right)>=0.5$时，预测 $y=1$。

当${h_\theta}\left( x \right)<0.5$时，预测 $y=0$ 。

根据上面绘制出的 **S** 形函数图像，因为$z={\theta^{T}}x$，则有：<br />$z={\theta^{T}}x>=0$  时，预测 $y=1$<br />$z={\theta^{T}}x<0$  时，预测 $y=0$

现在假设我们有一个模型：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654274403738-58fd8482-c792-49b5-bdb9-32e0488430c5.png#clientId=u050ebd2a-4af4-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=150&id=ubaae0d72&name=image.png&originHeight=206&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=59912&status=done&style=none&taskId=u9dcda3f8-9211-473d-92ae-3be988c180d&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=146&id=vnZGj&originHeight=206&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=425)

假设参数$\theta$ 是向量$\begin{bmatrix}
 3\\
 1\\1
\end{bmatrix}$。 则当$-3+{x_1}+{x_2} \geq 0$，即${x_1}+{x_2} \geq 3$时，模型将预测 $y=1$。<br />我们可以绘制直线${x_1}+{x_2} = 3$，这条线便是我们模型的分界线，将预测为1的区域和预测为 0的区域分隔开。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654274455294-dd607533-eb13-4b3d-ba5f-6c753463943d.png#clientId=u050ebd2a-4af4-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=267&id=ue3fc5d5e&name=image.png&originHeight=634&originWidth=1108&originalType=binary&ratio=1&rotation=0&showTitle=false&size=108529&status=done&style=none&taskId=ua42e2bdf-2081-4a8e-9483-8ba49176b4c&title=&width=466#crop=0&crop=0&crop=1&crop=1&height=267&id=ssxsG&originHeight=634&originWidth=1108&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=467)

假使我们的数据呈现这样的分布情况，怎样的模型才能适合呢？

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654274601471-b75e593f-e7a8-4199-89d1-16d7f8113060.png#clientId=u050ebd2a-4af4-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=130&id=u7701f526&name=image.png&originHeight=179&originWidth=241&originalType=binary&ratio=1&rotation=0&showTitle=false&size=59907&status=done&style=none&taskId=u27596fc5-d059-4f70-be86-a5e86a76276&title=&width=175.27272727272728#crop=0&crop=0&crop=1&crop=1&id=nE50s&originHeight=179&originWidth=241&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

因为需要用曲线才能分隔 $y=0$ 的区域和 $y=1$ 的区域，我们需要二次方特征：${h_\theta}\left( x \right)=g\left( {\theta_0}+{\theta_1}{x_1}+{\theta_{2}}{x_{2}}+{\theta_{3}}x_{1}^{2}+{\theta_{4}}x_{2}^{2} \right)$，取$\theta$=$\begin{bmatrix}
 -1\\
 0\\
0\\
1\\1
\end{bmatrix}$，则我们得到的判定边界恰好是圆点在原点且半径为1的圆形。事实上我们可以用非常复杂的模型来适应非常复杂形状的判定边界。

---

<a name="tSUVS"></a>
## 代价函数

应如何选择参数$\theta$来拟合逻辑回归模型？参数的优化目标或者说代价函数是是什么？

对于线性回归模型，我们定义的代价函数是所有模型误差的平方和。<br />理论上来说，我们也可以对逻辑回归模型沿用这个定义，但是问题在于，当我们将${h_\theta}\left( x \right)=\frac{1}{1+{e^{-\theta^{T}x}}}$带入到均方误差函数，我们得到的代价函数将是一个非凸函数（non-convexfunction）。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654275282934-b01f2cef-c196-4284-b19d-b925b60b154e.png#clientId=u050ebd2a-4af4-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=128&id=uc38a3f54&name=image.png&originHeight=176&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=55474&status=done&style=none&taskId=u63cf7306-3fcf-4429-ac45-946ba5d28a6&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=155&id=ylOKN&originHeight=176&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=528)

这意味着我们的代价函数有许多局部最小值，这将影响梯度下降算法寻找全局最小值。

线性回归的代价函数为：$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{1}{2}{{\left( {h_\theta}\left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}}$ 。<br />我们重新定义逻辑回归的代价函数为：$\bold{J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{{Cost}\left( {h_\theta}\left( {x}^{\left( i \right)} \right),{y}^{\left( i \right)} \right)}}$，其中

$Cost(h_{\theta}(x),y)=\left\{\begin{matrix}
-log(h_{\theta }(x))&，y=1\\
-log(1-h_{\theta }(x))&，y=0
\end{matrix}\right.$

:::info
$Cost\left( {h_\theta}\left( x \right),y \right)$函数的特点是：

- 当实际的  $y=1$ 且${h_\theta}\left( x \right)$也为 1 时误差为 0，当 $y=1$ 但${h_\theta}\left( x \right)$不为1时误差随着${h_\theta}\left( x \right)$变小而变大；
- 当实际的 $y=0$ 且${h_\theta}\left( x \right)$也为 0 时代价为 0，当$y=0$ 但${h_\theta}\left( x \right)$不为 0时误差随着 ${h_\theta}\left( x \right)$的变大而变大。
:::

<br />**代价函数的简化：**<br />可将 $Cost\left( {h_\theta}\left( x \right),y \right)$简化为：$Cost\left( {h_\theta}\left( x \right),y \right)=-y\times log\left( {h_\theta}\left( x \right) \right)-(1-y)\times log\left( 1-{h_\theta}\left( x \right) \right)$

带入代价函数得到：<br />$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$<br />即：$\bold{J\left( \theta  \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}}$

在得到这样一个代价函数以后，我们便可以用梯度下降算法来求得能使代价函数最小的参数了。算法为：

**Repeat** {<br />$\theta_j := \theta_j - \alpha \frac{\partial}{\partial\theta_j} J(\theta)$(**simultaneously update all** )<br />}

求导后得到：<br />**Repeat** {<br />$\theta_j := \theta_j - \alpha \frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {h_\theta}\left( \mathop{x}^{\left( i \right)} \right)-\mathop{y}^{\left( i \right)} \right)}}\mathop{x}_{j}^{(i)}$**(simultaneously update all** )<br />}

我们可以发现虽然逻辑回归的$J(\theta)$与线性回归中的代价函数定义不同，但求导之后的**形式是相同的**。之前用于线性回归监控梯度下降收敛的方法，在此处同样。适用另外，在运行梯度下降算法之前，进行特征缩放依旧是非常必要的。

**推导过程：**

:::info
$J\left( \theta  \right)=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}$<br />考虑：<br />${h_\theta}\left( {{x}^{(i)}} \right)=\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}}$

则：<br />${{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)$<br />$={{y}^{(i)}}\log \left( \frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}} \right)+\left( 1-{{y}^{(i)}} \right)\log \left( 1-\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}} \right)$<br />$=-{{y}^{(i)}}\log \left( 1+{{e}^{-{\theta^T}{{x}^{(i)}}}} \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1+{{e}^{{\theta^T}{{x}^{(i)}}}} \right)$

所以：<br />$\frac{\partial }{\partial {\theta_{j}}}J\left( \theta  \right)=\frac{\partial }{\partial {\theta_{j}}}[-\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( 1+{{e}^{-{\theta^{T}}{{x}^{(i)}}}} \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1+{{e}^{{\theta^{T}}{{x}^{(i)}}}} \right)]}]$<br />$=-\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\frac{-x_{j}^{(i)}{{e}^{-{\theta^{T}}{{x}^{(i)}}}}}{1+{{e}^{-{\theta^{T}}{{x}^{(i)}}}}}-\left( 1-{{y}^{(i)}} \right)\frac{x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}}]$<br />$=-\frac{1}{m}\sum\limits_{i=1}^{m}{{y}^{(i)}}\frac{x_j^{(i)}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}-\left( 1-{{y}^{(i)}} \right)\frac{x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}]$<br />$=-\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{{{y}^{(i)}}x_j^{(i)}-x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}+{{y}^{(i)}}x_j^{(i)}{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}}$<br />$=-\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{{{y}^{(i)}}\left( 1\text{+}{{e}^{{\theta^T}{{x}^{(i)}}}} \right)-{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}}x_j^{(i)}}$<br />$=-\frac{1}{m}\sum\limits_{i=1}^{m}{({{y}^{(i)}}-\frac{{{e}^{{\theta^T}{{x}^{(i)}}}}}{1+{{e}^{{\theta^T}{{x}^{(i)}}}}})x_j^{(i)}}$<br />$=-\frac{1}{m}\sum\limits_{i=1}^{m}{({{y}^{(i)}}-\frac{1}{1+{{e}^{-{\theta^T}{{x}^{(i)}}}}})x_j^{(i)}}$<br />$=-\frac{1}{m}\sum\limits_{i=1}^{m}{[{{y}^{(i)}}-{h_\theta}\left( {{x}^{(i)}} \right)]x_j^{(i)}}$<br />$=\frac{1}{m}\sum\limits_{i=1}^{m}{[{h_\theta}\left( {{x}^{(i)}} \right)-{{y}^{(i)}}]x_j^{(i)}}$
:::

Python代码实现：
```python
import numpy as np
    
def cost(theta, X, y):
  theta = np.matrix(theta)
  X = np.matrix(X)
  y = np.matrix(y)
  first = np.multiply(-y, np.log(sigmoid(X* theta.T)))
  second = np.multiply((1 - y), np.log(1 - sigmoid(X* theta.T)))
  return np.sum(first - second) / (len(X))
```

---

<a name="BX7Hi"></a>
## 高级优化
<br />	梯度下降并不是我们可以使用的唯一算法，还有其他一些算法，更高级、更复杂。这些方法可以用来计算代价函数$J\left( \theta  \right)$和偏导数项$\frac{\partial }{\partial {\theta_j}}J\left( \theta  \right)$两个项。**共轭梯度法 BFGS** (**变尺度法**) 和**L-BFGS** (**限制变尺度法**) 就是其中一些更高级的优化算法，它们需要有一种方法来计算 $J\left( \theta  \right)$，以及需要一种方法计算导数项，然后使用比梯度下降更复杂的算法来最小化代价函数。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654330005877-82648e80-e82b-402c-97e7-e65acf2e9ad5.png#clientId=ua2bff67a-9914-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=275&id=u21fe5042&name=image.png&originHeight=564&originWidth=1093&originalType=binary&ratio=1&rotation=0&showTitle=false&size=97736&status=done&style=none&taskId=ucdef3ee8-eea5-4589-82be-d802447c6e2&title=&width=532#crop=0&crop=0&crop=1&crop=1&height=249&id=OQ0nw&originHeight=564&originWidth=1093&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=482)<br />这三种算法有许多优点：

通常不需要手动选择学习率 $\alpha$，所以对于这些算法的一种思路是，给出计算导数项和代价函数的方法，你可以认为算法有一个智能的内部循环，而且，事实上，他们确实有一个智能的内部循环，称为**线性搜索**(**line search**)算法，它可以自动尝试不同的学习速率 $\alpha$，并自动选择一个好的学习速率 $\alpha$，因此它甚至可以为每次迭代选择不同的学习速率，那么你就不需要自己选择。这些算法实际上在做更复杂的事情，不仅仅是选择一个好的学习速率，所以它们往往最终比梯度下降收敛得快多了。

---

<a name="N04Cy"></a>
## 多分类学习：一对多

本节将谈到如何使用逻辑回归 (logistic regression)来解决多分类学习问题，具体来说是通过一个叫做"一对多" (**one-vs-all**) 的分类算法。

例：假如说你现在需要一个学习算法能自动地将邮件归类到不同的文件夹里，或者说可以自动地加上标签，那么，你也许需要一些不同的文件夹，或者不同的标签来完成这件事，来区分开来自工作的邮件、来自朋友的邮件、来自家人的邮件或者是有关兴趣爱好的邮件，那么，我们就有了这样一个分类问题：其类别有四个，分别用$y=1$、$y=2$、$y=3$、$y=4$ 来代表。

二元分类问题和多分类问题的数据集如下图所示<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654332519875-7087205e-1940-40e2-af60-207265a9cc43.png#clientId=u6885a1bf-cf38-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=259&id=ubc1170ef&name=image.png&originHeight=522&originWidth=990&originalType=binary&ratio=1&rotation=0&showTitle=false&size=42484&status=done&style=none&taskId=u5c0681fa-6037-4ddb-b384-dd546c5ac0b&title=&width=492#crop=0&crop=0&crop=1&crop=1&height=276&id=KPneU&originHeight=522&originWidth=990&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=524)

图示多分类问题用3种不同的符号来代表3个类别。问题是对于给出3个类型的数据集，我们如何得到一个学习算法来进行分类呢？

我们现在已经知道如何进行二元分类，可以使用逻辑回归将数据集一分为二为正类和负类。用一对多的分类思想，我们可以将其用在多类分类问题上。

:::info
**多分类学习的基本思路是** ：**“拆解法”**，即将多分类任务拆为若干个分类任务求解。最经典的拆分策略有三种.：“一对一" (One vs. One ，简称 **OvO**) ，“一对其余”(One vs. Rest ，简称 **OvR**) 和“多对多”(Many vs. Many，简称 **MvM**)。具体来说

- **OvO** 将为区分类别 $C_{j}$ 训练个分类器，该分类器把 中的 类样例作为正$C_{j}$ 类样例作为反例.在测试阶段，新样本将同时提交给所有分类器，于是我们将得到 $N(N -1)/2$ 个分类结果，最后把被预测得最多的类别作为最终分类结果。通常考虑各分类器的预测置信度，选择置信度最大的类别标记作为分类结果。
- **OvR** 则是每次将一个类的样例作为正例、所有其他类的样例作为反例来训练$N$个分类器。在测试时若仅有一个分类器预测为正类，则对应的类别标记作为最终分类结果
- **MvM** 是每次将若干个类作为正类，若干个其他类作为反类.显然， OvO、OvR是MvM 的特例。 MvM 的正、反类构造必须有特殊的设计，不能随意选取。详细方法见西瓜书p64.
:::

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654332419006-06b401d8-0e21-40ca-83e9-a345c6f0a52f.png#clientId=u6885a1bf-cf38-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=287&id=u19f33bdb&name=image.png&originHeight=394&originWidth=690&originalType=binary&ratio=1&rotation=0&showTitle=false&size=154764&status=done&style=none&taskId=u9516322d-3fe2-4084-9980-632328c85e7&title=&width=501.8181818181818#crop=0&crop=0&crop=1&crop=1&height=309&id=bwK9i&originHeight=394&originWidth=690&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=541)

下面详细介绍如何进行一对多的分类的“一对其余”OvR方法。（吴恩达视频内容）

现在我们有一个训练集，好比上图表示的有3个类别，我们用三角形表示 $y=1$，方框表示$y=2$，叉叉表示 $y=3$。我们下面要做的就是使用一个训练集，将其分成3个二元分类问题。

:::info

1. 首先我们将多个类中的一个类标记为正向类$(y=1)$，然后将其他所有类都标记为负向类，这个模型记作$h_θ^{(1)}(x)$。
1. 接着，类似地第我们选择另一个类标记为正向类（_y_=2），再将其它类都标记为负向类，将这个模型记作 $h_θ^{(2)}(x)$，依此类推。
1. 最后我们得到一系列的模型简记为：$h_θ^{(i)}(x)=P(y=i∣x;θ)$其中：$i=(1,2,3....k)$，我们将所有的分类机都运行一遍，然后对每一个输入变量，都选择最高可能性的输出变量，即选一个让$h_θ^{(i)}(x)$最大的$i$，即$\max_{i}h_θ^{(i)}(x)$
:::

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654330971278-2e5c46dd-2261-488f-9b00-29beadc7bf7d.png#clientId=u6885a1bf-cf38-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=372&id=u89e8f4fd&name=image.png&originHeight=674&originWidth=1151&originalType=binary&ratio=1&rotation=0&showTitle=false&size=114784&status=done&style=none&taskId=u9e856d09-0dd0-451f-8a9f-72ba15cab0c&title=&width=635#crop=0&crop=0&crop=1&crop=1&height=259&id=mgQZo&originHeight=674&originWidth=1151&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=443)

---

<a name="lDVp4"></a>
## 本章相关代码
[sklearn.linear_model.LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression)
```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0) # C为正则化项的倒数
lr.fit(X_train, y_train)
lr.predict_proba(X_test[0,:]) # 查看第一个测试样本属于各个类别的概率
plot_decision_regions(X_test, y_test, classifier=lr, test_idx=range(105,150))
plt.show()
```
![](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656592818050-30120da4-92e2-4f3b-8ce2-807620022532.png#clientId=u22cbec31-526a-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=237&id=u1f869850&originHeight=600&originWidth=800&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u6732b527-ed90-49fe-893f-90f6619438d&title=&width=316)

---

<a name="WUrwu"></a>
# 正则化(Regularization)
<a name="Ju1Ej"></a>
## 过拟合的问题

到现在为止，我们已经学习了几种不同的学习算法，包括线性回归和逻辑回归，它们能够有效地解决许多问题，但是当将它们应用到某些特定的机器学习应用时，会遇到**过拟合(over-fitting)**的问题，可能会导致它们效果很差。一种称为**正则化(regularization)**的技术，它可以改善或者减少过度拟合问题。

如果我们有非常多的特征，我们通过学习得到的假设可能能够非常好地适应训练集（代价函数可能几乎为0），但是可能会不能推广到新的数据。

下图是一个回归问题的例子：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654333578275-95a06fa8-5ebc-4fa8-97e9-bf3cc88ead10.png#clientId=u6885a1bf-cf38-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=239&id=u8690503e&name=image.png&originHeight=418&originWidth=1136&originalType=binary&ratio=1&rotation=0&showTitle=false&size=67116&status=done&style=none&taskId=ue24fe0fd-8fc9-45b1-aa9b-bc24c7e5a39&title=&width=650#crop=0&crop=0&crop=1&crop=1&height=203&id=AhvDU&originHeight=418&originWidth=1136&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=551)

- 第一个模型是一个线性模型，**欠拟合**，不能很好地适应我们的训练集；
- 第三个模型是一个四次方的模型，过于强调拟合原始数据，而丢失了算法的本质：预测新数据。我们可以看出，若给出一个新的值使之预测，它将表现的很差，是**过拟合**，虽然能非常好地适应我们的训练集但在新输入变量进行预测时可能会效果不好；
- 中间的模型似乎最合适。

:::info
**定义：**如果一味追求提高对训练数据的预测能力，所选模型的复杂度则往往会比真模型更高。这种现象称为**过拟合 Cover-fitting)** 。过拟合是指学习时选择的模型所包含的参数过多，以至出现这 模型对己知数据预测得很好，但对未知数据预测得很差的现象。（《统计学习方法》）
:::

分类问题中也存在这样的问题：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654333657430-5067c09a-c7cc-4f5e-adfb-db6e57750e74.png#clientId=u6885a1bf-cf38-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=316&id=uddd7617c&name=image.png&originHeight=560&originWidth=1149&originalType=binary&ratio=1&rotation=0&showTitle=false&size=152666&status=done&style=none&taskId=u2786458a-0759-4e31-9a7e-b5178258897&title=&width=649#crop=0&crop=0&crop=1&crop=1&height=228&id=TFGn3&originHeight=560&originWidth=1149&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=467)<br />$x$ 的次数越高，拟合的越好，但相应的预测的能力就可能变差。

**问题是，如果我们发现了过拟合问题，应该如何处理？**
:::info

1. **丢弃一些不能帮助我们正确预测的特征**。可以是手工选择保留哪些特征，或者使用一些模型选择的算法来帮忙（例如**PCA**）
2. **正则化**： 保留所有的特征，但是减少参数的大小（**magnitude**）。
3. **交叉验证：**重复地使用数据;把给定的数据进行切分，将切分的数据集组合为训练集与测试集，在此基础上反复地进行训练、测试以及模型选择（详见《统计学习方法》p24.）。
:::

---

<a name="F7DlZ"></a>
## 提前停止

对于梯度下降这一类迭代学习的算法，还有一个与众不同的正则化方法，就是在验证误差达到最小值时停止训练，该方法叫作**“提前停止法”（即“早停”Early Stopping）**。

训练的复杂模型（高阶多项式回归模型）,经过一轮一轮的训练，训练集上的预测误差（RMSE）自然不断下降，同样其在验证集上的预测误差也随之下降。但是，一段时间之后，验证误差停止下降反而开始回升。这说明模型开始**过拟合**训练数据。 <br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656649406128-50ef77fb-15ce-4450-8c67-ec7cb2b75dd1.png#clientId=u44911542-ece2-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=250&id=dfX6W&name=image.png&originHeight=390&originWidth=597&originalType=binary&ratio=1&rotation=0&showTitle=false&size=82703&status=done&style=none&taskId=uce0af864-791e-4d32-86f7-4899b65afc6&title=&width=382.08)<br />通过早期停止法，一旦验证误差达到最小值就立刻停止训练。这是一个非常简单而有效的正则化技巧

---

<a name="FIZjz"></a>
## 代价函数*

上面的回归问题中如果我们的模型是：${h_\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}^2}+{\theta_{3}}{x_{3}^3}+{\theta_{4}}{x_{4}^4}$，高次项会导致过拟合的产生。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654408963616-4010a1f2-d46b-41c9-a6c1-4762627b9b82.png#clientId=u6f29f900-8876-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=224&id=uQebq&name=image.png&originHeight=308&originWidth=498&originalType=binary&ratio=1&rotation=0&showTitle=false&size=39529&status=done&style=none&taskId=u12949910-8378-4d7e-ae3b-66760d87ade&title=&width=362.1818181818182#crop=0&crop=0&crop=1&crop=1&height=244&id=mNA5h&originHeight=308&originWidth=498&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=394)<br />由上图我们可以看到，当模型的复杂度增大时，训练误差会逐渐减小并趋向于 0； 而测试误差会先减小，达到最小值后又增大。当选择的模型复杂度过大时，过拟合现象就会发生。

:::info

- **结构风险最小化** (Cstructural risk minimization, SRM) 是为了防止过拟合而提出来的策略。结构风险最小化等价于**正则化** (regularization) 。结构风险是在经验风险上加上**表示模型复杂度的正则化项(regularizer )**或**罚项 (Cpenalty term)** 。
- **正则化**是结构风险最小化策略的实现。正则化项一般是**模型复杂度**的单调递增函数，模型越复杂，正则化值就越大。（《统计学习方法》）
- **正则化的作用是选择经验风险与模型复杂度同时较小的模型。**
:::

**正则化一般具有如下形式:**<br />$\min_{\theta} J(\theta)，\bold{J\left( \theta  \right)=\frac{1}{2m}[\sum\limits_{i=1}^{m}{{{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta_{j}^{2}}]}}$

其中$\lambda$ 又称为正则化参数（Regularization Parameter），可以为参数$\theta$的范数，此处为$L_{2}$范数，该式称为“岭回归(ridge regression)”。 $L1$范数时称为 **"Lasso"** 回归。注：根据惯例，我们不对${\theta_{0}}$ 进行惩罚。

正则化项可以取不同的形式；范数 (norm) 是常用的正则化项。其中$L_{2}$范数$\left \| \omega  \right \| _{2}$倾向于$\omega$的分量取值尽量均衡，即非零分量个数尽量稠密；而 $L_{0}$ 范数$\left \| \omega  \right \| _{0}$和$L_{1}$范数$\left \| \omega  \right \| _{1}$则倾向于$\omega$的分量尽量稀疏 即非零分量个数尽量少。

:::info
**“**$L_{1}$**范数和**$L_{2}$**范数正则化都有助于降低过拟合风险，但前者还会带来一个额外的好处：它比后者更易于获得"稀疏" (sparse) 解，即它求得的会有更少的非零分量.”**（——西瓜书）
:::

假设$x$仅有两个属性，要在平方误差与正则化项之间折中，则采用两种范式的优化目标函数的解$\omega_1$（即我们式中的$\theta_1$）和$\omega_2$（即我们的$\theta_2$），必定出现在下图中平方误差项等值线与正则化项等值线相交处。<br />	可以看到采用$L_{1}$范数时平方误差项等值线与正则化项等值线的常出现在坐标轴上，即$\omega_{1}$和$\omega_{2}$为0； 而在采用$L_{2}$范数时，两者交点常出现在某个象限中 即$ω_1$和$ω_2$均非 0；换言之，采用$L_1$范数比$L_2$范数更易于得到稀疏解。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654409865759-6cd805b3-4eb1-4b1b-b6b6-5fec8614f7a9.png#clientId=u6f29f900-8876-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=322&id=u558c9d8a&name=image.png&originHeight=443&originWidth=444&originalType=binary&ratio=1&rotation=0&showTitle=false&size=88227&status=done&style=none&taskId=u7a7bb5bf-7c72-494c-a021-9e969485258&title=&width=322.90909090909093#crop=0&crop=0&crop=1&crop=1&height=327&id=HN8D7&originHeight=443&originWidth=444&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=328)

经过正则化处理的模型与原模型的可能对比如下图所示：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654335937448-4c17549a-3a7d-42cb-96d3-68f37ac6bafb.png#clientId=u6885a1bf-cf38-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=177&id=u76da3629&name=image.png&originHeight=243&originWidth=300&originalType=binary&ratio=1&rotation=0&showTitle=false&size=68741&status=done&style=none&taskId=u9c855be4-29d7-4223-83c7-81bc8dd4fd7&title=&width=218.1818181818182#crop=0&crop=0&crop=1&crop=1&height=191&id=EKy7S&originHeight=243&originWidth=300&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=236)

如果选择的正则化参数$\lambda$ 过大，则会把所有的参数都最小化了，导致模型变成 ${h_\theta}\left( x \right)={\theta_{0}}$，也就是上图中红色直线所示的情况，造成欠拟合。<br />为什么增加的一项$\lambda =\sum\limits_{j=1}^{n}{\theta_j^{2}}$ 可以使$\theta$的值减小呢？<br />因为如果我们令$\lambda$的值很大的话，为了使代价函数尽可能的小，所有的$\theta$的值（不包括${\theta_{0}}$）都会在一定程度上减小。但若$\lambda$的值过大，那么$\theta$(不包括${\theta_{0}}$)的值都会趋近于0，这样我们所得到的只能是一条平行于$x$轴的直线。所以对于正则化，我们要取一个合理的 $\lambda$ 的值，这样才能更好的应用正则化。

回顾一下代价函数，为了使用正则化，让我们把这些概念应用到到线性回归和逻辑回归中去，那么就能避免过度拟合了。

---

<a name="QbMLk"></a>
## 正则化线性回归

对于线性回归的求解，我们之前推导了两种学习算法：一种基于梯度下降，一种基于正规方程。

正则化线性回归的代价函数为：

$J\left( \theta  \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{[({{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta _{j}^{2}})]}$

如果我们要使用梯度下降法令这个代价函数最小化（$\theta_0$未进行正则化），所以梯度下降算法将分两种情形：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654336564345-326bc7b1-578d-4927-ae03-670881c2f0e6.png#clientId=u6885a1bf-cf38-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=195&id=u7796502c&name=image.png&originHeight=268&originWidth=638&originalType=binary&ratio=1&rotation=0&showTitle=false&size=32633&status=done&style=none&taskId=u974a2a72-2933-462f-90ad-d3e7a8bbd0f&title=&width=464#crop=0&crop=0&crop=1&crop=1&height=175&id=poihO&originHeight=268&originWidth=638&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=416)

对上面的算法中$j=1,2,...,n$时的更新式子进行调整可得：<br />${\theta_j}:={\theta_j}(1-a\frac{\lambda }{m})-a\frac{1}{m}\sum\limits_{i=1}^{m}{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})x_{j}^{\left( i \right)}}$

可以看出，正则化线性回归的梯度下降算法的变化在于，每次都在原有算法更新规则的基础上令$\theta$值减少了一个额外的值。

我们同样也可以利用**正规方程来求解正则化线性回归模型**，方法如下所示：

$\theta=\begin{pmatrix}
X^{T}X+\lambda\begin{bmatrix}
  0&  &  &  & \\
  &  1&  &  & \\
  &  &  .&  & \\
  &  &  &  .& \\
  &  &  &  &1
\end{bmatrix} 
\end{pmatrix}^{-1}X^{T}y$

式中的矩阵为 $(n+1)*(n+1)$维矩阵


---

<a name="AVWYF"></a>
## 正则化的逻辑回归模型

针对逻辑回归问题，我们在之前的课程已经学习过两种优化算法：我们首先学习了使用梯度下降法来优化代价函数$J\left( \theta  \right)$，接下来学习了更高级的优化算法，这些高级优化算法需要你自己设计代价函数$J\left( \theta  \right)$。

对于逻辑回归，代价函数增加一个正则化的表达式，得到正则化的代价函数：

$J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{[-{{y}^{(i)}}\log \left( {h_\theta}\left( {{x}^{(i)}} \right) \right)-\left( 1-{{y}^{(i)}} \right)\log \left( 1-{h_\theta}\left( {{x}^{(i)}} \right) \right)]}+\frac{\lambda }{2m}\sum\limits_{j=1}^{n}{\theta _{j}^{2}}$

要最小化该代价函数，通过求导，得出梯度下降算法为：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654337092607-bb5fcf41-49c4-489b-97bb-ac0fd19f3604.png#clientId=u6885a1bf-cf38-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=200&id=bwLRb&name=image.png&originHeight=275&originWidth=637&originalType=binary&ratio=1&rotation=0&showTitle=false&size=32736&status=done&style=none&taskId=u9e00e2d9-002c-42f2-a8e3-f9c88f21b1b&title=&width=463.27272727272725#crop=0&crop=0&crop=1&crop=1&height=174&id=r3KRm&originHeight=275&originWidth=637&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=402)

注意：
:::info

1. 虽然正则化的逻辑回归中的梯度下降和正则化的线性回归中的表达式看起来一样，但由于两者的${h_\theta}\left( x \right)$不同所以实际还是有很大差别。
1. ${\theta_{0}}$不参与其中的任何一个正则化。 ${\theta_{0}}$的更新规则与其他参数不同。
:::

---

<a name="ZU5j2"></a>
## 本章相关代码
[sklearn.linear_model.Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html?highlight=ridge#sklearn.linear_model.Ridge)<br />[sklearn.linear_model.SGDRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html?highlight=sgdregressor#sklearn.linear_model.SGDRegressor)<br />[sklearn.linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html?highlight=lasso#sklearn.linear_model.Lasso)<br />[Keras documentation: EarlyStopping](https://keras.io/api/callbacks/early_stopping/)

```python
from sklearn.linear_model import Ridge

# 使用AndreLouis Cholesky 矩阵分解技术（闭式解）的岭回归
ridge_reg = Ridge(alpha=1, solver="cholesky", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict(X_new) # 查看预测值
```
```python
from sklearn.linear_model import Ridge

# solver选择"sag"
ridge_reg = Ridge(alpha=1, solver="sag", random_state=42)
ridge_reg.fit(X, y)
ridge_reg.predict(X_new)
```
```python
from sklearn.linear_model import SGDRegressor

# penalty选项为罚项的类型，l2则意为正则项为l2范数的一半
sgd_reg = SGDRegressor(penalty="l2", max_iter=1000, tol=1e-3, random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict((X_new) # 查看预测值
```
```python
sgd_reg = SGDRegressor(penalty="elasticnet", max_iter=1000, tol=1e-3, random_state=42)
sgd_reg.fit(X, y.ravel())
sgd_reg.predict((X_new) # 查看预测值
```
```python
from sklearn.linear_model import Lasso

lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X, y)
lasso_reg.predict(X_new) # 查看预测值 
```
```python
from copy import deepcopy
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(max_iter=1, tol=-np.infty, warm_start=True,
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)

minimum_val_error = float("inf")
best_epoch = None
best_model = None
for epoch in range(1000):
    sgd_reg.fit(X_train, y_train)  # continues where it left off
    y_val = sgd_reg.predict(X_val)
    val_error = mean_squared_error(y_val, y_val_predict)
    if val_error < minimum_val_error:
        minimum_val_error = val_error
        best_epoch = epoch
        best_model = deepcopy(sgd_reg) # 记录下最佳模型
```
```python
from sklearn.linear_model import SGDRegressor

# 设置early_stopping = 'true'
sgd_reg = SGDRegressor(max_iter=1, early_stopping = 'true'
                       penalty=None, learning_rate="constant", eta0=0.0005, random_state=42)
 ... 
```
```python
# 导入模块
import numpy as np
from keras import callbacks
from keras.models import Sequential
from keras.layers import Dense

# 早停法
callback = callbacks.EarlyStopping(monitor='loss', patience=3)

# 以简单模型为例
model = Sequential()
model.add(Dense(10)) # 10个全连接层
model.compile(optimizer='SGD', loss='mse')

history = model.fit(X_train,y_train,
                    epochs=100, batch_size=1, callbacks=[callback],
                    verbose=2)

epoch_len=len(history.history['loss'])
print(epoch_len)  # 输出训练停止的代数
```

---


<a name="B8lwo"></a>
# 决策树 (Decision Tree)
<a name="K9tSJ"></a>
## 概述

决策树是一种基本的分类与回归方法。决策树学习通常包括3个步骤：特征选择、决策树的生成、剪枝。

**决策树模型：**
:::info
用决策树分类，从根结点开始，对实例的某特征进行测试（决策中每个判定问题都是对某个特征的“测试”），根据测试结果，将实例分配到其子结点；这时，每个子结点对应着该特征的一个取值。如此递归地对实例进行测试并分配，直至达到叶结点。最后将实例分到叶结点的类中。
:::

由决策树的个别结点到叶结点的每一条路径构建一条规则；路径上的内部结点对应着规则的条件，而叶结点的类对应着规则的结论，即每个测试的结果或是导出最终结论，或是导出进一步的判定问题。

决策树学习的损失函数通常是正则化的极大似然函数。决策树的学习目标是一损失函数为目标函数的最小化。


从概率的角度来看，决策树还表示给定特征条件下类的条件概率分布。这一条件概率分布定义在特征空间的一个划分上。假设$X$为表示特征的随机变量$Y$为表示类的随机变量，那么这个条件概率分布可以表示为$P(Y|X)$。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656342779012-1527d269-8f78-4fce-8f0a-701f306ba96c.png#clientId=ubf71af2a-0695-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=412&id=wKLI3&name=image.png&originHeight=811&originWidth=895&originalType=binary&ratio=1&rotation=0&showTitle=false&size=193938&status=done&style=none&taskId=uc5a9e189-76e0-483a-9bf6-65882232d69&title=&width=454.79998779296875)

决策树分类时将该节点的实例强行分到条件概率大的那一类中。

---

<a name="X6abE"></a>
## 特征选择

特征选择在于选取对训练数据具有分类能力的特征，即如何选择最优划分特征来划分特征空间。通常随着划分过程的进行，我们希望决策树的分支结点所包含的样本尽可能属于同一类，即结点的“**纯度**”**(purity)** 越来越高。

常用的选择准则有：**信息增益、信息增益率、基尼系数**

:::info
**“信息熵” (information entropy)** 是表示随机变量不确定性的度量，是度量样本集合纯度最常用的一种指标。$Ent(D)$的值越小，则$D$的纯度越高。

$\operatorname{Ent}(D)=-\sum_{k=1}^{|\mathcal{Y}|} p_{k} \log _{2} p_{k}$<br />$p_k$是当前样本集合中第$k$类样本所占比例（或概率），$k=1,2,...,|\mathcal{Y}|$
:::

**条件熵(conditional entropy)**$Ent(Y|X)$表示在己知随机变$X$的条件下随机变量$Y$的不确定性。

随机变量$X$给定的条件下随机变量$Y$的条件熵定义为$X$给定条件下$Y$的条件概率分布的熵对$X$的数学期望：<br />$Ent(Y \mid X)=\sum_{i=1}^{n} p_{i} Ent\left(Y \mid X=x_{i}\right)$

当熵和条件熵中的概率由数据估计（特别是极大似然估计）得到时，所对应的熵与条件熵分别称为“**经验熵”(empirical entropy)** 和**“经验条件熵”(empirical conditional entropy)**

此时训练集$D$关于特征$a$的信息熵为：$Ent(D)=-\sum_{k=1}^{|\mathcal{Y}|} \frac{\left|C_{k}\right|}{|D|} \operatorname{log_2}(\frac{C_k}{|D|})$，其中设有$|\mathcal{Y}|$个类$C_k$；$k=1,..,|\mathcal{Y}|$；$|C_k|$为属于类$C_k$的样本个数，$\sum_{k=1}^{K}{|C_k|}=|D|$

:::info
**“信息增益”(information gain)**表示得知$X$的信息而使$Y$的信息不确定性减少的程度。

特征$a$对数据集$D$的信息增益定义为$D$的经验熵$Ent(D)$与给定特征$a$条件下的经验条件熵$Ent(D|a)$之差<br />$Gain(D,a)=Ent(D)-Ent(D|a)$

信息增益表示由于特征$a$而使得对数据集$D$的分类不确定性减少的程度。信息增益越大则意味着使用属性$a$来进行划分所获得的纯度提升越大。

信息增益公式的计算可以推导为：$\operatorname{Gain}(D, a)=\operatorname{Ent}(D)-\sum_{v=1}^{V} \frac{\left|D^{v}\right|}{|D|} \operatorname{Ent}\left(D^{v}\right)$，其中$V$为离散特征$a$可能取值的个数，$D^v$为在第$v$个分支上在特征$a$上取值为$a^v$的样本。

由此公式中$\frac{|D^{v}|}{|D|}$可以看到**样本数越多的分支结点影响越大。**
:::

:::info
**“信息增益率”(information gain ratio)**$Grain_ratio(D,a)$定义为其信息增益$Grain(D,a)$与训练数据集$D$关于特征$a$的熵$IV(a)$之比。

$Grain_ratio(D,a)=\frac{Gain(D,a)}{IV(a)}$

其中$IV(a)=-\sum_{v=1}^{V} \frac{\left|D^{v}\right|}{|D|} \operatorname{log_2}(\frac{|D^{v}|}{|D|})$称为属性$a$的固有值

以信息增益作为划分，存在偏向于选择取值较多的特征得问题，此时则可以采用信息增益率。
:::

:::info
**“基尼系数”(Gini index)**反映了从数据集$D$随机抽取两个样本，其类别标记不一致的概率，基尼系数越小，则数据集$D$纯度越高。<br />$Gini(D)=\sum_{k=1}^{|\mathcal{Y}|}\sum_{k \ne k'}p_kp_{k'}=\sum_{k=1}^{|\mathcal{Y}|}p_k(1-p_k)=1-\sum_{k=1}^{|\mathcal{Y}|}p_k^2$

在二分类任务中，基尼系数与熵之半曲线很接近，都可以近似的表示分类误差率，因此可以使用基尼系数来作为分类任务的准则。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656381680145-1c3cb9f2-8cb1-4844-b5bb-12af3eb137b1.png#clientId=u95253bf8-e455-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=199&id=uf383d100&name=image.png&originHeight=349&originWidth=604&originalType=binary&ratio=1&rotation=0&showTitle=false&size=95277&status=done&style=none&taskId=ua47a1605-a383-4e43-9661-a249cf67293&title=&width=344.55999755859375)
:::

---

<a name="lMSVt"></a>
## 决策树的生成

<a name="tF1TL"></a>
### 决策树生成过程：
![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656351066169-a1c5cecc-0aa0-4345-a73d-c882e08215e0.png#clientId=ubf71af2a-0695-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=360&id=u9ed5d274&name=image.png&originHeight=562&originWidth=800&originalType=binary&ratio=1&rotation=0&showTitle=false&size=196710&status=done&style=none&taskId=u42662646-5712-4718-bf99-f478d1059e6&title=&width=512)<br />显然决策树生成是一种递归算法，而划分的关键是第8行，即如何选择划分属性。

有三种情况会导致递归返回：

1. 当前结点包含的样本全属于同一类别。
1. 当前特征（属性）集为空，或是所有样本在所有属性上取值相同，无法划分。此时我们标记当前结点为叶结点，并将其类别设定为该结点所含样本最多的类别。
1. 当前节点包含的样本集合为空，不能划分。此时把当前结点标记为叶结点，并将其类别设定为其父节点所含样本最多的类别。

<a name="D01rV"></a>
### ID3 学习算法


ID3 学习算法以**信息增益**来进行决策树的特征划分选择，即生成过程第8行的选择特征的目标为$a_*=\arg max_{a \in A} Gain(D,a)$


<a name="KZnRm"></a>
### C4.5 决策树算法


C4.5算法为避免信息增准则对可取值数目较多的特征有所偏好，而使用**信息增益率**来选择最优化分特征。

C4.5 算法并不是直接选择增益率最大的候选划分特征，而是先从候选划分特征这种找出信息增益高于平均水平的特征，再从中选择增益率最高的。


<a name="BInmd"></a>
### CART决策树

CART算法是在给定输入随机变量$X$的条件下输出随机变量$Y$的条件概率分布。使用**基尼系数**来选择最优化分特征，即$a_*=\arg max_{a \in A} Gain_index(D,a)$.

CART生成分类树计算停止的条件是结点中的样本个数小于预定阈值，或者样本集的基尼系数小于预定阈值，或者没有特征集为空。

CART 不仅可以生成上述分类树，还可以生成最小二乘回归树，详见《统计学习方法》p82

---

<a name="V98GM"></a>
## 剪枝

决策树的剪枝往往通过极小化决策树整体损失函数来实现：$\min C_{\alpha}(T)=C(T)+\alpha| T|$

其中$C(T)=\sum_{t=1}^{|T|} N_{t} H_{t}(T)=-\sum_{t=1}^{|T|} \sum_{k=1}^{K} N_{t k} \log \frac{N_{t k}}{N_{t}}$，$|\mathrm T|$为叶结点个数，$N_t$为叶结点$t$的样本点个数，$N_{tk}$为其中$k$类的样本点个数

式子中$C(T)$表示模型对训练数据的预测误差，$|T|$表示模型复杂度，$\alpha$控制两者间的影响。剪枝就是当$\alpha$确定后，选择损失函数最小的模型。

<a name="MhHY6"></a>
### 预剪枝


预剪枝是指在决策树生成过程中，对每个结点在划分前先进行估计，即边建边剪。若当前结点的划分不能带来决策树泛化性能提升（用验证集判定），则停止划分并将当前结点标记为叶结点。

判断时可采用精度，信息增益量等作为评价指标。

例：西瓜数据集划分的训练集（上部）与验证集（下部）<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656382749879-f2ec22ad-44f7-47ef-a0a3-fbc5118fb378.png#clientId=u95253bf8-e455-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=377&id=uc83730c0&name=image.png&originHeight=696&originWidth=881&originalType=binary&ratio=1&rotation=0&showTitle=false&size=319932&status=done&style=none&taskId=u5db93b34-3f38-4db2-b693-efff18f1440&title=&width=477.83001708984375)<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656382604958-bd1d8539-019c-468b-a35c-05e73584a9a5.png#clientId=u95253bf8-e455-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=252&id=u2b27ddad&name=image.png&originHeight=520&originWidth=938&originalType=binary&ratio=1&rotation=0&showTitle=true&size=173381&status=done&style=none&taskId=u6b512453-7c55-48e8-9332-4461b35ea91&title=%E6%9C%AA%E5%89%AA%E6%9E%9D%E5%86%B3%E7%AD%96%E6%A0%91&width=455.32000732421875 "未剪枝决策树")

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656382635057-4661bbed-2268-4ddd-8dd8-21e7019b376b.png#clientId=u95253bf8-e455-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=227&id=u9b7020a1&name=image.png&originHeight=407&originWidth=870&originalType=binary&ratio=1&rotation=0&showTitle=true&size=184224&status=done&style=none&taskId=u7305e990-6766-4d5c-8bce-00deb9c54f2&title=%E9%A2%84%E5%89%AA%E6%9E%9D%E5%86%B3%E7%AD%96%E6%A0%91&width=485.79998779296875 "预剪枝决策树")


预剪枝基于“贪心”，虽然禁止一些分支展开降低了过拟合风险，但有些分支虽然当前划分不能提升泛化能力，但其基础上进行的后续划分却有可能显著提高性能，但预剪枝会禁止这些结点展开，所以预剪枝决策树可能会有欠拟合风险。

<a name="ylB87"></a>
### 后剪枝


后剪枝则是先从训练集生成一棵完整的决策树， 然后自底向上地对非叶结点进行考察，建完再剪。若将该结点对应的子树替换为叶结点能带来决策树泛化性能提升，则将该子树替换为叶结点。

例：基于上节西瓜数据集<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656382972887-9be5db39-1c40-4bce-9556-dbb9b413178b.png#clientId=u95253bf8-e455-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=246&id=u83047d51&name=image.png&originHeight=450&originWidth=1061&originalType=binary&ratio=1&rotation=0&showTitle=false&size=212212&status=done&style=none&taskId=uef804bb1-c7bd-4e1f-8f2f-5814538eef4&title=&width=580.0299682617188)

一般情形下后剪枝决策树的欠拟合风险很小，泛化能力往往优于预剪枝决策树，但时间开销通常较大。

---

<a name="SyuzV"></a>
## 连续值处理（C4.5的方案）

由于连续属性可取值数目不再有限，因此不能直接根据连续属性的可取值来对结点进行划分。最简单的策略是采用**“二分法”（bi-partition）**对连续属性进行划分。

给定样本集$D$和连续特征$a$，假定$a$在$D$上出现$n$个不同的取值，将这些值从小到大进行排序，记为$\{a^1,..., a^n \}$. 划分点$t$可将$D$分为子集$D_t^-$和$D_t^+$ ，其中$D_t^-$包含那些在属性$a$上取值不大于$t$样 本，$D_t^+$同理。

显然对相邻的属性取值$a^i$和$a^{i+1}$来说$t$在区间$[a^i,a^{i+1})$中取任意值所产生的划分结果相同。因此对于连续属性$a$我们可以考察包含$n-1$个元素的候选划分点集合<br />$T_{a}=\left\{\frac{a^{i}+a^{i+1}}{2} \mid 1 \leqslant i \leqslant n-1\right\}$

即把区间中位点作为划分点，然后则可以像离散点一样考察，选取最优的划分点。

例：连续属性值的信息增益公式：<br /> $Gain(D,a)=\max_{t \in T_a}Gain(D,a,t)=\max _{t \in T_{a}} \operatorname{Ent}(D)-\sum_{\lambda \in\{-,+\}} \frac{\left|D_{t}^{\lambda}\right|}{|D|} \operatorname{Ent}\left(D_{t}^{\lambda}\right)$<br />其中$Gain(D,a,t)$是样本集$D$基于划分点$t$二分后的信息增益，所以我们就选择使其最大化的划分点。

---

<a name="jmyw0"></a>
## 缺失值处理（C4.5的方案）

现实任务中常会遇到不完整样本，即样本的某些属性值缺失，如下：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656395545254-eda51b9a-4a3e-4304-93e6-b870bdc1202e.png#clientId=u95253bf8-e455-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=304&id=u105a3928&name=image.png&originHeight=613&originWidth=879&originalType=binary&ratio=1&rotation=0&showTitle=false&size=304359&status=done&style=none&taskId=u510a3bc4-f49e-4d07-a181-4ecdd6701b5&title=&width=436.55999755859375)


1. **属性值缺失情况下进行划分属性选择**

给定训练集$D$和属性$a$。令$\tilde D$表示$D$中在属性$a$上没有缺失值的样本子集，其他符号与前几节类似。假定我们为每个样本$x$赋予一个权重$w_x$并定义：

$\begin{aligned}
&无缺样本比例：\rho=\frac{\sum_{\boldsymbol{x} \in \tilde{D}} w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in D} w_{\boldsymbol{x}}}\\
&无缺样本中第k类比例：\tilde{p}_{k}=\frac{\sum_{\boldsymbol{x} \in \tilde{D}_{k}} w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in \tilde{D}} w_{\boldsymbol{x}}} \quad(1 \leqslant k \leqslant|\mathcal{Y}|)\\
&无缺样本在a上取a^v比例：\tilde{r}_{v}=\frac{\sum_{\boldsymbol{x} \in \tilde{D}^{v}} w_{\boldsymbol{x}}}{\sum_{\boldsymbol{x} \in \tilde{D}} w_{\boldsymbol{x}}} \quad(1 \leqslant v \leqslant V)
\end{aligned}$

则信息增益的计算式可推广为：$Gain(D,a)=\rho \times Gain(\tilde{D},a )=\rho \times\left(\operatorname{Ent}(\tilde{D})-\sum_{v=1}^{V} \tilde{r}_{v} \operatorname{Ent}\left(\tilde{D}^{v}\right)\right)$<br />其中$Ent(\tilde{D})=-\sum_{k=1}^{|\mathcal{Y}|}\tilde{p_k}\log_2 \tilde{p_k}$


2. **样本在给定划分属性上值缺失情况下的划分**

若样本$x$在划分属性$a$上的取值己知，则将$x$划入与其取值对应的子结点，且样本权值在于结点中保持为$w_x$.

若样本$x$在划分属性$a$上的取值未知，则将$x$同时划入所有子结点，并且样本权值在与属性值$a^v$对应的子结点中调整为$\tilde{r_v} \cdot w_x$.直观地看，这就是让同一个样本以不同的概率划入到不同的子结点中去.

---

<a name="pibOd"></a>
## 算法对比

**ID3算法优缺点**：

- 不能对连续数据进行处理，只能通过连续数据离散化进行处理；
- 采用信息增益进行数据分裂容易偏向取值较多的特征，准确性不如信息增益率；
- 缺失值不好处理。

**C4.5算法优缺点**：

- 产生的规则易于理解；准确率较高；实现简单；
- 对数据进行多次顺序扫描和排序，效率较低；
- 只适合小规模数据集，需要将数据放到内存中

---

<a name="TjHkO"></a>
## 本章相关代码
[sklearn.tree.DecisionTreeClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html?highlight=decisiontreeclassifier#sklearn.tree.DecisionTreeClassifier)<br />[sklearn.tree.export_graphviz](https://scikit-learn.org/stable/modules/generated/sklearn.tree.export_graphviz.html?highlight=export_graphviz#sklearn.tree.export_graphviz)<br />[sklearn.ensemble.RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforestclassifier#sklearn.ensemble.RandomForestClassifier)
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:] # petal length and width
y = iris.target

# scikit=-learn使用CART算法
# 设置criterion参数，以基尼系数（默认）作为评判准则
# 可以设置不同的停止条件
tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42,criterion='gini')
tree_clf.fit(X, y)

tree_clf.predit(X_new) # 预测类
tree_clf.predict_prob(X_new) # 预测可能时各个类的概率
```
```python
from graphviz import Source # 需下载并配置环境变量
from sklearn.tree import export_graphviz

export_graphviz(
        tree_clf,
        out_file=os.path.join(IMAGES_PATH, "iris_tree.dot"),
        feature_names=iris.feature_names[2:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
    )

Source.from_file(os.path.join(IMAGES_PATH, "iris_tree.dot"))
```
```python
from sklearn.ensemble import RandomForestClassifier

# 使用由五棵树组成的随机森林
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)
```

---

<a name="Vyibx"></a>
# 神经网络(Neural Networks)

<a name="hokcJ"></a>
## 非线性假设

无论是线性回归还是逻辑回归都有这样一个缺点，即：当特征太多时，计算的负荷会非常大。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654482052301-e914cbee-87a8-4417-9e85-bfb6b4da88c9.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=140&id=u9fb7e2fd&name=image.png&originHeight=192&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=115273&status=done&style=none&taskId=u9ff19809-d7e7-4e2c-ba2f-22f730b417f&title=&width=436.3636363636364#crop=0&crop=0&crop=1&crop=1&height=147&id=EkThy&originHeight=192&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=460)

当我们使用$x_1$, $x_2$ 的多次项式进行预测时，我们可以应用的很好。假设我们有非常多的特征，例如大于100个变量，我们希望用这100个特征来构建一个非线性的多项式模型，结果将是数量非常惊人的特征组合，即便我们只采用两两特征的组合$(x_1x_2+x_1x_3+x_1x_4+...+x_2x_3+x_2x_4+...+x_{99}x_{100})$，我们也会有接近5000个组合而成的特征。这对于一般的逻辑回归来说需要计算的特征太多了。<br />普通的逻辑回归模型，不能有效地处理这么多的特征，这时候我们需要神经网络。

---

<a name="O6032"></a>
## 模型表示

（建议看西瓜书）<br />神经网络模型建立在很多神经元之上，每一个神经元又是一个个学习模型。这些神经元（也叫**激活单元activation unit**）采纳一些特征作为输出，并且根据本身的模型提供一个输出。下图是一个以逻辑回归模型作为自身学习模型的神经元示例，在神经网络中，参数又可被成为**权重（weight）**。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654482144713-ec0ba9de-4dfc-46a3-a675-c83c83ff9f01.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=218&id=ue2298aad&name=image.png&originHeight=632&originWidth=1146&originalType=binary&ratio=1&rotation=0&showTitle=false&size=95910&status=done&style=none&taskId=u275536ce-5073-4813-92ab-8f96c9d5a48&title=&width=395#crop=0&crop=0&crop=1&crop=1&height=280&id=tjOt2&originHeight=632&originWidth=1146&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=507)

任何优化算法都需要一些初始的参数。到目前为止我们都是初始所有参数为0，这样的初始方法对于逻辑回归来说是可行的，但是对于神经网络来说是不可行的。 如果我们令所有的初始参数都为0，这将意味着我们第二层的所有激活单元都会有相同的值。同理，如果我们初始所有的参数都为一个非0的数，结果也是一样的。

**我们通常初始参数为**$±ε$**之间的随机值，**$ε$**为一个非常小的值**


:::info
**M-P	神经元模型：**<br />神经元接收到来自其他神经元传递过来的输入信号，这些输入信号通过带权重的连接(connection) 进行传递，神经元接收到的总输入值将与神经元的**阀值**（也就是**偏差**$bias$）进行比较，然后通过"激活函数" (activation function )处理以产生神经元的输出.（激活函数常用$sigmoid$函数）<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654482256722-a3c5853d-5554-48d1-a1e4-96caa42985fe.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=173&id=inYDB&name=image.png&originHeight=312&originWidth=685&originalType=binary&ratio=1&rotation=0&showTitle=false&size=94795&status=done&style=none&taskId=u1cb21aeb-7f56-47ea-91b4-faec3f9ec99&title=&width=379.18182373046875#crop=0&crop=0&crop=1&crop=1&height=201&id=Er9yt&originHeight=312&originWidth=685&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=441)	![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654484721301-be389b8f-dd40-4783-8b2e-534443911809.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=203&id=u033adff0&name=image.png&originHeight=350&originWidth=373&originalType=binary&ratio=1&rotation=0&showTitle=false&size=39823&status=done&style=none&taskId=uf4eab4e6-8b40-47cb-9598-e8fec83837e&title=&width=216.27273559570312#crop=0&crop=0&crop=1&crop=1&height=203&id=Ezzgb&originHeight=350&originWidth=373&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=216)<br />注意：$y=f\left(\sum_{i=1}^{n} w_{i} x_{i}-\theta\right)$
:::

:::info
**感知机 (Perceptron)** 由两层神经元组成，输入层接收外界输入信号后传递给输出层， 输出层是 M-P 神经元，亦称“阈值逻辑单元”(threshold logic unit)。感知机学习规则非常简单，对训练样例$(x,y)$，若当前感知机的输出为$\hat{y}$则感知机权重将按如下调整，其中$\eta$为学习率(learning rate).<br />$\begin{array}{c}
w_{i} \leftarrow w_{i}+\Delta w_{i} \\
\Delta w_{i}=\eta(y-\hat{y}) x_{i}
\end{array}$<br />（多层神经网络的学习算法详见 8.5误差反向传播）
:::

我们设计出了类似于神经元的神经网络，效果如下：（此种神经元之间不存在同层连接， 也不存在跨层连接 这样的神经经网络结构通常称为“多层前馈神经网” (multi-layer feedforward neural networks）<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654482189159-0a8df684-96bb-4c9a-a8ab-8befe4ac46e9.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=168&id=u804c77da&name=image.png&originHeight=197&originWidth=400&originalType=binary&ratio=1&rotation=0&showTitle=false&size=63005&status=done&style=none&taskId=u32a3dc56-7bf7-42ee-b37d-450d745dea4&title=&width=340.9090881347656#crop=0&crop=0&crop=1&crop=1&height=177&id=Rxqrl&originHeight=197&originWidth=400&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=360)

其中$x_1$, $x_2$, $x_3$是输入单元（**input units**），我们将原始数据输入给它们。$a_1$, $a_2$, $a_3$是中间单元，它们负责将数据进行处理，然后呈递到下一层。最后是输出单元，它负责计算${h_\theta}\left( x \right)$。

神经网络模型是许多逻辑单元按照不同层级组织起来的网络，每一层的输出变量都是下一层的输入变量。下图为一个3层的神经网络，第一层成为**输入层（Input Layer）**，最后一层称为**输出层（Output Layer）**，中间一层成为**隐藏层（Hidden Layers）**。我们为每一层都增加一个**偏差单位**（**bias unit**）：

:::info
神经网络的学习过程，就是根据训练数据来调整神经元之间的 “连接权” (connection weight) 以及每个功能神经元的阈值。（西瓜书）
:::

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654482390443-36bc9aa1-383a-4b5e-8937-a01c6fe08c2f.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=266&id=ue8f4249d&name=image.png&originHeight=621&originWidth=1022&originalType=binary&ratio=1&rotation=0&showTitle=false&size=119759&status=done&style=none&taskId=u8eab5793-5acf-41ce-a45d-289bd3a79ee&title=&width=437.27276611328125#crop=0&crop=0&crop=1&crop=1&height=268&id=cZYF4&originHeight=621&originWidth=1022&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=441)

下面引入一些标记法来帮助描述模型：

- $a_{i}^{\left( j \right)}$ 代表第$j$ 层的第 $i$ 个激活单元。
- ${{\Theta }^{\left( j \right)}}$代表从第 $j$ 层映射到第$j+1$ 层时的权重的矩阵，例如${{\Theta }^{\left( 1 \right)}}$代表从第一层映射到第二层的权重的矩阵。如果神经网络第$j$层有$s_j$个单元，第$j+1$层有$s_{j+1}$个单元，那么矩阵$\Theta^{(j)}$的维度即为$s_j \times (s_{j+1}+1)$

对于上图所示的模型，激活单元和输出分别表达为：

$a_{1}^{(2)}=g(\Theta _{10}^{(1)}{{x}_{0}}+\Theta _{11}^{(1)}{{x}_{1}}+\Theta _{12}^{(1)}{{x}_{2}}+\Theta _{13}^{(1)}{{x}_{3}})$<br />$a_{2}^{(2)}=g(\Theta _{20}^{(1)}{{x}_{0}}+\Theta _{21}^{(1)}{{x}_{1}}+\Theta _{22}^{(1)}{{x}_{2}}+\Theta _{23}^{(1)}{{x}_{3}})$<br />$a_{3}^{(2)}=g(\Theta _{30}^{(1)}{{x}_{0}}+\Theta _{31}^{(1)}{{x}_{1}}+\Theta _{32}^{(1)}{{x}_{2}}+\Theta _{33}^{(1)}{{x}_{3}})$

即：$a^{(j+1)}=g(\Theta^{j} x)$

${{h}_{\Theta }}(x)=g(\Theta _{10}^{(2)}a_{0}^{(2)}+\Theta _{11}^{(2)}a_{1}^{(2)}+\Theta _{12}^{(2)}a_{2}^{(2)}+\Theta _{13}^{(2)}a_{3}^{(2)})$，

上面进行的讨论中只是将特征矩阵中的一行（一个训练实例）喂给了神经网络，我们需要将整个训练集都喂给我们的神经网络算法来学习模型。

我们可以知道：每一个$a$都是由上一层所有的$x$和每一个$x$所对应的决定的。（我们把这样从左到右的算法称为**前向传播算法**( **FORWARD PROPAGATION** )）

---

<a name="nYP06"></a>
## 前向传播工作过程

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654483441239-fb5eebe0-105b-4e90-b40e-6ee6c22bef08.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=300&id=u2ed29dea&name=image.png&originHeight=590&originWidth=1129&originalType=binary&ratio=1&rotation=0&showTitle=false&size=215204&status=done&style=none&taskId=u394f580c-7038-41a4-8eb8-e3b88a929e2&title=&width=574#crop=0&crop=0&crop=1&crop=1&height=316&id=dPnTT&originHeight=590&originWidth=1129&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=605)

相对于使用循环来编码，利用向量化的方法会使得计算更为简便。以上面的神经网络为例，计算第二层的值：

我们令 ${{z}^{\left( 2 \right)}}={{\Theta }^{\left( 1 \right)}}x$，则 ${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$ ，计算后添加 $a_{0}^{\left( 2 \right)}=1$。 计算输出的值为：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654484060366-ff859e57-c74d-46c6-84d2-0410e4d6c583.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=84&id=ua0d0a48a&name=image.png&originHeight=85&originWidth=573&originalType=binary&ratio=1&rotation=0&showTitle=false&size=11481&status=done&style=none&taskId=u0e351ef3-47f8-4b36-8f56-4347c8f5477&title=&width=563.727294921875#crop=0&crop=0&crop=1&crop=1&id=dAdTF&originHeight=85&originWidth=573&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

我们令 ${{z}^{\left( 3 \right)}}={{\Theta }^{\left( 2 \right)}}{{a}^{\left( 2 \right)}}$，则 $h_\theta(x)={{a}^{\left( 3 \right)}}=g({{z}^{\left( 3 \right)}})$。

这只是针对训练集中一个训练实例所进行的计算。如果我们要对整个训练集进行计算，我们需要将训练集特征矩阵进行转置，使得同一个实例的特征都在同一列里。即：<br />${z}^{(2)}={{\Theta }^{\left( 1 \right)}}\times {{X}^{T}}\\$<br />${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$

---

<a name="PDWc4"></a>
## 应用示例

从本质上讲，神经网络能够通过学习得出其自身的一系列特征。在普通的逻辑回归中，我们被限制为使用数据中的原始特征$x_1,x_2,...,{{x}_{n}}$，我们虽然可以使用一些二项式项来组合这些特征，但是我们仍然受到这些原始特征的限制。在神经网络中，原始特征只是输入层，在我们上面三层的神经网络例子中，第三层也就是输出层做出的预测利用的是第二层的特征，而非输入层中的原始特征，我们可以认为第二层中的特征是神经网络通过学习后自己得出的一系列用于预测输出变量的新特征。

神经网络中，感知机（两层神经元：一层输入层，一层输出层M-P神经元）可用来表示逻辑运算，比如**逻辑与(AND)、逻辑或(OR)、逻辑非(NOT)**。

:::info
**与、或、 非问题**都是线性可分(linearly separable) 的问题.可以证明 [Minsky and Papert, 1969]，若两类模式是线性可分的，即存在一个线性超平面能将它们分开，则感知机的的学习过程一定会收敛(converge) 而求得适当的权向 量$\omega=(\omega_1;\omega_2;...;\omega_{n+1})$；否则感知机学习过程将会发生振荡(fluctuation) $\omega$则难以稳定下来，不能求得合适解。逻辑异或，即为线性不可分问题。（西瓜书）<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654484529677-cb2b6882-bb95-41c5-8b1b-69678b1a9a29.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=342&id=ua8d2870e&name=image.png&originHeight=563&originWidth=775&originalType=binary&ratio=1&rotation=0&showTitle=false&size=165754&status=done&style=none&taskId=uad41d9de-cc02-482d-a3a9-bf08bac7e6c&title=&width=470.6363830566406#crop=0&crop=0&crop=1&crop=1&height=333&id=vBjG8&originHeight=563&originWidth=775&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=458)
:::

**逻辑与(AND)过程**：

我们可以用这样的一个神经网络表示**AND** 函数：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654484816182-c329f13d-57b1-46c6-a034-e4cb8dae1b96.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=177&id=u36194335&name=image.png&originHeight=244&originWidth=300&originalType=binary&ratio=1&rotation=0&showTitle=false&size=43654&status=done&style=none&taskId=u0e93dc66-4522-461b-896c-6de364890ed&title=&width=218.1818181818182#crop=0&crop=0&crop=1&crop=1&height=168&id=MbOwy&originHeight=244&originWidth=300&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=207)

其中$\theta_0 = -30, \theta_1 = 20, \theta_2 = 20$<br />我们的输出函数$h_\theta(x)$即为：$h_\Theta(x)=g\left( -30+20x_1+20x_2 \right)$

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654484874596-58f66df6-6b53-4302-9dfd-25c2028f9177.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=193&id=u298ba74d&name=image.png&originHeight=310&originWidth=597&originalType=binary&ratio=1&rotation=0&showTitle=false&size=34473&status=done&style=none&taskId=u96e5bab0-3729-4efe-b27d-87a0c1d85f4&title=&width=372.18182373046875#crop=0&crop=0&crop=1&crop=1&height=137&id=gG732&originHeight=310&originWidth=597&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=263)

所以我们有：$h_\Theta(x) \approx \text{x}_1 \text{AND} \, \text{x}_2$；$h_\Theta(x)$这就是**AND**函数。

**OR**与**AND**整体一样，区别只在于权重的取值不同。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654484949140-8e982544-1491-4547-bdd4-f1d71faa09b5.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=119&id=u4af281b2&name=image.png&originHeight=208&originWidth=783&originalType=binary&ratio=1&rotation=0&showTitle=false&size=59682&status=done&style=none&taskId=ua0d333f6-627c-42eb-88f1-bd5798a5878&title=&width=448.45458984375#crop=0&crop=0&crop=1&crop=1&height=108&id=Tmz2w&originHeight=208&originWidth=783&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=408)

**NOT函数如下**<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654485222739-d216b7c1-5875-4e26-a901-39b712b41b18.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=101&id=u23687708&name=image.png&originHeight=181&originWidth=772&originalType=binary&ratio=1&rotation=0&showTitle=false&size=66386&status=done&style=none&taskId=ued525c5b-d9c2-4c45-9d51-017e9d278ae&title=&width=430.45458984375#crop=0&crop=0&crop=1&crop=1&height=97&id=qJMem&originHeight=181&originWidth=772&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=413)

我们可以利用神经元来组合成更为复杂的神经网络以实现更复杂的运算。例如我们要实现**逻辑异或(XNOR)**，即：$\text{XNOR}=( \text{x}_1\, \text{AND}\, \text{x}_2 )\, \text{OR} \left( \left( \text{NOT}\, \text{x}_1 \right) \text{AND} \left( \text{NOT}\, \text{x}_2 \right) \right)$<br />只需将表示**AND**的神经元和表示$\left( \text{NOT}\, \text{x}_1 \right) \text{AND} \left( \text{NOT}\, \text{x}_2 \right)$的神经元以及表示**OR**的神经元进行组合：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654485136990-540e2ac9-c2b8-4bb4-9d77-14c3bd20cde1.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=166&id=ue98956aa&name=image.png&originHeight=284&originWidth=1008&originalType=binary&ratio=1&rotation=0&showTitle=false&size=150783&status=done&style=none&taskId=u7e3adf0f-1937-46b5-97e4-a10e6898721&title=&width=589.0909423828125#crop=0&crop=0&crop=1&crop=1&height=170&id=sCNyy&originHeight=284&originWidth=1008&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=605)<br />我们就得到了一个能实现 $\text{XNOR}$ 运算符功能的神经网络。

---

<a name="sXfPN"></a>
## 误差反向传播

多层网络的学习能力比单层感知机强得多。欲有效的训练多层网络（求梯度）需要其他学习算法——**误差逆传播(error BackPropagation ，简称 BP)，**是迄今最成功的神经网络学习算法

**BP算法网络及算法中的变量符号：**（假设隐藏层和输出层神经元都使用sigmoid函数）<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654489860116-b1a04478-a9d8-45b8-98c4-c34189c1410f.png#clientId=u03b20a1b-ca60-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=271&id=u12dbbd9e&name=image.png&originHeight=372&originWidth=646&originalType=binary&ratio=1&rotation=0&showTitle=false&size=156528&status=done&style=none&taskId=u29d90e36-5907-4d09-adcf-448eac6ad81&title=&width=469.8181818181818#crop=0&crop=0&crop=1&crop=1&height=244&id=SGZnp&originHeight=372&originWidth=646&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=423)<br />（ ps：输出层第$j$个神经元的阈值用$\theta_j$表示，隐藏层第$j$个神经元的阈值用$γ_h$表示 ）

:::info
**标准BP算法更新公式推导：**(详见西瓜书p102.)<br />对训练样例$(x_k,y_k)$, 假定神经网络的输出为$\hat{{y}}_{k}=\left(\hat{y}_{1}^{k}, \hat{y}_{2}^{k}, \ldots, \hat{y}_{l}^{k}\right)$, 即

$\hat{y}_{j}^{k}=f\left(\beta_{j}-\theta_{j}\right)$,

则网络在该训练样例上的均方误差为（其他损失函数同理）<br />$E_{k}=\frac{1}{2} \sum_{j=1}^{l}\left(\hat{y}_{j}^{k}-y_{j}^{k}\right)^{2}$

BP 是一个法代学习算法，在迭代的每一轮中采用广义的的感知机学规则对参数进行更新估计，即<br />$v \leftarrow v+\Delta v$<br />BP 算法基于**梯度下降 (gradient descent) 策略**， 以目标的负梯度方向对参数进行调整。即：

$\boldsymbol{\Delta w_{h j}=-\eta \frac{\partial E_{k}}{\partial w_{h j}}}$

学习率$\eta \in(0,1)$控制算法每一步的步长，步长过大容易振荡，步长过小则收敛过慢

**正式推导：**<br />由**链式法则**（理解“向后传播”的关键）和网络结构可知：

$\frac{\partial E_{k}}{\partial w_{h j}}=\frac{\partial E_{k}}{\partial \hat{y}_{j}^{k}} \cdot \frac{\partial \hat{y}_{j}^{k}}{\partial \beta_{j}} \cdot \frac{\partial \beta_{j}}{\partial w_{h j}}$，其中令$\frac{\partial \beta_{j}}{\partial w_{h j}}=b_h$

又因为对于$sigmoid$函数有$f'(x)=f(x)(1-f(x))$，故有<br />$\begin{aligned}
g_{j} &=-\frac{\partial E_{k}}{\partial \hat{y}_{j}^{k}} \cdot \frac{\partial \hat{y}_{j}^{k}}{\partial \beta_{j}} \\
&=-\left(\hat{y}_{j}^{k}-y_{j}^{k}\right) f^{\prime}\left(\beta_{j}-\theta_{j}\right) \\
&=\hat{y}_{j}^{k}\left(1-\hat{y}_{j}^{k}\right)\left(y_{j}^{k}-\hat{y}_{j}^{k}\right)
\end{aligned}$<br />故得到BP算法中关于$w_{hj}$的更新公式：$\bold{\Delta w_{h j}=\eta g_{j} b_{h}}$

同理可得：<br />$\begin{aligned}
\Delta \theta_{j} &=-\eta g_{j} \\
\Delta v_{i h} &=\eta e_{h} x_{i} \\
\Delta \gamma_{h} &=-\eta e_{h}
\end{aligned}$

其中：<br />$\begin{aligned}
e_{h} &=-\frac{\partial E_{k}}{\partial b_{h}} \cdot \frac{\partial b_{h}}{\partial \alpha_{h}} \\
&=-\sum_{j=1}^{l} \frac{\partial E_{k}}{\partial \beta_{j}} \cdot \frac{\partial \beta_{j}}{\partial b_{h}} f^{\prime}\left(\alpha_{h}-\gamma_{h}\right)\\
&=\sum_{j=1}^{l} w_{h j} g_{j} f^{\prime}\left(\alpha_{h}-\gamma_{h}\right) \\&=b_{h}\left(1-b_{h}\right) \sum_{j=1}^{l} w_{h j} g_{j}
\end{aligned}$
:::

:::info
**总结理解：**对每个训练样例：先将输入示例提供给输入层神经元，然后逐层将信号前传，直到产生输出层的结果；然后计算输出层的误差(第 4-5 行) ，再将误差逆向传播至隐层神经元（第6行），最后根据隐层神经元的误差来别连接权和|词值进行调整(第7行)。该法代过程循环进行，直到达到某些停止条件为止。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654489440084-47d4c4f6-00a4-42e9-9fca-f9fccc409148.png#clientId=u03b20a1b-ca60-4&crop=0&crop=0.0189&crop=1&crop=1&from=paste&height=265&id=UuXtd&name=image.png&originHeight=364&originWidth=690&originalType=binary&ratio=1&rotation=0&showTitle=false&size=132461&status=done&style=none&taskId=ue31a64e2-ec86-4509-b1ec-750963883d3&title=&width=502#crop=0&crop=0&crop=1&crop=1&height=215&id=dzCAi&originHeight=364&originWidth=690&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=407)
:::

通常我们使用的 BP 算法的目标是要最小化训练集上的累积误差，即：<br />$E=\frac{1}{m} \sum_{k=1}^{m} E_{k}$<br />标准 BP 算法往往需进行更多次数的法代。累积 BP 算法直接针对累积误差最小化，它在读取整个训练集$D$一遍后才对参数进行更新， 其参数更新的频率低得多

---

<a name="K0PoB"></a>
## BP神经网络的相关处理

**过拟合处理：**

BP 神经网络经常遭遇过拟合：其训练误差持续降低，但测试误差却可能上升。有两种策略常用来缓解过拟合。

- 第一种策略是**“早停”**(early stopping)：将数据分成训练集和验证集，训练集用来计算梯度、更新连接权和阈值；验证集用来估计误差，若训练集误差降低但验证集误差升高则停止训练，同时返回具有最小验证集误差的连接权和阈值。（见正则化章节——提前停止）
- 第二种策略是**“正则化” **(regularization)，其基本思想是在误差目标函数中增加一个用于描述网络复杂度的部分，则损失函数可变为：( $\lambda\in(0,1)$ 用于对经验误差与网络复杂度这两项进行折中，常通过交叉验证法来估计)

$E=\lambda \frac{1}{m} \sum_{k=1}^{m} E_{k}+(1-\lambda) \sum_{i} w_{i}^{2}$

**跳出局部最小处理：**

全局最小和局部最小：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654492530686-62541602-4b7e-4d0f-8434-b9ccf68b6a68.png#clientId=u03b20a1b-ca60-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=212&id=u8dd2f022&name=image.png&originHeight=291&originWidth=401&originalType=binary&ratio=1&rotation=0&showTitle=false&size=114838&status=done&style=none&taskId=ud2de4129-312c-472a-b41c-d56ded64746&title=&width=291.6363636363636#crop=0&crop=0&crop=1&crop=1&height=253&id=wmAMC&originHeight=291&originWidth=401&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=349)

常用三种方法跳出局部最小：
:::info

- **以不同参数值初始化多个神经网络，按标准方法训练后，取其误差小的解作为终参**。这相当于从多个不同的初始点开始搜索， 这样就可能陷入不同的局部极小从中进行选择有可能获得更接近全局最小的结果。我们通常初始参数为$±ε$之间的随机值。
- **“模拟退火”(simulated annealing) 技术 **。模拟退火在每一步都 以二定的概率接受比当前解更差的结果，从而有助“跳出”局部极小。在每步迭代过程中，接受“次优解”的概率要随着时间的推移而逐渐降从而保证算法稳定。
- **随机梯度下降法。**与标准梯度下降法精确计算梯度不同， 随机梯度下降法在计算梯度时加入了随机因素。即便陷入局部极小点，它计算出仍可能不为零，这样就有机会跳出局部极小继续搜索。
:::

（也可也采用后文中的梯度检验）

---

<a name="aoxAT"></a>
## 多类分类

当我们有不止两种分类时（也就是$y=1,2,3….$），比如以下这种情况，该怎么办？如果我们要训练一个神经网络算法来识别路人、汽车、摩托车和卡车，在输出层我们应该有4个值。例如，第一个值为1或0用于预测是否是行人，第二个值用于判断是否为汽车。

输入向量$x$有三个维度，若干中间层，输出层4个神经元分别用来表示4类，也就是每一个数据在输出层都会出现${{\left[ a\text{ }b\text{ }c\text{ }d \right]}^{T}}$，且$a,b,c,d$中仅有一个为1，表示当前类。下面是该神经网络的可能输出结果示例：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654488297282-13dfc4c1-cf55-4734-8646-d159c817ef99.png#clientId=u03b20a1b-ca60-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=92&id=u9bd06f86&name=image.png&originHeight=127&originWidth=715&originalType=binary&ratio=1&rotation=0&showTitle=false&size=66406&status=done&style=none&taskId=u88515d7e-cdab-4410-a447-b5ffff5d89b&title=&width=520#crop=0&crop=0&crop=1&crop=1&height=92&id=RX90h&originHeight=127&originWidth=715&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=518)

---

<a name="oz1sD"></a>
## 代价函数

首先引入一些新标记方法：

假设神经网络的训练样本有$m$个，每个包含一组输入$x$和一组输出信号$y$，$L$表示神经网络层数，$S_I$表示每层的**neuron**个数($S_l$表示输出层神经个数)，$S_L$代表最后一层中处理单元的个数。

将神经网络的分类定义为两种情况：

- 二类分类：$S_L=0, y=0\, or\, 1$表示哪一类；
- $K$类分类：$S_L=k, y_i = 1$表示分到第$i$类；$(k>2)$

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655517211533-84a5cc1b-a1bd-4b57-bea9-0d97caf254d2.png#clientId=u2748dfdf-134f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=213&id=dsCtR&name=image.png&originHeight=293&originWidth=728&originalType=binary&ratio=1&rotation=0&showTitle=false&size=206741&status=done&style=none&taskId=ubd0b1e63-7b2a-4226-8048-b72432c1b4b&title=&width=529.4545454545455#crop=0&crop=0&crop=1&crop=1&height=219&id=LBEYY&originHeight=293&originWidth=728&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=543)

我们回顾逻辑回归问题中我们的代价函数为：

$J\left(\theta \right)=-\frac{1}{m}\left[\sum\limits_{i=1}^{m}{y}^{(i)}\log{h_\theta({x}^{(i)})}+\left(1-{y}^{(i)}\right)log\left(1-h_\theta\left({x}^{(i)}\right)\right)\right]+\frac{\lambda}{2m}\sum_\limits{j=1}^{n}{\theta_j}^{2}$

在逻辑回归中，我们只有一个输出变量，又称标量（**scalar**），也只有一个因变量$y$，但是在神经网络中，我们可以有很多输出变量，**我们的**$h_\theta(x)$**是一个维度为**$K$**的向量**，并且我们训练集中的因变量也是同样维度的一个向量，因此我们的代价函数会比逻辑回归更加复杂一些，为：$\newcommand{\subk}[1]{ #1_k }$

$h_\Theta\left(x\right)\in \mathbb{R}^{K}\quad，其中{\left({h_\Theta}\left(x\right)\right)}_{i}={i}^{th} \text{output}$

$\begin{aligned}
J(\Theta)=-\frac{1}{m}\left[\sum_{i=1}^{m} \sum_{k=1}^{K} y_{k}^{(i)} \log \left(h_{\Theta}\left(x^{(i)}\right)\right)_{k}+\left(1-y_{k}^{(i)}\right) \log \left(1-\left(h_{\Theta}\left(x^{(i)}\right)\right)_{k}\right)\right]
&+\frac{\lambda}{2 m} \sum_{l=1}^{L-1} \sum_{i=1}^{s_{l}} \sum_{j=1}^{s_{l+1}}\left(\Theta_{j i}^{(l)}\right)^{2}
\end{aligned}$<br />	这个看起来复杂很多的代价函数背后的思想还是一样的，我们希望通过代价函数来观察算法预测的结果与真实情况的误差有多大，唯一不同的是，对于每一行特征，我们都会给出$K$个预测，基本上我们可以利用循环，对每一行特征都预测$K$个不同结果，然后在利用循环在$K$个预测中选择可能性最高的一个，将其与$y$中的实际数据进行比较。

正则化的那一项只是排除了每一层$\Theta_0$后每一层的$\Theta$ 矩阵的和。最里层的循环$j$循环所有的行（由$s_{l+1}$  层的激活单元数决定），循环$i$则循环所有的列（由该层（$s_l$层）的激活单元数所决定）。

---

<a name="x8MLD"></a>
## 梯度检验

当我们对一个较为复杂的模型（例如神经网络）使用梯度下降算法时，可能会存在一些不容易察觉的错误，意味着，虽然代价看上去在不断减小，但最终的结果可能并不是最优解。

为了避免这样的问题，我们采取一种叫做梯度的数值检验（**Numerical Gradient Checking**）方法。这种方法的思想是通过估计梯度值来检验我们计算的导数值是否真的是我们要求的。

对梯度的估计采用的方法是在代价函数上沿着切线的方向选择离两个非常近的点然后计算两个点的平均值用以估计梯度。即对于某个特定的 $\theta$，我们计算出在 $\theta-\varepsilon$ 和 $\theta+\varepsilon$ 的代价值（$\varepsilon$是一个非常小的值，通常选取 0.001），然后求两个代价的平均，用以估计在 $\theta$ 处的代价值。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655523652812-ca056ef0-2f05-4ad1-a18e-48896ffd8f5a.png#clientId=u4922338d-8960-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=206&id=ud2f34090&name=image.png&originHeight=310&originWidth=1046&originalType=binary&ratio=1&rotation=0&showTitle=false&size=42708&status=done&style=none&taskId=u68244946-37c1-4ffc-907d-64eb012b437&title=&width=694#crop=0&crop=0&crop=1&crop=1&height=156&id=KWKMp&originHeight=310&originWidth=1046&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=525)

当$\theta$是一个向量时，我们则需要对偏导数进行检验。因为代价函数的偏导数检验只针对一个参数的改变进行检验，下面是一个只针对$\theta_1$进行检验的示例：

$\frac{\partial}{\partial\theta_1}=\frac{J\left(\theta_1+\varepsilon_1,\theta_2,\theta_3...\theta_n \right)-J \left( \theta_1-\varepsilon_1,\theta_2,\theta_3...\theta_n \right)}{2\varepsilon}$

最后我们还需要对通过反向传播方法计算出的偏导数进行检验。

根据上面的算法，计算出的偏导数存储在矩阵 $D_{ij}^{(l)}$ 中。检验时，我们要将该矩阵展开成为向量，同时我们也将 $\theta$ 矩阵展开为向量，我们针对每一个 $\theta$ 都计算一个近似的梯度值，将这些值存储于一个近似梯度矩阵中，最终将得出的这个矩阵同 $D_{ij}^{(l)}$ 进行比较。

---

<a name="CYPq2"></a>
## 总结

小结一下使用神经网络时的步骤：

第一件要做的事是选择网络结构，即决定选择多少层以及决定每层分别有多少个单元。

第一层的单元数即我们训练集的特征数量。最后一层的单元数是我们训练集的结果的类的数量。

如果隐藏层数大于1，确保每个隐藏层的单元个数相同，通常情况下隐藏层单元的个数越多越好。

我们真正要决定的是隐藏层的层数和每个中间层的单元数。

:::info
训练神经网络：

1. 参数的随机初始化
2. 利用正向传播方法计算所有的$h_{\theta}(x)$
3. 编写计算代价函数 $J$ 的代码
4. 利用反向传播方法计算所有偏导数
5. 利用数值检验方法检验这些偏导数
6. 使用优化算法来最小化代价函数
:::

---

<a name="SEhkC"></a>
# 模型评估与选择

<a name="i9cxv"></a>
## 引言

当我们运用训练好了的模型来预测未知数据的时候发现有较大的误差，我们下一步可以做什么？

:::info

1.  获得更多的训练样本——解决高方差 
1.  尝试减少特征的数量——解决高方差 、
1.  尝试获得更多的特征——解决高偏差 
1.  尝试增加多项式特征——解决高偏差 
1.  尝试减少正则化程度λ——解决高偏差 
1.  尝试增加正则化程度λ——解决高方差 
:::

我们不应该随机选择上面的某种方法来改进我们的算法，而是运用一些机器学习诊断法来帮助我们知道上面哪些方法对我们的算法是有效的。

---

<a name="S7yOV"></a>
## 模型评估方法

当我们确定学习算法的参数的时候，我们考虑的是选择参量来使训练误差最小化。

:::info
**“留出法”(hold out)：**直接将数据集划分为两个互斥的集合，其中一个集合作为训练集$S$ ，另一个作为测试集$T$，.在$S$上训练出模型后，用$T$来评估其测试误差，作为对泛化误差的估计。
:::

训练/测试集的划分要尽可能保持数据分布一致性，避免因数据划分过程引入额外的偏差而对最终结果产生影响。

使用留出法时，一般要采用若干次随机划分、重复进行实验评估后取平均值作为留出法的评估结果。通常将大约2/3~4/5的样本用于训练，剩余样本用于测试。

:::info
**“简单交叉验证”。**即吴课程中所讲的数据七三分。
:::

:::info
**“k折交叉验证法”(cross validation)：**将数据集$D$分为$k$个大小相似的互斥子集， 每个子集 $D_i$尽可能保持数据分布的一致性。然后，每次用$k-1$个子集的并集作为训练集，其余的那个子集作测试集，这样就可获得$k$组训练测试集，从而可进$k$次训练和测试，最终返回的是$k$个测试结果的均值。
:::

交叉验证示意图：<br />![](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656560455339-356404bd-10e9-46b7-a344-c1be3af39263.png#clientId=u22cbec31-526a-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=310&id=u2a937b36&originHeight=613&originWidth=885&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=ub61f6b67-48d2-4e41-ae9d-bddcd1fb399&title=&width=447)

假定数据集$D$中包含$m$个样本，若令$k=m$ 则得到了交叉验证法的一个特例：**“留一法”( Leave-One-Out)**，简称 **LOO** 。

留一法不受随机样本划分方式的影响，因为$m$个样本只有唯一的方式划分为$m$个子集——每个子集包含一个样本。留一法的评估结果往往被认为是比较准确，但不适用于数据集比较大时。

:::info
**“自助法”(bootstrapping)：**给定包含$m$个样本的数据集$D$，我们对它进行采样产生数据集$D'$， 每次随机从$D$中挑选一个样本 将其拷贝放入$D'$，然后再将该样本放回初始数据集$D$中，使得该样本在下次采样时仍有可能被采到，这个过程重复执行$m$次后我们就得到了包含个样本的数据集$D'$。（简单来说就是有放放回采样）我们可将$D'$作为训练集，$D/D'$作为测试集。
:::

自助法可以减少训练样本规模不同造成的影响，同时还可以比较高效的进行实验估计，在数据集较小、难以有效划分训练/测试集时很有用。


**模型选择的方法为：**

1.  使用训练集训练出若干个模型 
1.  根据任务不同选择不同的性能度量指标
1.  根据测试集测试结果选取性能最好的模型 

---

<a name="v0Irc"></a>
## 偏差和方差

当你运行一个学习算法时，如果这个算法的表现不理想，那么多半是出现两种情况：要么是偏差比较大，要么是方差比较大。高偏差和高方差的问题基本上来说是欠拟合和过拟合的问题。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655645838901-dc29a49b-79b1-43c7-b79a-189d74eefa3f.png#clientId=uc963bbdc-1d80-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=245&id=u4d31f1bd&name=image.png&originHeight=428&originWidth=1121&originalType=binary&ratio=1&rotation=0&showTitle=false&size=58705&status=done&style=none&taskId=u8420838a-1a0e-4f4d-a5c0-f909804d455&title=&width=641)


“偏差”度量了学习算法的期望预测与真是结果的偏离程度，即刻画了学习算法本身的拟合能力；<br />“方差”度量了同大小训练集变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响<br />“噪声”表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界，即刻画了学习任务本身的难度<br />“泛化误差”反映了学习方法的泛化能力，即对新数据的拟合能力


**Bias/variance**<br />**Training error:**				               $J_{train}(\theta) = \frac{1}{2m}\sum_\limits{i=1}^{m}(h_{\theta}(x^{(i)})-y^{(i)})^2$<br />**Cross Validation error:**				$J_{cv}(\theta) = \frac{1}{2m_{cv}}\sum_\limits{i=1}^{m}(h_{\theta}(x^{(i)}_{cv})-y^{(i)}_{cv})^2$

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655645985003-5f6ab542-4890-423d-aba6-2d4bdca6cf57.png#clientId=uc963bbdc-1d80-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=273&id=u34741bd1&name=image.png&originHeight=589&originWidth=1118&originalType=binary&ratio=1&rotation=0&showTitle=false&size=131063&status=done&style=none&taskId=ucef8c4ae-ff8d-4433-b088-22f3fb9a678&title=&width=519)

- 对于训练集，当 $d$ 较小时，模型拟合程度更低，误差较大；随着 $d$ 的增长，拟合程度提高，误差减小
- 对于交叉验证集，当 $d$ 较小时，模型拟合程度低，误差较大；但是随着 $d$ 的增长，误差呈现先减小后增大的趋势，转折点是我们的模型开始过拟合训练数据集的时候。

如果我们的交叉验证集误差较大，我们如何判断是方差问题还是偏差问题呢？根据上面的图表，我们知道:

- 训练集误差和交叉验证集误差近似时：偏差问题/欠拟合
- 交叉验证集误差远大于训练集误差时：方差问题/过拟合


以回归问题为例：
:::info
学习算法的期望预测为：$\bar{f}(\boldsymbol{x})=\mathbb{E}_{D}[f(\boldsymbol{x} ; D)]$<br />使用样本数不同训练集产生的方差为：$\operatorname{var}(\boldsymbol{x})=\mathbb{E}_{D}\left[(f(\boldsymbol{x} ; D)-\bar{f}(\boldsymbol{x}))^{2}\right]$<br />噪声为：$\varepsilon^{2}=\mathbb{E}_{D}\left[\left(y_{D}-y\right)^{2}\right]$<br />期望输出与真实标记的差别称为“偏差”为：$\text { bias }^{2}(\boldsymbol{x})=(\bar{f}(\boldsymbol{x})-y)^{2}$<br />泛化误差：$E(f ; D)=\mathbb{E}_{D}\left[\left(f(\boldsymbol{x} ; D)-y_{D}\right)^{2}\right]$
:::

学习方法的泛化能力 Cgeneralization ability) 是指由该方法学习到的模型对未知数据的预测能力，是学习方法本质上重要的性质。

泛化误差可分解为偏差、方差与噪声之和：$E(f ; D)=\text { bias }^{2}(\boldsymbol{x})+\operatorname{var}(\boldsymbol{x})+\varepsilon^{2}$

**偏差一方差分解说明：**泛化性能是由学习算法的能力、数据的充分性以及学习任务本身的难度所共同决定的。给定学习任务，为了取得好的泛化性能，则需使偏差较小，即能够充分拟合数据，并且使方差较小，即使得数据扰动产生的影响小。

泛化误差、偏差、方差的关系示意图<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655645476927-c4271970-9cc5-4b5d-a4b0-ca6d3900db1b.png#clientId=uc963bbdc-1d80-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=294&id=uf4901229&name=image.png&originHeight=517&originWidth=680&originalType=binary&ratio=1&rotation=0&showTitle=false&size=63707&status=done&style=none&taskId=u52e03071-d185-406a-8f3d-8330c08ff92&title=&width=386.54547119140625)

---


<a name="alCPN"></a>
## 正则化和偏差/方差

在我们在训练模型的过程中，一般会使用一些正则化方法来防止过拟合。但是我们可能会正则化的程度太高或太低，即我们在选择$λ$的值时也需要思考偏差和方差的问题。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655646300800-77e1edb6-0856-44df-b275-e3c934f1c47c.png#clientId=uc963bbdc-1d80-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=307&id=WFPjg&name=image.png&originHeight=624&originWidth=1128&originalType=binary&ratio=1&rotation=0&showTitle=false&size=113163&status=done&style=none&taskId=u255f1d36-2f5a-447d-810a-d7128a17b67&title=&width=555)

我们选择一系列的想要测试的 $\lambda$ 值，通常是 0-10之间的呈现2倍递增关系的值<br />如：$0,0.01,0.02,0.04,0.08,0.15,0.32,0.64,1.28,2.56,5.12,10$

选择$\lambda$的方法为：

1. 使用训练集训练出$k$个不同程度正则化的模型
2. 用这$k$个模型分别对测试集计算的出相应性能指标
3. 选择得出性能最好的模型

选择交叉验证法时，训练误差和交叉验证误差与$\lambda$ 的关系如下图所示<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655646790278-674d9737-3363-4bd2-8460-08a54afe2652.png#clientId=uc963bbdc-1d80-4&crop=0&crop=0&crop=0.9943&crop=1&from=paste&height=374&id=u33dae818&name=image.png&originHeight=564&originWidth=529&originalType=binary&ratio=1&rotation=0&showTitle=false&size=36647&status=done&style=none&taskId=ucb639ce0-b011-462d-9090-6a1b3895f17&title=&width=351)<br />• 当 $\lambda$ 较小时，训练集误差较小（过拟合）而交叉验证集误差较大<br />• 随着  的增加，训练集误差不断增加（欠拟合），而交叉验证集误差则是先减小后增加

---

<a name="zfQdm"></a>
## 学习曲线

学习曲线是一种很好来判断某一个学习算法是否处于偏差、方差问题的工具。学习曲线是将训练集误差和交叉验证集误差作为训练集样本数量（$m$）的函数所绘制的图表。

即，如果我们有100行数据，我们从1行数据开始，逐渐学习更多行的数据。思想是：当训练较少行数据的时候，训练的模型将能够非常完美地适应较少的训练数据，但是训练出来的模型却不能很好地适应交叉验证集数据或测试集数据。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655648477880-4d102b0e-04ec-44c6-bd10-d300931c56be.png#clientId=uc963bbdc-1d80-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=327&id=u4c940434&name=image.png&originHeight=624&originWidth=1097&originalType=binary&ratio=1&rotation=0&showTitle=false&size=98048&status=done&style=none&taskId=u602c8f45-45e9-4c32-aa95-f7f7042540e&title=&width=575)


**如何利用学习曲线识别高偏差/欠拟合：**作为例子，我们尝试用一条直线来适应下面的数据，可以看出，无论训练集有多么大误差都不会有太大改观：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655648830435-6218ff48-78a8-4f7d-a667-24bfab0fc3af.png#clientId=uc963bbdc-1d80-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=233&id=u96876195&name=image.png&originHeight=630&originWidth=1070&originalType=binary&ratio=1&rotation=0&showTitle=false&size=91780&status=done&style=none&taskId=u21479782-e5d1-4f99-a58d-06e2cc8f893&title=&width=396)![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656642292901-ba28674c-1e1b-4e69-ad45-1de78f26fcec.png#clientId=u44911542-ece2-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=219&id=u427217fb&name=image.png&originHeight=406&originWidth=609&originalType=binary&ratio=1&rotation=0&showTitle=false&size=74664&status=done&style=none&taskId=uec44369d-616d-4ef1-a924-e96a4b8191c&title=&width=328.7599792480469)

也就是说在高偏差/欠拟合的情况下，增加数据到训练集不一定能有帮助。



**如何利用学习曲线识别高方差/过拟合：**假设我们使用一个非常高次的多项式模型，并且正则化非常小，可以看出，当交叉验证集误差远大于训练集误差时，往训练集增加更多数据可以提高模型的效果。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655648776058-faf856bb-9c3f-43dc-b359-26f6a99edb60.png#clientId=uc963bbdc-1d80-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=213&id=uc7e11077&name=image.png&originHeight=641&originWidth=1120&originalType=binary&ratio=1&rotation=0&showTitle=false&size=96668&status=done&style=none&taskId=ud2a623c0-532c-4c16-8e8a-537a78591c8&title=&width=373)![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656642309225-928b0c26-2c77-458a-bec3-66b72c7113ec.png#clientId=u44911542-ece2-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=240&id=u58bd3172&name=image.png&originHeight=401&originWidth=613&originalType=binary&ratio=1&rotation=0&showTitle=false&size=73224&status=done&style=none&taskId=u81231945-9263-4991-a4e1-07ba9540d58&title=&width=367.3199768066406)

也就是说在高方差/过拟合的情况下，增加更多数据到训练集可能可以提高算法效果。

---


<a name="Uk9gQ"></a>
## 神经网络的方差和偏差

- 使用较小的神经网络，类似于参数较少的情况，容易导致高偏差和欠拟合，但计算代价较小
- 使用较大的神经网络，类似于参数较多的情况，容易导致高方差和过拟合，虽然计算代价比较大，但是可以通过正则化手段来调整而更加适应数据。

通常选择较大的神经网络并采用正则化处理会比采用较小的神经网络效果要好。

对于神经网络中的隐藏层的层数的选择，通常从一层开始逐渐增加层数，为了更好地作选择，可以把数据分为训练集、测试集，针对不同隐藏层层数的神经网络训练神经网络，然后选择性能最好的神经网络模型

---

<a name="pp1l7"></a>
## 查准率、查全率和F1度量

**类偏斜（skewed classes）**问题。类偏斜情况表现为我们的训练集中有非常多的同一种类的样本，只有很少或没有其他类的样本。

例如：我们希望用算法来预测癌症是否是恶性的，在我们的训练集中，只有0.5%的实例是恶性肿瘤。假设我们编写一个非学习而来的算法，在所有情况下都预测肿瘤是良性的，那么误差只有0.5%。然而我们通过训练而得到的神经网络算法却有1%的误差。这时，误差的大小是不能视为评判算法效果的依据的。

二分类问题常用的评价指标是**“查准率”( 或“准确率”，precesion )** 和**“查全率”( 或“召回率”，recall )**

可将样例根据其真实类别与学习器预测类别的组合划分为真正例 TP (true positive) 、假正例 FP (false positive) 、真反例 TN (true negative)、假反例 FN (false negative) 四种情形，并得到混淆矩阵

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655720140480-2a5721c1-b465-41c9-91d9-a986e2bf3081.png#clientId=ueb3cc4c8-424e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=129&id=u10be2d0e&name=image.png&originHeight=194&originWidth=516&originalType=binary&ratio=1&rotation=0&showTitle=false&size=45252&status=done&style=none&taskId=u6ac87da7-d1e0-4b83-9cae-a3aabab49f1&title=&width=343.2727355957031)

相关指标计算公式：
:::info
**查准率：**$P=\frac{TP}{TP+FP}$，即预测正例中真实正例的比例<br />**查全率：**$R=\frac{TP}{TP+FN}$，即预测正例中真正例占真实样例全部正例的比率<br />**真正例率：**$TPR=\frac{TP}{TP+FN}$，即查全率<br />**假正例率：**$FPR=\frac{FP}{TN+FP}$，即假正例占真实样例中全部反例的比率<br />$F1$**度量：**$F1=\frac{2\times P\times R}{P+R}=\frac{2\times TP}{样例总数+TP-TN}=\frac{2TP}{2TP+FP+FN}$**，**查准率和查全率的调和平均，是分类器性能评估的常用度量
:::

**我们可以根据问题需要通过修改阈值来提高算法的查准率或查全率，通常我们选择使**$F1$**最大的阈值**

不同阈值下查准率和查全率的关系图如下：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655721259071-a2ff9f66-ad73-4905-871f-eb0bdea3fe94.png#clientId=ueb3cc4c8-424e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=324&id=ue3b4c9b8&name=image.png&originHeight=479&originWidth=546&originalType=binary&ratio=1&rotation=0&showTitle=false&size=125196&status=done&style=none&taskId=u8dc11ebf-5d00-4063-8586-c8dc4a15f39&title=&width=369.0909118652344)



如果你准备研究机器学习的东西，或者构造机器学习应用程序，最好的实践方法不是建立一个非常复杂的系统，拥有多么复杂的变量；而是构建一个简单的算法，并快速地实现它。然后再根据检验数据找出问题再不断改进。

---

<a name="gaBdN"></a>
## ROC和AUC

**ROC曲线 **综合考虑了学习器在不同任务下的“期望泛化性能”的好坏，即“一般情况下”泛化性能的好坏.。

根据学习器的预测结果对样例进行排序，按此顺序逐个把样本作为正例进行预测，每次计算出两个重要量的值，分别以它们为横、纵坐标作图'就得到了“ROC 曲线”。ROC 曲线的纵轴是“真正例率” (True Positive Rate ，简称 TPR) ，横轴是“假正例率“(False Positive Rate ，简称 FPR) ，两者分别定义为：<br />$TPR=\frac{TP}{TP+FN}$<br />$FPR=\frac{FP}{TN+FP}$


![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656575467029-62fe2fa9-7051-4a27-9ff1-5d517bbecede.png#clientId=u22cbec31-526a-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=300&id=ud3289b05&name=image.png&originHeight=656&originWidth=879&originalType=binary&ratio=1&rotation=0&showTitle=false&size=95569&status=done&style=none&taskId=u044f3005-346d-44a3-bf3e-5e37b78374e&title=&width=402.55999755859375)<br />虚线对角线对应于“随机猜测”模型，而点（0，1）对应于将所有正例排在所有反例之前的“理想模型”

学习器的比较时， P-R 图相似， 一个学习器的 ROC 曲线被另一个学习器的曲线完全“包住”， 则可断言后者的性能优于前者；若两个学习ROC 曲线发生交叉，则难以一般性地断言两者孰优孰。此时如果一定要进行比较 则较为合理的判据是比较 ROC 线下 的面积，即 **AUC (Area Under ROC Curve)**

---

<a name="A8FwZ"></a>
## 机器学习的数据

不要盲目地开始，而是花大量的时间来收集大量的数据，因为数据有时是唯一能实际起到作用的。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655721405364-2f17d26c-5c52-446e-b45e-475add5be1ae.png#clientId=ueb3cc4c8-424e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=260&id=joGnx&name=image.png&originHeight=386&originWidth=534&originalType=binary&ratio=1&rotation=0&showTitle=false&size=35343&status=done&style=none&taskId=ucdd94672-d21c-4333-84f1-afb029f8bce&title=&width=360.3636474609375)

趋势非常明显，大部分算法都具有相似的性能，其次，随着训练数据集的增大，在横轴上代表以百万为单位的训练集大小，从0.1个百万到1000百万，也就是到了10亿规模的训练集的样本，这些算法的性能也都对应地增强了。

事实上，如果你选择任意一个算法，可能是选择了一个“劣等的”算法，如果你给这个劣等算法更多的数据，那么从这些例子中看起来的话，它看上去很有可能会其他算法更好，甚至会比"优等算法"更好。在机器学习中普遍共识：“取得成功的人不是拥有最好算法的人，而是拥有最多数据的人”。

那么这种说法在什么时候是真，什么时候是假呢？

偏差问题，我么可以通过确保有一个具有很多参数的学习算法来解决，以便我们能够得到一个较低偏差的算法，并且通过用非常大的训练集来保证低方差。即此时提高数据规模将会是一个很好的得到高性能的学习算法方式。

---

<a name="FEoIX"></a>
## 本章相关代码
[sklearn.linear_model.LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linearregression#sklearn.linear_model.LinearRegression)<br />[sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train_test_split#sklearn.model_selection.train_test_split)<br />[sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html?highlight=standardscaler#sklearn.preprocessing.StandardScaler)<br />[sklearn.model_selection.cross_val_score](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html?highlight=cross_val_score#sklearn.model_selection.cross_val_score)<br />[sklearn.model_selection.KFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html?highlight=kfold#sklearn.model_selection.KFold)<br />[sklearn.model_selection.LeaveOneOut](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneOut.html?highlight=oneout#sklearn.model_selection.LeaveOneOut)

x_train为已知训练集，y_train为已知标签集
```python
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression() # 以训练线性回归模型为例
lin_reg.fit(X_train, y_train)
y_predictions = lin_reg.predict(X_test)
mse = mean_squared_error(y_train, y_predictions) # 计算均方误差
```
```python
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 学习曲线函数
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train) + 1):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m]) # 预测值
        y_val_predict = model.predict(X_val) #
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
```
```python
import pandas as pd

# 对一个DataFrame类型数据集合使用describe可以得到该数据集的均值、方差等
train_data.describe() 


# 数据预处理中一系列转换器的fit方法
from sklearn.preprocessing import StandardScaler
# 标准化
s = StandardScaler()
# fit拟合数据，并计算数据集的参数（均值、方差等），为transform()做准备 
s.fit_transform(X_train)
```
```python
#使用Scikit-Learn划分训练集和数据集
train_set, test_set = train_test_split(X_data, test_size=0.2, random_state=42)
```
```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42) # 使用梯度下降分类器模型
sgd_clf.fit(X_train, y_train)

# 使用k-折交叉验证评估性能


# 方式一, SGD模型，3个折叠，以准确率位置为指标
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# 方式二，返回划分后的数据集
# 后续验证还需要for循环，更麻烦
from model_selection import KFold

kf = Kfold(n_splits=3)
kf.split(X_train)
```
```python
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.split(X_train)
```
```python
from sklearn.metrics import precision_score, recall_score

#查准率
precision_score(y_train_5, y_train_pred)
# 召回率
recall_score(y_train_5, y_train_pred) 
# f1 度量
f1_score(y_train_5, y_train_pred)
```
```python
from sklearn.metrics import precision_recall_curve

# 使用precision_recall_curve()计算所有可能的阈值的查准度和召回率
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# # 使用matplotlib绘制绘制P-R图
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plot_precision_vs_recall(precisions, recalls)

plt.show()
```
```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores) #计算多种阈值的TPR和FPR

# 使用matplotlib绘制曲线
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    [...] # Add axis label and grid

plot_roc_curve(fpr, tpr)
plt.show()
```
| **指标** | **描述** | **衡量方法** |
| --- | --- | --- |
| Mean Absolute Error(MAE) | 平均绝对值误差 | from sklearn.metrics import mean absolute error |
| Mean Square Error(MSE) | 均方误差 | from sklearn.metrics import mean_squared crror |
| Root-Mcan-Square Error(RMSE) | 均方根误差 | from sklearn.metrics import mean_squared_error <br />from math import sqrt |
| R-Squared | R平方值 | from sklearn.metrics import r2_scorc |


---

<a name="zW2Qb"></a>
# 支持向量机(Support Vector Machines)

<a name="cdip5"></a>
## SVM概述

:::info
**支持向量机 ( Csupport vector machines. SVM) **是一种二分类分类模型。它的基本模型是定义在特征空间上的间隔最大的线性分类器，间隔最大使它有别于感知机。

**“支持向量”**即距离超平面最近的几个训练样本点。支持向量可以使约束条件等式成立。

**支持向量机学习的基本思想**：求解能够正确划分训练数据集井且几何间隔最大的分离超平面。
:::


样本空间中划分超平面可以用线性方程描述：${w}^{\mathrm{T}} {x}+b=0$，其中${w}$为超平面法向量

分类决策函数为：$f(x)=\operatorname{sign}\left(w^{*} \cdot x+b^{*}\right)$

样本空间中任意点到$x$到超平面$(w,b)$的距离为：$\gamma=\frac{\left|\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}+b\right|}{\|\boldsymbol{w}\|}$两个异类支持向量到超平面的距离，即“间隔”为$\gamma=\frac{2}{\|{w}\|}$

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655792725677-5f4623e0-7a20-48b9-b62d-58928c5e0306.png#clientId=u341dc832-4d0f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=260&id=u68433998&name=image.png&originHeight=431&originWidth=598&originalType=binary&ratio=1&rotation=0&showTitle=false&size=51209&status=done&style=none&taskId=u50d90585-5209-47f2-b071-3ec2ae9d3ce&title=&width=361.3999938964844)


:::info
支持向量机目的找到最大间隔的划分超平面，所以**支持向量机的基本问题**即为：<br />$\begin{aligned}
&\min_{{w}, b}\frac{1}{2}\|\boldsymbol{w}\|^{2}\\
& s.t. \quad y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right) \geqslant 1, \quad i=1,2, \ldots, m
\end{aligned}$

通过使用拉格朗日乘子法可以得到该问题的**对偶问题**：<br />$\begin{align}
&\min _{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j}x_{i}^\mathrm{T} x_{j}-\sum_{i=1}^{m} \alpha_{i}\\
&s.t. \quad \sum_{i =  1}^{m} \alpha_{i} y_{i} = 0\quad， \alpha_{i} \geqslant 0, \quad i = 1,2, \cdots, m

\end{align}$
:::

求解出$\alpha$后再求出$w,b$即可得到最终模型：$f(\boldsymbol{x}) =\boldsymbol{w}^{\mathrm{T}}  \boldsymbol{x}+b=\sum_{i=1}^{m} \alpha_{i} y_{i}\boldsymbol{x}_{i}^{\mathrm{T}} \boldsymbol{x}+b$<br />通常建立模型之前还要对特征值进行标准化处理。

---

<a name="xosxI"></a>
## 线性支持向量机与软间隔

对于线性不可分的数据集，通常除去数据中的一些特异点，剩下的大部分样本点组成的集合是线性可分的。<br />**“软间隔”**即允许某些样本不满足约束，示意图如下：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655796096678-b4f2e7a8-babd-4bde-9670-cad76eacc29a.png#clientId=u341dc832-4d0f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=273&id=uf6b775f3&name=image.png&originHeight=421&originWidth=609&originalType=binary&ratio=1&rotation=0&showTitle=false&size=82860&status=done&style=none&taskId=u5afc64e0-0492-4944-a9f6-476eb2b5e5a&title=&width=395.20001220703125)

为了使特异点满足约束条件，我们可以对每个样本点引入一个松弛变量$\xi_i \ge0$，用来表征某样本不满足约束的程度。

**约束条件**变为:$y_i(w^{\mathrm{T}}x_i+b)\ge1-\xi_i$<br />**目标函数**变为：$\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i}$ ，其中 ($C>0$)

:::info
可以得到线性不可分问题的**线性支持向量机（软间隔支持向量机）模型**：<br />$\begin{align}
&\min _{w, b, \xi} \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i} \\
&\text { s.t. } \quad y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N \quad，\xi_{i} \geqslant 0, \quad i=1,2, \cdots, N
\end{align}$

$C$称为惩罚项，一般由实际问题确定，显然$C$很大时$\xi$必须小，则允许有较大错误；$C$很小时，则$\xi$可以比较大，即可容许较大的错误。

所用分类决策函数与解得超平面形式同线性可分支持向量机相同。
:::

**线性支持向量机的优化目标**：$\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \ell_{0 / 1}\left(y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)-1\right)$

$\ell_{0 / 1}$是“0/1损失函数”，可用一些“替代函数”代替

**三种常用的替代损失函数如下：**

- 合页 ( hinge ) 损失函数：$\ell_{\text {hinge }}(z)=\max (0,1-z)$
- 指数损失 ( exponential loss )函数：$\ell_{\exp }(z)=\exp (-z)$
- 对率损失 ( logistic loss )函数：$\ell_{\log }(z)=\log (1+\exp (-z))$

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655798548406-a9f19137-4cf8-477f-ac7f-1121021b98c2.png#clientId=u341dc832-4d0f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=323&id=ZPiYy&name=image.png&originHeight=554&originWidth=919&originalType=binary&ratio=1&rotation=0&showTitle=false&size=136996&status=done&style=none&taskId=u6bfb5431-a506-4597-b046-05f8293dc5f&title=&width=535.2000122070312)

若采用合页损失函数，则优化目标为：$\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \max \left(0,1-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)$

---

<a name="wp2lc"></a>
## 合页损失函数与正则化

合页损失函数性能最好，所用通常使用它作为损失函数

采用合页损失函数的优化目标可改写为如下形式：<br />$\min _{\boldsymbol{w}, b}  \sum_{i=1}^{m} \max \left(0,1-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)+\lambda\|\boldsymbol{w}\|^{2}$

也就是说，当样本点$(x_i,y_i)$被正确分类且函数间隔（确信度）$y_i(w^{\mathrm{T}}x_i+b) >1$时，损失为0，否则损失为$1-y_i(w^{\mathrm{T}} x_i)$

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655801903780-8e440c3c-fea9-46b5-b018-fca4e2b4a4f2.png#clientId=u341dc832-4d0f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=251&id=u3e48eb51&name=image.png&originHeight=404&originWidth=812&originalType=binary&ratio=1&rotation=0&showTitle=false&size=77325&status=done&style=none&taskId=u09315d31-326f-42ac-8318-6a2e1b77364&title=&width=504.60003662109375)<br />0-1损失函数为二分类问题真正的损失函数，由图中可以看出合页损失函数为0-1损失函数的上界。之所以使用合页函数是因为0-1损失函数不是连续可导的，优化比较困难。


改写过的目标函数第1项为经验风险，第2项是系数为$\lambda$的$w$的$L2$范数，是正则化项。

而对于改写前的目标函数：$\min _{\boldsymbol{w}, b} \frac{1}{2}\|\boldsymbol{w}\|^{2}+C \sum_{i=1}^{m} \max \left(0,1-y_{i}\left(\boldsymbol{w}^{\mathrm{T}} \boldsymbol{x}_{i}+b\right)\right)$<br />我们可以写成更一般的形式：$\min _{f} \Omega(f)+C \sum_{i=1}^{m} \ell\left(f\left(\boldsymbol{x}_{i}\right), y_{i}\right)$

其中$\Omega(f)$称为**“结构风险 ( structual risk )”**，$\sum_{i=1}^{m} \ell\left(f\left(\boldsymbol{x}_{i}\right), y_{i}\right)$称为**“经验风险 ( empirical risk )”**<br />此时$\Omega (f)$称为**正则化项**，$C$称为正则化常数。正则化项常用$L_p$范数，其中$L_2$范数$\| w\|_2$倾向于$w$的分量取值尽量均衡，即分零分量尽量稠密；而$L_0$和$L_1$范数则倾向于$w$的分量尽量稀疏，即分量分数尽量少

(更多有关正则化可见 7.2节)

---

<a name="OzZYQ"></a>
## 非线性支持向量机与核函数

非线性问题往往不好求解，所以希望将非线性问题转化为线性问题。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655804470617-4c5f83ec-b448-48be-ba7d-d1f66149adeb.png#clientId=u341dc832-4d0f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=240&id=uc82c86d3&name=image.png&originHeight=520&originWidth=977&originalType=binary&ratio=1&rotation=0&showTitle=false&size=117716&status=done&style=none&taskId=ud123b007-c3e4-4f4c-a6f3-b962d0975fe&title=&width=451)

:::info
**核技巧的基本思想：**将样本从原始空间映射到高维的特征空间，使得这个样本在这个特征空间内线性可分。

令$\mathrm{\phi}(x)$表示将$x$映射后的特征向量，则高维空间中对应超平面为：$f(x)=w^{\mathrm{T}}\phi(x)+b$<br />（目标函数和其对偶问题只需将$x$替换为$\phi(x)$即可）
:::

目标函数的对偶问题为：<br />$\begin{align}
&\min _{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j}\phi(x_{i})^\mathrm{T} \phi(x_{j})-\sum_{i=1}^{m} \alpha_{i}\\
&
 s.t.  \sum_{i =  1}^{m} \alpha_{i} y_{i} = 0 \quad \alpha_{i} \geqslant 0, \quad i = 1,2, \cdots, m

\end{align}$


由于求解涉及到计算$\phi(x_i)^{\mathrm{T}}\phi(x_j)$，即样本$x_i$与$x_j$映射到特征空间之后的内积，由于特征空间的维度可能很高，所以直接计算往往很困难，而且我们通常不知道$\phi(·)$是什么形式，所以我们引入**“核函数”(kernel function) **$\boldsymbol{\kappa(·,·)}$

:::info
$x_i$与$x_j$在特征空间中的内积等于它们在原始样本空间通过核函数$\kappa(x_i,x_j)$计算的结果.
:::

<br />所以我们不必去计算高位空间中的内积，则目标函数可以重写为：

$\begin{align}
&\min _{\alpha} \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} \alpha_{i} \alpha_{j} y_{i} y_{j}\kappa(x_i,x_j)-\sum_{i=1}^{m} \alpha_{i}\\
& s.t.  \sum_{i =  1}^{m} \alpha_{i} y_{i} = 0 \quad\alpha_{i} \geqslant 0, \quad i = 1,2, \cdots, m
\end{align}$<br />$\begin{array}{ll}

\end{array}$

求解后得到：$f(x)=\sum_{i=1}^{m} \alpha_{i} y_{i} \kappa\left(\boldsymbol{x}, \boldsymbol{x}_{i}\right)+b$，这一结果也称为“支持向量展式”<br /> <br />**定理：**只要一个对称函数所对应的核矩阵是半正定的，它就能作为核函数使用。即对于一个半正定核矩阵，总能找到一个与之对应的映射$\phi$

常用**“高斯核函数”（Gaussian kernel function）：**$\kappa\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\exp \left(-\frac{\left\|\boldsymbol{x}_{i}-\boldsymbol{x}_{j}\right\|^{2}}{2 \sigma^{2}}\right)$<br />其中$\sigma$为高斯核的带宽 ( width )，不使用核函数又称为**“线性核函数” **(**linear kernel**)

不同带宽下核函数图像：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655812110786-986f31ea-80db-43dc-84c5-dfaf73e7f5ea.png#clientId=u341dc832-4d0f-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=291&id=u74a240bc&name=image.png&originHeight=594&originWidth=1054&originalType=binary&ratio=1&rotation=0&showTitle=false&size=205689&status=done&style=none&taskId=uf71c9587-d474-41e3-ae0b-69820a024b3&title=&width=516)

- $C=1/λ$
- $C$ 较大时，相当于$λ$较小，可能会导致过拟合，高方差；
- $C$ 较小时，相当于$λ$较大，可能会导致低拟合，高偏差；
- $σ$较大时，可能会导致低方差，高偏差；
- $σ$较小时，可能会导致低偏差，高方差。

<br />

---

<a name="O5S93"></a>
## logistic 回归和SVM的比较

**相同点：**

1. LR和SVM都可以处理分类问题，且一般都用于处理线性二分类问题（在改进的情况下可以处理多分类问题
1. 两个方法都可以增加不同的正则化项，如L1、L2等等。所以在很多实验中，两种算法的结果是很接近的。

**不同点：**

1. 从目标函数来看，区别在于logistic 回归采用的是logistical loss，SVM主要采用的是hinge loss. 这两个损失函数的目的都是增加对分类影响较大的数据点的权重，减少与分类关系较小的数据点的权重。
1. SVM的处理方法是只考虑support vectors，也就是和分类最相关的少数点，去学习分类器。而logistic 回归中所有点都会起作用，只是通过非线性映射大大减小了离分类平面较远的点的权重，相对提升了与分类最相关的数据点的权重。
1. 逻辑回归一般应用梯度下降的方法求解，支持向量机应用SMO求解拉格朗日乘子，进而求解最优划分曲线方程。
1. 线性可分支持向量机本身就是结构风险最小化，而logistic 回归需要通过正则项权衡结构风险


**下面是一些普遍使用的准则：**
:::info
设$n$为特征数，$m$为训练样本数。

1. 如果相较于$m$而言，$n$要大许多，即训练集数据量不够支持我们训练一个复杂的非线性模型，我们选用逻辑回归模型或者不带核函数的支持向量机。

2. 如果$n$较小，而且$m$大小中等，例如$n$在 1-1000 之间，而$m$在10-10000之间，使用高斯核函数的支持向量机。

3. 如果$n$较小，而$m$较大，例如$n$在1-1000之间，而$m$大于50000，则使用支持向量机会非常慢，解决方案是选取并增加更多的特征，然后使用逻辑回归或不带核函数的支持向量机。
:::

神经网络在以上三种情况下都可能会有较好的表现，但是训练神经网络可能非常慢，选择支持向量机的原因主要在于它的代价函数是凸函数，不存在局部最小值。

---

<a name="lwZb3"></a>
## 相关代码
[sklearn.svm.SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=support_vectors_)<br />[sklearn.svm.LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC)<br />[sklearn.preprocessing.PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html?highlight=polynomialfeatures#sklearn.preprocessing.PolynomialFeatures)<br />[sklearn.svm.LinearSVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html?highlight=linearsvr#sklearn.svm.LinearSVR)<br />[sklearn.svm.SVR](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html?highlight=svr#sklearn.svm.SVR)
```python
from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=float("inf")) # 训练线性SVC模型
svm_clf.fit(X, y)

#像往常一样使用模型进行预测
svm_clf.predict(X_new)
```
```python
svm_clf = SVC(kernel="linear", C=10**9) # 通过改变C值来实现
svm_clf1.support_vectors_ # 获得支持向量
```
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


svm_clf = Pipeline([
        ("scaler", StandardScaler()), #标准化
        ("linear_svc", LinearSVC(C=1, loss="hinge", random_state=42)),
        # C=1，hinge损失函数，默认’squared_hinge’
    ])

svm_clf.fit(X, y)
svm_clf.predict(X_new)
```
```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

polynomial_svm_clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)), # 多项式转换器
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

polynomial_svm_clf.fit(X, y)
```
```python
from sklearn.svm import SVC

poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5)) # 多项式内核
        #超参数coef0控制的是模型受高阶多 项式还是低阶多项式影响的程度。
        #寻找正确的超参数值的常用方法是网格搜索
    ])
poly_kernel_svm_clf.fit(X, y)
```
```python
rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf", gamma=5, C=0.001)) # gamma为核系数（带宽）
    ])
rbf_kernel_svm_clf.fit(X, y)
```
```python
from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5, random_state=42) # epsilon控制宽度
svm_reg.fit(X, y)
```
```python
from sklearn.svm import SVR

# 使用核函数，degree为多项式次数
svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1, gamma="scale")
svm_poly_reg.fit(X, y)
```

---

<a name="mbMG5"></a>
# 贝叶斯分类器
<a name="hcnLO"></a>
## 前言——贝叶斯决策论

贝叶斯决策论(Bayesian decision theory) 是概率框架下实施决策的基本方法。

假设有种可能的类别标记，即，是将一个真实标记为的样本误分类为所产生的损失。基于后验概率可获得将样本分类为所产生的期望损失(expected loss) ，即在样本上的“条件风险”(conditional risk)<br />$R\left(c_{i} \mid \boldsymbol{x}\right)=\sum_{j=1}^{N} \lambda_{i j} P\left(c_{j} \mid \boldsymbol{x}\right)$<br />我们的任务是找到一个判定准则：$h:\mathcal{X} \longmapsto \mathcal{Y}$以最小化**总体风险：**<br />$R(h)=\mathbb{E}_{\boldsymbol{x}}[R(h(\boldsymbol{x}) \mid \boldsymbol{x})]$


对每个样本$x_i$，若能最小化条件风险 $R(h(x)|x)$，则总体风险$R(h)$将被最小化，这就产生了**“贝叶斯判定准则” (Bayes decision rule)**： 为最小化总体风险，只需在每个样本上选择那个能使条件风险 $R(c \mid x)$最小的类别标记，即<br />$h^{*}(x)=\underset{c \in \mathcal{Y}}{\arg \min } R(c \mid \boldsymbol{x})$<br />此时，$h^*$为**贝叶斯最优分类器 (Bayes optimal classifier)**，与之对应的总体风$R(h*)$称为贝叶斯风险(Bayes risk), $1 -R(h*)$反映了分类器所能达到的最好性能，即通过机器学习所能产生的模型精度的理论上限.


误判损失$\lambda_{ij}$可写作：$\lambda_{i j}=\left\{\begin{array}{ll}
0, & \text { if } i=j \\
1, & \text { otherwise }
\end{array}\right.$，可推出此时条件风险$R\left(c_{i} \mid \boldsymbol{x}\right)=1-P(c\mid x)$，于是则得到了：**最小化分类错误的贝叶斯最优分类器：**

$h^{*}(\boldsymbol{x})=\underset{c \in \mathcal{Y}}{\arg \max } P(c \mid \boldsymbol{x})=\underset{c \in \mathcal{Y}}{\arg \max }P(c \mid x)=\frac{P(c) P(x \mid c)}{P(x)}$

即对每个样本选择能使后验概率最大的类别标记.


欲使用贝叶斯判定准则来最小化决策风险，首先要获得后验概率，主要有两种策略：

1. “判别式模型”：根据样本$x$直接建模$P(c \mid x)$进行预测
1. **“生成式模型”**：根据贝叶斯定理，通过$P(x\mid c)，P(c)，P(x)$得到。本章主要探讨此策略。

- 类先验概率表示样本空间中各类样本的比例，根据大数定律，当训练集充足且数据独立同分布时，可通过各样本类别出现的频率对进行估计。
- 类条件概率，由于涉及的所有属性的联合概率，不易直接求。而且由于其可能的取值很多，训练集中可能根本没有，但“未被观测到”与“出现概率为零”通常是不同的，其估计方法后面的小节讨论。

---

<a name="bSGQB"></a>
## 极大似然估计

估计类条件概率的一种常用策略是先假定其具有某种确定的概率分布形式，再基于训练样本对概率分布的参数进行估计。事实上，概率模型的训练过程就是**参数估计（parameter estimation）**过程。

假设$P(x \mid c)$具有确定的形式并且被参数向量$\theta_c$唯一确定，则我们的任务就是利用训练集$D$估计参数$θ_c$. 我们将$P(x \mid c)$记为$P( x\mid θ_c)$

$D_c$表示训练集$D$中第$c$类样本组成的集合，假设这些样本是独立同分布的，则参数$\theta_c$对于数据集$D_c$的似然是：<br />$P\left(D_{c} \mid \boldsymbol{\theta}_{c}\right)=\prod_{\boldsymbol{x} \in D_{c}} P\left(\boldsymbol{x} \mid \boldsymbol{\theta}_{c}\right)$

使用对数似然（log-likehood）$LL(\theta_c)=\mathrm{log} P(D_c\mid \theta_c)=\sum_{\boldsymbol{x} \in D_c} \mathrm{log} P(\boldsymbol{x}\mid \theta_c)$，此时参数$\theta_c$的极大似然估计：<br />$\hat{\theta_c} = \underset{\theta_c}{\arg \max} LL(\theta_c)$<br />这种参数化的方法虽能使类条件概率估计变得相对简单，但估计结果的准确性严重依赖于所假设的概率分布形式是否符合潜在的真实数据分布。

---

<a name="tTISX"></a>
## 朴素贝叶斯分类器

为避免$P(x \mid c)$造成的困难，朴素贝叶斯分类器 (naÏve Bayes classifier)  采用了**“属性条件独立性假设” (attribute conditional ependence assumption)**：**对已知类别，假设所有属性相互独立**。这是一个较强的假设，朴素贝叶斯法由此得名，也称简单贝叶斯。

基于属性条件独立假设，贝叶斯定理$P(c\mid x)$可写为：<br />$P(c \mid \boldsymbol{x})=\frac{P(c) P(\boldsymbol{x} \mid c)}{P(\boldsymbol{x})}=\frac{P(c)}{P(\boldsymbol{x})} \prod_{i=1}^{d} P\left(x_{i} \mid c\right)$<br />其中$d$为属性数目，$x_i$为$\boldsymbol{x}$在第$i$个属性上的取值。

由于对所有类别$P(x)$相同，则可以得到**朴素贝叶斯分类器的表达式：**<br />$h_{n b}(\boldsymbol{x})=\underset{c \in \mathcal{Y}}{\arg \max } \frac{P(c)}{P(\boldsymbol{x})} \prod_{i=1}^{d} P\left(x_{i} \mid c\right)=\underset{c \in \mathcal{Y}}{\arg \max } P(c) \prod_{i=1}^{d} P\left(x_{i} \mid c\right)$

**类先验概率**的极大似然估计：$P(c)=\frac{|D_c|}{|D|}$，其中$D_c$表示训练集$D$中第$c$类样本组成的集合<br />**类条件概率**的极大似然估计：$P(x_i|c)=\frac{|D_{c,x_i}|}{|D_c|}$，其中$D_{c,x}$表示$D_c$中在第$i$个属性上取值为$x_i$的样本组成的集合。

类先验概率似然估计推导：
:::info
设 $P(c)$=$\theta$，$n=|D_c|=\sum_{i=1}^{N}I(y_i=c)$，即$N$个样本中有$n$个样本类别为$c$<br />对$N$个样本，$P(y_1=c,y_2=c,...,y_n=c,y_{n+1} \ne c,...,y_{N-1} \ne c,y_{N}\ne c)$=${\theta}^n{(1-\theta)}^{(N-n)}$

使用对数似然函数：$LL(\theta)$=$\mathrm{ln}({\theta}^n{(1-\theta)}^{(N-n)})$=$n\mathrm{ln}\theta+(N-n)\mathrm{ln}(1-\theta)$<br />两边取对$\theta$导数：$\frac{\partial LL(\theta)}{\partial \theta}=\frac{n}{\theta}-\frac{N-n}{1-\theta}$<br />取极值，偏导得0，则推出$\theta=\frac{n}{N}=\frac{|D_c|}{|D|}$
:::

---

<a name="ouXHe"></a>
## 学习与分类算法

**朴素贝叶斯算法：**
:::info

1. 计算先验概率和条件概率

$P(c_k)=\frac{|D_{c_k}|}{|D|}$，$P(x_i|c_k)=\frac{|D_{c_k,x_i}|}{|D_{c_k}|}$

2. 对给定实例使用朴素贝叶斯模型

$P(c_k) \prod_{i=1}^{d} P\left(x_{i} \mid c_k\right)$

3. 找到使分类器概率最大的类别，即确定实例$x$的类别

$h_{n b}(\boldsymbol{x}=\underset{c \in \mathcal{Y}}{\arg \max } P(c) \prod_{i=1}^{d} P\left(x_{i} \mid c\right)$
:::

例：有如下训练数据，试学习一个朴素贝叶斯分类器来确定$x=\{2,S\}$的类别$y$<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656730062595-5b07ad75-67d5-48e0-a94e-3d54ed744758.png#clientId=u2e07fef4-8d22-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=109&id=uce2b9d70&name=image.png&originHeight=109&originWidth=618&originalType=binary&ratio=1&rotation=0&showTitle=false&size=29719&status=done&style=none&taskId=u9a3cecf8-52bc-477e-9721-664650f0104&title=&width=618)

1. **计算先验概率和条件概率：**

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656730224559-10bb1510-5f12-4258-9d74-4c8e36f47cff.png#clientId=u2e07fef4-8d22-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=194&id=ua34564bd&name=image.png&originHeight=223&originWidth=622&originalType=binary&ratio=1&rotation=0&showTitle=false&size=65844&status=done&style=none&taskId=ueafac7ff-09a7-4f6f-ad71-0b6f4a8cab4&title=&width=541)

2. **对**$x=\{2,S\}$**使用朴素贝叶斯，计算：**

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656730277747-77e6c36c-82ef-452b-8b32-d29f5054b6b5.png#clientId=u2e07fef4-8d22-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=89&id=u983b8d54&name=image.png&originHeight=95&originWidth=548&originalType=binary&ratio=1&rotation=0&showTitle=false&size=24422&status=done&style=none&taskId=u635d8202-b5fe-4668-9fe4-cfc850fc2a3&title=&width=515)

3. **找到使分类器概率最大的类别，即确定实例**$x$**的类别：**

因为$P(Y=-1) P\left(X^{(1)}=2 \mid Y=-1\right) P\left(X^{(2)}=S \mid Y=-1\right)$最大，所以$y=-1$

---

<a name="kgoqO"></a>
## 贝叶斯估计
用极大似然估计条件概率很可能会出现所要估计的概率值为0的情况，为了避免其他属性携带的信息被训练集中未出现的属性值“抹去”，在估计概率值时通常要进行“平滑”(smoothing) ，采用**贝叶斯估计。**

**修正后的类先验概率：**$P(c)=\frac{|D_c|+\lambda}{|D|+\lambda K}$<br />**修正后的类条件概率：**$P(x_i|c)=\frac{|D_{c,x_i}|+\lambda}{|D_c|+\lambda S_i}$<br />其中$K$为样本可能的类别数，$S_i$为第$i$个属性可能的取值数。式中$\lambda > 0$，等价于在随机变量各个取值的频数上赋予一个正数，可以发现若$\lambda=0$时，即为极大似然估计。

贝叶斯估计中我们常取$\lambda=1$，此时称为**“拉普拉斯修正”** (Laplacian correction）

训练集变大时，修正过程所引入的先验（prior）的影响也会逐渐变得可忽略，使得估计值趋向于实际概率。

---

<a name="yZYGq"></a>
# 聚类(Clustering)

<a name="REedF"></a>
## 无监督学习：简述

无监督学习是从无标注的数据中学习数据的统计规律或者说内在结构的机器学习，主要包括聚类、降维、概率估计。此类学习任务中研究最多、应用最广的是**“聚类” (clustering)**

无监督学习三要素：

1. 模型。模型就是函数$z=g_{\theta}(x)$，条件概率分布$P_{\theta}(z|x)$，或条件概率分布$P_{\theta}(x|z)$
1. 策略。在不同问题中策略有不同的形式，但都可以表示为目标函数的优化。
1. 算法。通常为迭代算法，通过迭代达到目标函数的最优化，比如“梯度下降法”。


在一个典型的监督学习中，我们有一个有标签的训练集，我们的目标是找到能够区分正样本和负样本的决策边界，与此不同的是，在非监督学习中，我们的数据没有附带任何标签，我们拿到的数据就是这样的：在这里我们有一系列点，却没有标签。因此，我们的训练集可以写成只有$x^{(1)}$,$x^{(2)}$…..一直到$x^{(m)}$。我们没有任何标签$y$

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655888376284-c17161fd-793e-4aa4-b502-ec5c27ac3f63.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=243&id=u95644b5c&name=image.png&originHeight=350&originWidth=488&originalType=binary&ratio=1&rotation=0&showTitle=false&size=57238&status=done&style=none&taskId=ub4024782-8265-4361-8e31-402e75f8184&title=&width=338.3999938964844)

聚类试图将数据集中的样本划分为若干个通常是不相交的子集，每个子集称为一个**“簇"”(cluster). **聚类的结果可用包含$m$个元的**簇标记向量**$\lambda=(\lambda_1;\lambda_2;...;\lambda_m)$表示，即$x_j \in C_{\lambda_j}$

如果一个样本只能属于一个类，则称为**“硬聚类”(hard clustering)**，硬据类时，每一个样本属于某一类$z_i=g_{\theta}(x_i)\ , i=1,2,...,m$

如果一个样本可以属于多个类，则称为**“软聚类”(soft clustering)**，软聚类时，每一个样本依概率属于每一个类$P_{\theta}(z_i|x_i) , i=1,2,...,m$

---

<a name="IagWq"></a>
## 聚类任务有效性指标

聚类的核心概念是相似度 (similarity) 或距离 ( distance )。

聚类的性能度量大致有两类：一类是将聚类结果与某个“参考模型” (reference model) 进行比较，称为“外部指标” (external dex)；另一类是直接考察聚类结果而不利用任何参考模型，称为“内部指标” (internal index). 

:::info
**外部指标：**

1. **Jaccard系数 (Jaccard Coefficient)：**$JC=\frac{a}{a+b+c}$

2. **FM指数 (Fowles and Mallows Index )：**$\mathrm{FMI}=\sqrt{\frac{a}{a+b} \cdot \frac{a}{a+c}}$

3. **Rand指数 (Rand Index)：**$\mathrm{RI}=\frac{2(a+d)}{m(m-1)}$

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655913394551-bbe95477-2805-43e6-9d82-03b5b024aed6.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=162&id=u3087dbd9&name=image.png&originHeight=245&originWidth=695&originalType=binary&ratio=1&rotation=0&showTitle=false&size=60557&status=done&style=none&taskId=ud8060fcd-8d7a-4e1a-bf9f-969e3c09192&title=&width=460)

**上述性能度量的结果值均在 [0, 1]区间，值越大越好**
:::

:::info
**内部指标：**

1. **DB指数 (Davies-Bouldin Index)：**$\mathrm{DBI}=\frac{1}{k} \sum_{i=1}^{k} \max _{j \neq i}\left(\frac{\operatorname{avg}\left(C_{i}\right)+\operatorname{avg}\left(C_{j}\right)}{d_{\operatorname{cen}}\left(\boldsymbol{\mu}_{i}, \boldsymbol{\mu}_{j}\right)}\right)$

<br />

2. **Dunn指数 (Dunn Index)：**$\mathrm{DI}=\min _{1 \leqslant i \leqslant k}\left\{\min _{j \neq i}\left(\frac{d_{\min }\left(C_{i}, C_{j}\right)}{\max _{1 \leqslant l \leqslant k} \operatorname{diam}\left(C_{l}\right)}\right)\right\}$

$\mu$代表簇$C$的中心点$\mu=\frac{1}{|C|} \sum_{1 \leqslant i \leqslant|C|} \boldsymbol{x}_{i}$;$avg(C)$对应簇$C$内样本间的平均距离； $diam(C)$对应簇内样本间的最远距离；$d_{min}(C_i,C_j)$对应两簇最近样本间距离；$d_{cen}(C_i.C_j)$对应两簇中心点间距离；

**DBI的值越小越好，DI的值越大越好**
:::

在聚类中，可以将样本集合看作是向量空间中点的集合，以该空间的距离表示样本之间的相似度。<br />对于函数$\boldsymbol{dist(·, ·)}$若它是一个“距离度量”(distance measure) 则，需要满足一些基本性质：<br />（1）非负性；（2）同一性；（3）对称性；（4）直递性（三角不等式）

:::info
对于给定样本$x_i=(x_{i1},x_{i2};...;x_{in})$与$x_j=(x_{j1},x_{j2};...;x_{jn})$，最常用的是**“闵可夫斯基距离”(Minkowski distance)，**定义为：<br />$\operatorname{dist}_{\mathrm{mk}}\left(\boldsymbol{x}_{i}, \boldsymbol{x}_{j}\right)=\left(\sum_{u=1}^{n}\left|x_{i u}-x_{j u}\right|^{p}\right)^{\frac{1}{p}}$<br />这里$p\ge1$，为$x_i-x_j$的$L_P$范数.
:::

$p=2$时，闵可夫斯基距离即为**“欧氏距离 ”(Euclidean distance)；**$p=1$时为**“曼哈顿距离”(Manhattan distance)；**$p=\infin$时为**“切比雪夫距离”( Chebyshev distance)**

**闵可夫斯基距离可用于有序属性。**

:::info
无序属性可采用$\mathrm{VDM}$**(Value Difference Metric)。**令$m_{u,a}$表示在属性$u$上取值为$a$的样本数，$m_{u,a,i}$表示在第$i$个样本簇在属性$u$上取值为$a$的样本数，$k$为样本簇数，则属性$u$上两个离散值$a$与$b$之间的 VDM 距离为：<br />$\operatorname{VDM}_{p}(a, b)=\sum_{i=1}^{k}\left|\frac{m_{u, a, i}}{m_{u, a}}-\frac{m_{u, b, i}}{m_{u, b}}\right|^{p}$
:::

用于相似度度量的距离未比满足距离度量所有的基本性质，尤其是直递性。这样的距离称为“非度量距离”



:::info
**“马哈拉诺比斯距离”(Mahalanobis distance)，简称马氏距离。**考虑各分量（特征）之间的相关性，并于各分量的尺度无关。给定一个样本集合$X=(x_{ij})_{m\times n}$，其协方差矩阵记作$S$，样本$x_i$与$x_j$之间的马氏距离定义为：<br />$d_{i j}=\left[\left(x_{i}-x_{j}\right)^{T} S^{-1}\left(x_{i}-x_{j}\right)\right]^{\frac{1}{2}}$
:::

当$S$为单位矩阵时，即样本数据的各个分量互相独立且各个分量的方差为1时， 马氏距离就是欧氏距离，所以马氏距离是欧氏距离的推广。

:::info
**“Pearson相关系数”。**绝对值越接近1，表示样本越相似。<br />${{\rho }_{XY}}=\frac{\operatorname{cov}(X,Y)}{{{\sigma }_{X}}{{\sigma }_{Y}}}=\frac{E[(X-{{\mu }_{X}})(Y-{{\mu }_{Y}})]}{{{\sigma }_{X}}{{\sigma }_{Y}}}=\frac{\sum\limits_{i=1}^{n}{(x_i-{{\mu }_{X}})(y_i-{{\mu }_{Y}})}}{\sqrt{\sum\limits_{i=1}^{n}{{{(x_i-{{\mu }_{X}})}^{2}}}}\sqrt{\sum\limits_{i=1}^{n}{{{(y_i-{{\mu }_{Y}})}^{2}}}}}$
:::

:::info
**“夹角余弦”。**越接近于1，表示样本越相似。样本对于给定样本$x_i=(x_{i1},x_{i2};...;x_{in})$与<br />$x_j=(x_{j1},x_{j2};...;x_{jn})$的夹角余弦为：$s_{i j}=\frac{\sum\limits_{k=1}^{m} x_{k i} x_{k j}}{\left[\sum\limits_{k=1}^{m} x_{k i}^{2} \sum\limits_{k=1}^{m} x_{k j}^{2}\right]^{\frac{1}{2}}}$
:::


---

<a name="diECM"></a>
## 原型聚类

“原型”指样本空间中具有代表性的点。原型聚类即“基于原型的聚类”，此类算法假设聚类结构能通过一组原型刻画，通常算法先对圆形进行初始化，然后对原型进行迭代求解。采用不同的原型表示，不同的求解方式将产生不不同的算法。


<a name="ZJ4Tt"></a>
### k-均值算法 k-means

$k$均值算法针对聚类所得簇划分$C=\{C_i,C_2,...,C_k\}$最小化平方误差：$E=\sum_{i=1}^{k} \sum_{\boldsymbol{x} \in C_{i}}\left\|\boldsymbol{x}-\mu_{i}\right\|_{2}^{2}$<br />$E$值越小，簇内样本相似度越高，但此最小化问题是一个NP难问题，$k$均值算法采用贪心策略，通过**迭代**优化来近似求解。

对于数据集中的每一个数据，按照距离$k$个中心点的距离，将其与距离最近的中心点关联起来，与同一个中心点关联的所有点聚成一类。

:::info
$k$**均值聚类算法每次迭代步骤：**

1. 首先随机选择$k$个样本作为**聚类中心**（**cluster centroids**），计算每个样本到类中心的距离，将每个样本指派到与其最近的中心的类中，构成新聚类
1. 计算各类中样本的均值，作为新的聚类中心，重复以上步骤直到收敛。
:::

$k$聚类算法时间复杂度为$O(mnk)$<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655918725212-a1b25681-a17b-49ab-8786-b747f102ee0f.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=462&id=uf519bf6e&name=image.png&originHeight=762&originWidth=854&originalType=binary&ratio=1&rotation=0&showTitle=false&size=258813&status=done&style=none&taskId=ufd389f97-0f8c-4544-adc5-3fa6c9dce03&title=&width=518.2000122070312)

**初始化方法的改进：**
:::info
使用一次随机的初始化聚类中心可能不会收敛到最优点，一种解决方案是使用不同的随机初始化多次运行，并保留最优解。根据模型**最小惯性**原则选取最优点，惯性即每个实例与其最接近的中心点之间的均方距离。
:::

$k$**值的选择：**
:::info

1. 惯性作为集群$k$的函数曲线时曲线总时递减的，但曲线通常包含一个称为“肘”的**拐点**，可以选此拐点对应的$k$值

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656768521391-b8abb9d5-943b-46d5-a473-250ab59cb49d.png#clientId=u54e96ed9-7a42-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=262&id=ubd35167b&name=image.png&originHeight=380&originWidth=522&originalType=binary&ratio=1&rotation=0&showTitle=false&size=36371&status=done&style=none&taskId=uceba3aaf-0b8b-43a5-870d-8b5ec40f579&title=&width=360)

2. 一种更精确的方法是使用**轮廓分数**，它是所有实例的平均轮廓系数（silhouette coefficient），选取轮廓分数最大时对应的$k$值。实例的轮廓系数等于$\frac{b-a}{\max(a,b)}$，其中$a$是与同一集群中其他实例的平均距离（即集群内平均距离），$b$是平均最近集群距离（即到下一个最近集群实例的平均距离）
2. 找到最大的**Gap statistic**所对应的$k$，**Gap Statistic**定义为：

$\operatorname{Gap}(K)=E\left(\log D_{k}\right)-\log D_{k}$<br />其中$E(logD_k)$是$logD_k$的期望，一般通过蒙特卡洛模拟产生。我们在样本所在的区域内按照均匀		分布随机地产生和原始样本数一样多的随机样本，并对这个随机样本做$k$均值，得到一个$D_k$；重复	       多次就可以计算出$E(logD_k)$的近似值。
:::

$k$**均值聚类特点：**

1. 类别数$k$需事先指定，但$k$很难确定，一般是尝试不同$k$值检验各聚类结果的质量，选取最优的$k$值。聚类结果的质量可用类的平均直径表示。也可以使用不同$k$下代价突变时的k值
1. 该方法属于启发式方法，不能保证收敛到全局最优，选择不同的初始中心会得到不同的聚类结果。
1. 该方法不适用于任意形状的簇，仅在凸形簇结果上效果较好
1. 样本点只能被划分到单一的类中。 

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655919072873-f660520a-8407-483f-a4a7-31265e9fc1b9.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=237&id=u9501d7e1&name=image.png&originHeight=394&originWidth=703&originalType=binary&ratio=1&rotation=0&showTitle=false&size=201386&status=done&style=none&taskId=u78181b25-2049-49f1-802c-cdcb4597476&title=&width=422.4000244140625)


算法描述：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655920461387-e82d6a70-b9ca-4070-8cb3-4d87b250c57f.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=394&id=u472d8d4b&name=image.png&originHeight=594&originWidth=794&originalType=binary&ratio=1&rotation=0&showTitle=false&size=157618&status=done&style=none&taskId=ud86bcfba-6119-4be4-baeb-30df46436f9&title=&width=527.2000122070312)

---

<a name="exzQh"></a>
### 学习向量量化 LVQ

**“学习向量量化” (Learning Vector Quantization，LVQ)** 也是试图找到一组原型向量来刻画聚类结构， 但与一般聚类算法不同的是， LVQ 假设数据样本带有类别标记，学习过程利用样本的这些监督信息辅助聚类。

**LVQ** 目标是学得一组$n$维原型向量 $\{p_1,p_2,...,p_q\}$ 每个原型向量代表一个聚类簇。每个原型向量$p_i$定义了与之相关的一个区域$R_i$，该区域中每个样本与$p_i$的距离不大于它与其他原型向量的距离。

:::info
**LVQ算法每次迭代步骤**<br />随机选取一个有标记的训练样本，找出与其距离最近的原型向量，并根据两者标记是否一致来对原型向量进行相应更新。对样本$x_j$若与其最近的原型向量$p_{i*}$   标记类别相同，则令$p_{i*}$向$x_j$方向靠拢，更新原型向量为：$p'=o_{i*}+\eta \cdot (x_j-p_{i*})$，学习率$\eta \in (0,1)$；若不同，则远离。
:::

迭代示例：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655921881825-7f26a165-2704-44fb-a62e-e319ad933dcf.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=415&id=u130191ed&name=image.png&originHeight=766&originWidth=836&originalType=binary&ratio=1&rotation=0&showTitle=false&size=258434&status=done&style=none&taskId=ue6c2f019-57e2-4566-afb7-4b77b18705a&title=&width=452.79998779296875)


**算法描述：**<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655921856567-89301187-a541-408f-b615-54c178deae44.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=326&id=u38e764e5&name=image.png&originHeight=482&originWidth=716&originalType=binary&ratio=1&rotation=0&showTitle=false&size=148465&status=done&style=none&taskId=ubd9e8c4d-68cb-4d95-99d5-176b8c5d3bd&title=&width=484.79998779296875)

---

<a name="cpR6m"></a>
### 高斯混合聚类

---

<a name="Cm9oD"></a>
## 密度聚类 DBSCAN

密度聚类算法从样本密度的角度来考察样本之间的可连接性，并给予可连接样本不断扩展聚类簇以获得最终的聚类结果。

**"DBSCAN" (Density-Based Spatal Clustering of Application with Noise ) **是一种著名的密度聚类算法，它基于一组“邻域”(neighborhood) 参数$(\mathrm{\epsilon},MinPts)$来刻画样本分布的紧密程度。

:::info
**密度聚类相关概念：**

- $\mathrm{\epsilon}$**-邻域**：样本集$D$中与$x_j$距离不大于$\mathrm{\epsilon}$的样本集合
- **核心对象：**若$x_j$的$\mathrm{\epsilon}$-邻域至少包含$MinPts$个样本，则$x_j$是一个核心对象。即$\mathrm{\epsilon}$-邻域密度达到设定阈值的点
- **密度直达(directly density-reachable) ：**若$x_j$位于$x_i$的$\mathrm{\epsilon}$-邻域中，且$x_i$是核心对象，则称$x_j$由$x_i$密度直达
- **密度可达(density-reachable)：**若存在序列$\{q_0,q_1,…,q_n\}$，对任意$q_{i+1}$由$q_i$是密度直达的，则称从$q_0$到$q_n$密度可达，这实际上是密度直达的“传播”
- **密度相联(density-connected)：**对于$x_i$与$x_j$，若存在$x_k$使得$x_i$与$x_j$均由$x_k$密度可达则称$x_i$与$x_j$密度相联
- **噪声(noise) 或异常(anomaly样本)：**不属于任何簇的样本
:::


**DBSCAN **将“簇”定义为：由密度可达关系导出的最大密度相联样本集合。即若$x$为核心对象，由$x$密度可达的所有样本组成的集合记为$X=\{x' \in D|x'由x密度可达\}$，$X$即为满足连接性与最大性的“簇”.

:::info
**DBSCAN 算法步骤**

1. 根据给定邻域参数$(\mathrm{\epsilon},MinPts)$找出所有核心对象加入集合$\Omega$
1. 从$\Omega$中随机选取一个核心对象为出发点，找出由其密度可达的样本生成聚类簇$C_i$，并将$C_i$中包含对象在$\Omega$中去除
1. 迭代执行第2步，直到$\Omega$为空
:::

迭代示例：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655952109064-1bf5f360-ef42-4cc5-a9f9-3fd7e31da3fd.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=473&id=uccfd56ad&name=image.png&originHeight=765&originWidth=857&originalType=binary&ratio=1&rotation=0&showTitle=false&size=218493&status=done&style=none&taskId=u8f9179ff-0eb2-4e7b-a449-3edc61776f8&title=&width=529.6000366210938)

**使用两个不同邻域半径的DBSCAN聚类**<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656770058904-d811eaaa-0bcd-4ace-ad1e-66d539f0890b.png#clientId=u54e96ed9-7a42-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=224&id=uc6167615&name=image.png&originHeight=224&originWidth=620&originalType=binary&ratio=1&rotation=0&showTitle=false&size=88898&status=done&style=none&taskId=ua075930b-e125-476a-8bfb-c74c484de97&title=&width=620)

**密度聚类算法特点：**

1. 无需指定$k$值，即簇的个数，但需要指定$\mathrm{\epsilon}$,$MinPts$（$\mathrm{\epsilon}$可根据$k$-距离，寻找突变点来确定）
1. 可以发现任意形状的簇
1. 擅长寻找离群点（检测任务）
1. 处理高维数据比较困难（可以先降维）

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655952622788-8980eba2-ac52-41d3-8d66-e27f39b94ce7.png#clientId=u836e3c4a-692e-4&crop=0.005&crop=0.0189&crop=1&crop=1&from=paste&height=223&id=u4fb3e860&name=image.png&originHeight=331&originWidth=493&originalType=binary&ratio=1&rotation=0&showTitle=false&size=236986&status=done&style=none&taskId=u57eee686-830d-4f09-a21c-ffb1e792f62&title=&width=332)


**算法描述：**<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655952142551-d4c929e7-d146-4abf-8269-ab321edbd1e4.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=494&id=uabe0a72e&name=image.png&originHeight=782&originWidth=670&originalType=binary&ratio=1&rotation=0&showTitle=false&size=202562&status=done&style=none&taskId=u6f309fc4-10e9-402b-9de1-53e94807ba3&title=&width=423)

---

<a name="Im2WY"></a>
## 层次聚类

层次聚类有**聚合 (agglomerative) **或**自下而上 (bottom-up)** 聚类，**分裂 (divisive)** 或**自上而下 (top-down)** 聚类两种方法。

聚合聚类开始将每个样本各自分到 个类:之后将相距最近的两类合井，建立一个新的类，重复此操作直到满足停止条件。分裂聚类开始将所有样本分到 个类:之后将己有类中相距最远的样本分到两个新的类，重复此操作直到满足停止条件。

聚合聚类需要确定三要素：

1. 距离或相似度
1. 合并规则（一般是类间距离最小）
1. 停止条件（可以是类的个数或类直径达到阈值）

**AFNES (AGglomera-tive NESting) 算法**是一种采用自下而上聚合策略的层次聚类算法。

:::info
**AFNES算法步骤：**

1. 先将数据集中的每个样本看作初始聚类簇，并初始化相应距离矩阵
1. 不断合并**距离最近**的聚类簇，并对合并得到的聚类簇的距离矩阵进行更新。不断重复该步。直到达到预设的聚类簇数。
:::

合并过程可以用树状图表示<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655953476115-4fd1e369-d569-4e74-81d2-1ef54033bc61.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=570&id=u61b70653&name=image.png&originHeight=712&originWidth=889&originalType=binary&ratio=1&rotation=0&showTitle=false&size=94252&status=done&style=none&taskId=u90b0c187-d1fa-4c6b-b280-09dcf49105a&title=&width=711.2)<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655953498952-6ddb4ff5-e7db-4242-afca-61a0fe8dff1c.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=602&id=ubb810f3c&name=image.png&originHeight=753&originWidth=836&originalType=binary&ratio=1&rotation=0&showTitle=false&size=262509&status=done&style=none&taskId=u5da75b7b-9cde-4062-a382-7e4fab83123&title=&width=668.8)<br />**AFNES **通常使用$d_{min}$，$d_{max}$，$d_{avg}$

**平均距离：**$d_{\mathrm{avg}}\left(C_{i}, C_{j}\right)=\frac{1}{\left|C_{i}\right|\left|C_{j}\right|} \sum_{\boldsymbol{x} \in C_{i}} \sum_{\boldsymbol{z} \in C_{j}} \operatorname{dist}(\boldsymbol{x}, \boldsymbol{z})$

**算法描述：**<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655953620164-3ac00d71-8587-4869-b8b3-8a94f7938839.png#clientId=u836e3c4a-692e-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=555&id=ufe49b181&name=image.png&originHeight=781&originWidth=525&originalType=binary&ratio=1&rotation=0&showTitle=false&size=183164&status=done&style=none&taskId=ubf98e546-6177-4325-90bf-f1491d7bea1&title=&width=373)

---

<a name="Y3yT7"></a>
## 本章相关代码
[sklearn.cluster.KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html?highlight=kmeans#sklearn.cluster.KMeans)<br />[sklearn.cluster.MiniBatchKMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html?highlight=kmeans#sklearn.cluster.MiniBatchKMeans)<br />[sklearn.metrics.silhouette_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html?highlight=silhouette_score#sklearn.metrics.silhouette_score)<br />[sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV)<br />[sklearn.cluster.AgglomerativeClustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html?highlight=agglomerativeclustering#sklearn.cluster.AgglomerativeClustering)
```python
from sklearn.cluster import KMeans

k = 5 # 给出聚类数
kmeans = KMeans(n_clusters=k, init="random", random_state=42)
y_pred = kmeans.fit_predict(X)
kmeans.predict(X_new) # 将新实例分配给中心点最接近的集群

kmeans.labels_ # 标签副本
kmeans.cluster_centers_ # 聚类中心
kmeans.transform(X_new) # 测量每个实例到每个中心点的距离
kmeans.inertia_ # 模型惯性
kmeans.score(X) # 返回负惯性
```
```python
from sklearn.cluster import MiniBatchKMeans

minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
minibatch_kmeans.fit(X)
```
```python
kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)] # 在[1,10)中寻找
inertias = [model.inertia_ for model in kmeans_per_k]

plt.plot(range(1, 10), inertias, "bo-")
plt.show()
```
```python
from sklearn.metrics import silhouette_score

# 基本格式 silhouette_score(X, kmeans.labels_)

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)] # 在[1,10)中寻找
silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]

# 做出轮廓分数关于k值图像
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.show()
```
```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)),
])
pipeline.fit(X_train, y_train)

param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

grid_clf.best_params_ # 输出最优k值
grid_clf.score(X_test, y_test) # 输出得分
```
```python
kmeans = KMeans(n_clusters=50)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx] # 得到代表性样本

# 标签传播
percentile_closest = 20 # 传播到最接近中心点20%的实例

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]

for i in range(k):
    in_cluster = (kmeans.labels_ == i) # 判断每个实例是否在集群i中的 布尔列表
    cluster_dist = X_cluster_dist[in_cluster] # 找到集群i中的元素到中心的距离
    cutoff_distance = np.percentile(cluster_dist, percentile_closest) #找到分位数
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1 # 在集群中但超过范围的距离置-1
 
partially_propagated = (X_cluster_dist != -1) # 返回距离不等于-1的索引
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]
```
```python
from sklearn.cluster import AgglomerativeClustering
# lingkage设为average时，每次将簇中所有点之间平均距离最小的两个簇合并，即AFNES算法
# 默认为ward，使得所有簇中的方差增加最小，适用于大多数数据集
agg = AgglomerativeClustering(n_clusters=3，linkage='average')
assignment = agg.fit_predict(X)
```
```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.05, min_samples=5) # 设置两个参数
dbscan.fit(X)
# dbscan无predict()，即它无法预测新实例属于哪个集群
# 可以先使用DBSCAN进行分类，再使用k近邻法等对新样本进行预测，见后章章节

dbscan.fit_predict(X) # 训练并输出X中每个实例对应的标签
dbscan.labels_ # 得到所有标签
dbscan.components_ # 得到所有核心实例
dbscan.core_sample_indices_ # 得到所有核心实例的索引
```

---

<a name="KON3D"></a>
# 分类任务中的 k-近邻法
<a name="K2jI4"></a>
## k-近邻法概述

**“**$k$**-近邻法”(k-Nearest Neighborhood, kNN) **是一种常用的监督学习方法。

:::info
$k$**-近邻法主要思想：**给定测试样本，训练集中基于某种距离度量找出与该测试样本最邻近的$k$个实例，根据这$k$个邻居的信息来进行预测。分类任务中可以使用“投票法”；回归任务中可使用“平均法”。
:::

本节讨论分类任务中的$k$-近邻法，使用投票法，即某新输入样本最邻近$k$个训练实例多数属于某个类，则该新输入样本就属于哪个类。

$k$-近邻法的特殊情况时$k=1$时，此时称为**“最邻近算法”。**最邻近法将实例$x_i$的类$y_i$作为其单元中所有点的类标记（class lable）。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656035299071-a63251cc-822c-457a-a155-27db7022755f.png#clientId=uf2379bb7-f3d8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=190&id=u3823398b&name=image.png&originHeight=323&originWidth=573&originalType=binary&ratio=1&rotation=0&showTitle=false&size=108831&status=done&style=none&taskId=u90b7967c-878e-4cd2-8252-16c557cb4be&title=&width=337.7200012207031)

**算法描述：**
:::info

1. 输入训练数据集：$T=\{(x_1,y_1,(x_2,y_2),...,(x_m,y_m))\}$，其中$x_i \in\mathcal{{\chi}}\subseteq  \mathbb{R}^n$为实例的特征向量，$y_{i} \in \mathcal{Y}=\left\{c_{1}, c_{2}, \cdots, c_{K}\right\}$为实例所属类别，$i=1,2,...,m$
1. 输入新实例$x$，根据给定的距离度量，在训练集$T$中找出与$x$最邻近的$k$个点，涵盖这$k$个点$x$的邻域记作$N_k(x)$
1. 在$N_k(x)$中根据分类决策规则（如投票法）决定$x$所属类别，即

$y=\arg \max _{c_{j}} \sum_{x_{i} \in N_{k}(x)} I\left(y_{i}=c_{j}\right), \quad i=1,2, \cdots, m ; j=1,2, \cdots, K$<br />$I$为指示函数，即当$y_i=c_j$时$I=1$否则为0
:::

---

<a name="r73yc"></a>
## k 近邻模型
:::info
$k$-近邻法中，当训练集、距离度量(如欧氏距离)、$k$值及分类决策规则（如投票法）确定后，对于任何 个新的输入实例，它所属的类唯一地确定。$k$-近邻法无显式的训练过程。

$k$-近邻模型的特征空间一般是$\mathbb{R}^n$。一般使用的距离是欧氏距离，也可以是其他距离，如$\mathrm{L}^p$距离。
:::

$k$-近邻法的模型对应特征空间的一个划分<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656039468032-6e922ce0-0451-485b-a123-ba192fe73344.png#clientId=uf2379bb7-f3d8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=272&id=u9b46e363&name=image.png&originHeight=511&originWidth=614&originalType=binary&ratio=1&rotation=0&showTitle=false&size=186873&status=done&style=none&taskId=u78eda2f7-6df0-460c-81d0-d73216a6eb5&title=&width=326.9599914550781)

**k值选择**
:::info

- $k$值较小，近似误差减小，估计误差增大，意味着整体模型复杂度较大，易发生过拟合
- $k$值较大，估计误差减小，近似误差增大，意味着整体模型复杂度较小。

$k$**值一般取一个较小的值，然后通过交叉验证法来选取最优的**$k$**值**
:::

---

<a name="YiLmh"></a>
## 分类决策规则

- $k$邻近法分类决策规则一般是投票法（多数表决），分类函数是$f: \mathbb{R}^{n} \rightarrow\left\{c_{1}, c_{2}, \cdots, c_{K}\right\}$

- $k$**最近邻分类的误分类率：**$\frac{1}{k} \sum_{x_{i} \in N_{k}(x)} I\left(y_{i} \neq c_{j}\right)=1-\frac{1}{k} \sum_{x_{i} \in N_{k}(x)} I\left(y_{i}=c_{j}\right)$**，**要使误分类率最小（经验风险最小）则必须使得$\frac{1}{k} \sum_{x_{i} \in N_{k}(x)} I\left(y_{i}=c_{j}\right)$最大，投票法等价于**经验风险最小化**。

- **最近邻法误分类的概率**：$P(e r r)=1-\sum_{c \in \mathcal{Y}} P(c \mid \boldsymbol{x}) P(c \mid \boldsymbol{z})\leqslant 2 \times\left(1-P\left(c^{*} \mid \boldsymbol{x}\right)\right)$，其中$c^{*}=\arg \max _{c \in \mathcal{Y}} P(c \mid \boldsymbol{x})$表示最优分类器的结果。可以看到最近邻分类器的泛化错误率不超过**贝叶斯最优分类器**错误概率的两倍。

---

<a name="hsCE6"></a>
## k近邻法的实现：kd树

$kd$树是二叉树，表 维空间的一个划分 (Cpartition) 。构造$kd$树相当于不断地用垂直于坐标轴的超平面将 间切分，构成一系列的$k$维超矩形区域。$kd$树的每个结点对应于一个$k$维超矩形区域

特征空间划分示意图：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656041505550-a188a995-1a12-4719-95bf-1be57bf579ff.png#clientId=uf2379bb7-f3d8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=250&id=ud18e1c65&name=image.png&originHeight=391&originWidth=405&originalType=binary&ratio=1&rotation=0&showTitle=false&size=29494&status=done&style=none&taskId=u094ece0f-beae-40fd-8bfe-4cf1145e3b2&title=&width=259.2)

**构造平衡**$kd$**树算法：**
:::info
![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656041418499-8142c366-c8f1-4716-8501-4ddbc69266a1.png#clientId=uf2379bb7-f3d8-4&crop=0&crop=0.0509&crop=1&crop=1&from=paste&height=452&id=Hky4j&name=image.png&originHeight=751&originWidth=965&originalType=binary&ratio=1&rotation=0&showTitle=false&size=458046&status=done&style=none&taskId=u0404a31e-8271-4e12-a56b-7856c69dd7b&title=&width=581)
:::

**用**$kd$**树的最近邻搜索算法：**
:::info
![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656041452313-62e9dd55-fc68-4513-9312-a4a687c8c5e0.png#clientId=uf2379bb7-f3d8-4&crop=0&crop=0.0689&crop=1&crop=1&from=paste&height=427&id=CxEQH&name=image.png&originHeight=723&originWidth=957&originalType=binary&ratio=1&rotation=0&showTitle=false&size=408953&status=done&style=none&taskId=u61704bf3-acea-42f0-92ba-c91a7ca981e&title=&width=565)
:::

利用$kd$树可以省去对大部分数据点的搜索，从而减少搜索的计算量。

---

<a name="iexyh"></a>
## 本章相关代码

[sklearn.neighbors.KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html?highlight=kneighborsclassifier#sklearn.neighbors.KNeighborsClassifier)
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X_train)

knn = KNeighborsClassifier(n_neighbors=50)

# 课先使用聚类算法（如DBSCAN进行分类），再使用近邻法预测新样本
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])
knn.predict(X_new) # 输出样本类标签
knn.predict_proba(X_new) # 输出样本对每个分类的概率
```

---

<a name="lcXoS"></a>
# 降维(Dimensionality Reduction)

<a name="wAwmk"></a>
## 降维概述

统计分析中，数据的变量之间可能存在相关性，以致增加了分析的难度。于是，考虑由少数不相关的变量来代替相关的变量。事实上，在高维情形下 现的数据样本稀疏、 距离计算困 难等问是所有机器学习方法共同面 的严重障碍， 被称为" 维数灾难" (curse of dimensionality) .

**缓解维数灾难的 个重要途径是降维(dimension red uction) 亦称"“维数约简”，即通过某种数学变换将原始高维属性空间转变为 一个低维“子空间”(subspace)**

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655985757708-df5ef49e-0d2b-40d0-a891-ec5686066cdd.png#clientId=u76c62888-11e0-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=223&id=u891da214&name=image.png&originHeight=349&originWidth=633&originalType=binary&ratio=1&rotation=0&showTitle=false&size=158729&status=done&style=none&taskId=ua14db341-3de4-439b-915b-46786a02031&title=&width=405.12)

原始高维空间的样本点在这个低维嵌入子空间中更容易进行学习。

---

<a name="Bppv3"></a>
## 多维缩放 MDS

若要求原始空间中样本之间的距离在低维空间中得以保持，即得到**“多维缩放”** (Multiple Dimensional Scaling, 简称 MDS) 这样的一种经典降维方法。

设有$m$个样本在原始空间的距离矩阵$D\in \mathbb{R}^{m \times m}$，目标获得样本在$d'$维空间的表示$Z\in \mathbb{R}^{d'\times m}$，且任意两个样本在$d'$维空间中的欧氏距离等于原始空间中的距离，即$\|z_i-z_j\|=dist_{ij}$

令$B=Z^{\mathrm{T}}Z \in \mathbb{R}^{m \times m}$，其中$B$为降维后样本的内积矩阵，$b_{ij}={z_{i}}^{\mathrm{T}}z_j$，有<br />$\operatorname{dist}_{i j}^{2} =\left\|\boldsymbol{z}_{i}\right\|^{2}+\left\|\boldsymbol{z}_{j}\right\|^{2}-2 \boldsymbol{z}_{i}^{\mathrm{T}} \boldsymbol{z}_{j} =b_{i i}+b_{j j}-2 b_{i j}$ 

令降维后的样本$Z$被中心化，可以推导出：$b_{i j}=-\frac{1}{2}\left(d i s t_{i j}^{2}-d i s t_{i .}^{2}-d i s t_{. j}^{2}+d i s t_{. .}^{2}\right)$（10.10）

对$B$做**“特征值分解”(eigenvalue decomposition)，**$B=V\Lambda V^{\mathrm{T}}$**,**其中$\mathbf{\Lambda}=\operatorname{diag}\left(\lambda_{1}, \lambda_{2}, \ldots, \lambda_{d}\right)$为特征值构成的对角矩阵，$\lambda_{1} \geqslant \lambda_{2} \geqslant \ldots \geqslant \lambda_{d}$，$V$为特征向量矩阵。假定有$d^*$个非零特征值，其对应的对角阵为$\Lambda _*$，特征向量矩阵为$V_*$，则$Z$可以表达为：

$\mathbf{Z}=\mathbf{\Lambda}_{*}^{1 / 2} \mathbf{V}_{*}^{\mathrm{T}} \in \mathbb{R}^{d^{*} \times m}$

现实中为有效降维可取$d'\ll d$个最大特征值，则$\mathbf{Z}=\tilde{\mathbf{\Lambda}}^{1 / 2} \tilde{\mathbf{V}}^{\mathrm{T}} \in \mathbb{R}^{d^{\prime} \times m}$


**MDS 算法描述：**<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655988175867-47575d80-8426-48df-b3f2-75c38c548a74.png#clientId=u76c62888-11e0-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=155&id=ud628e8e2&name=image.png&originHeight=242&originWidth=757&originalType=binary&ratio=1&rotation=0&showTitle=false&size=111630&status=done&style=none&taskId=uc7585ca3-e6f2-472d-9e5f-a9c93058acc&title=&width=484.48)


通常来说获得低维子空间最简单的是对原始高维空间进行线性变换，即$Z=W^{\mathrm{T}}X$，其中$Z\in \mathbb{R}^{d'\times m}$是样本在新空间中的表达，$W^{\mathrm T} \in \mathbb{R}^{d \times d'}$是变换矩阵. 基于线性变换来进行的降维方法称为线性降维方法，都符合该基本形式，只不过不同方法对$W$的约束不同。

新空间中的属性是原空间中属性的线性组合。

---

<a name="hSIA5"></a>
## 主成分分析 PCA

<a name="FKBwI"></a>
### PCA 概述

主成分分析是最常用的一种降维方法。它利用正交变换把由线性相关变量表示的观测数据转换为少数几个由线性无关变量表示的数据，线性无关的变量称为**主成分**。

:::info
对原坐标系中的数据进行主成分分析等价于进行坐标系旋转变换，将数据投影到新坐标系的坐标轴； 新坐标系的第一坐标轴、第二坐标轴等分别表示第一主成分、 第二主成分等，数据在每个轴上的坐标值的平方表示相应变的方差；井且，**这个坐标系是在所有可能的新的坐标系中，坐标轴上的方差的和最大的。**也就是到轴距离平方和最小。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655992340614-887d111a-1e48-4e32-9ef3-b411615a517c.png#clientId=u76c62888-11e0-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=179&id=uade44836&name=image.png&originHeight=349&originWidth=524&originalType=binary&ratio=1&rotation=0&showTitle=false&size=41105&status=done&style=none&taskId=u2095efe0-ae98-439c-b1ac-c20f00d5a33&title=&width=268.3599853515625)
:::
<br />数学表示：<br />假定样本已进行了中心化，投影变换得到的坐标系为$\{w_1,w_2,...,w_d\}$，其中$w_i$为标准正交基向量。根据最近重构性可得优化目标：<br />$\begin{aligned}
&\min _{\mathbf{W}}-\operatorname{tr}\left(\mathbf{W}^{\mathrm{T}} \mathbf{X} \mathbf{X}^{\mathrm{T}} \mathbf{W}\right) \\
&\text { s.t. } \mathbf{W}^{\mathrm{T}} \mathbf{W}=\mathbf{I}
\end{aligned}$

对优化目标使用拉格朗日乘子法可得$XX^{\mathrm{T}}w_i=\lambda_i w_i$，对协方差矩阵$XX^{\mathrm{T}}$进行特征值分解，将特征向量由大到小排序，取前$d'$个特征值对应的特征向量构成矩阵$\mathrm{W}^*$，这就是主成分分析的解。

**算法描述：**<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1655993349617-ed15f779-ebdd-4de0-aaa5-a303cb3b88d2.png#clientId=u76c62888-11e0-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=174&id=u3cac3ad4&name=image.png&originHeight=320&originWidth=840&originalType=binary&ratio=1&rotation=0&showTitle=false&size=155718&status=done&style=none&taskId=u968856f6-113b-4a47-a249-93ce2114865&title=&width=456.5899963378906)

可通过在$d'$值不同的低维空间中对$k$近邻分类器进行交叉验证来选取较好的$d'$值，也可以设置一个重构阈值$t$ ，并选取使得大于等于$t$累计方差贡献率$\sum_{i=1}^{d'} \eta_{i}=\frac{\sum_{i=1}^{d'} \lambda_{i}}{\sum_{i=1}^{d} \lambda_{i}}$最小的$d'$值。

---

<a name="JnvQH"></a>
### 总体主成分分析

在数据总体(population) 上进行主成分分析称为总体主成分分析。

假设$\boldsymbol x=(x_1,x_2,...,x_m)^{\mathrm{T}}$是$m$维随机变量，则其均值向量$\boldsymbol{\mu}=E(\boldsymbol{x})=(\mu_1,\mu_2,...,\mu_m)^{\mathrm{T}}$，协方差矩阵为$\Sigma=\operatorname{cov}(\boldsymbol{x}, \boldsymbol{x})=E\left[(\boldsymbol{x}-\boldsymbol{\mu})(\boldsymbol{x}-\boldsymbol{\mu})^{\mathrm{T}}\right]$，

考虑线性变换$y_{i}=\alpha_{i}^{\mathrm{T}} \boldsymbol{x}=\alpha_{1 i} x_{1}+\alpha_{2 i} x_{2}+\cdots+\alpha_{m i} x_{m}$，则有：<br />$\begin{aligned}
&E\left(y_{i}\right)=\alpha_{i}^{\mathrm{T}} \mu, \quad i=1,2, \cdots, m \\
&\operatorname{var}\left(y_{i}\right)=\alpha_{i}^{\mathrm{T}} \Sigma \alpha_{i}, \quad i=1,2, \cdots, m \\
&\operatorname{cov}\left(y_{i}, y_{j}\right)=\alpha_{i}^{\mathrm{T}} \Sigma \alpha_{j}, \quad i=1,2, \cdots, m ; \quad j=1,2, \cdots, m
\end{aligned}$

**总体主成分定义：**
:::info
给定一个线性变换$y_{i}=\alpha_{i}^{\mathrm{T}} \boldsymbol{x}=\alpha_{1 i} x_{1}+\alpha_{2 i} x_{2}+\cdots+\alpha_{m i} x_{m}$，满足以下条件：

1. 系数向量$\alpha_i^T$是单位向量，即$\alpha_{i}^{\mathrm{T}} \alpha_{i}=1, i=1,2, \cdots, m$，线性变换是正交变换，$\alpha$是一组标准正交基。
1. 变量$y_i$与$y_j$线性无关，即$cov(y_i,y_j)=0(i≠j)$
1. $y_i$是与$y_1,y_2,...,y_i-1$都不相关的$x$的所有线性变换中方差最大，此时称$y_i$为$x$的第$i$主成分
:::


**主要性质：**
:::info
**定理：**设$x$是$m$维随机变量，$\Sigma$是$x$的协方差矩阵，其特征值分别为$\lambda_{1} \geqslant \lambda_{2} \geqslant \ldots \geqslant \lambda_{m}\geqslant0$，特征值对应的单位特征向量分别为$\alpha_1,\alpha_2,...,\alpha_m$则<br />$x$的第$k$个主成分为：<br />$y_{k}=\alpha_{k}^{\mathrm{T}} \boldsymbol{x}=\alpha_{1 k} x_{1}+\alpha_{2 k} x_{2}+\cdots+\alpha_{m k} x_{m}, \quad k=1,2, \cdots, m$

$x$的第$k$个主成分的方差是：<br />$\mathrm{var}(y_k)=\alpha^{\mathrm{T}}\Sigma \alpha_k=\lambda_k,\quad k=1,2,...,m$
:::

1. 总体主成分$y$的协方差矩阵是对角矩阵：$\Lambda=\mathrm{diag(\lambda_1,\lambda_2,...,\lambda_m)}$
1. 总体主成分$y$的方差之和等于随机变量$x$的方差之和：$\sum_{i=1}^{m}\lambda_i= \sum_{i=1}^{m}\sigma_{ii}$，$\sigma_{ii}$是随机变量$x_i$的方差，即协方差矩阵$\Sigma$的对角元素。
1. 第$k$个主成分$y_k$与变量$x_i$的相关系数$\rho(y_k,x_i)$称为因子负荷量 (factor loading) ，它表示第$k$个主成分$y_k$与变量$x_i$的相关关系。计算公式为：

$\rho\left(y_{k}, x_{i}\right)=\frac{\sqrt{\lambda_{k}} \alpha_{i k}}{\sqrt{\sigma_{i i}}}, \quad k, i=1,2, \cdots, m$

4. 第$k$个主成分$y_k$与$m$个变量的因子负荷量满足：

$\sum_{i=1}^{m} \sigma_{i i} \rho^{2}\left(y_{k}, x_{i}\right)=\lambda_{k}$ 

5. $m$个主成分与第$i$个变量$x_i$的因子负荷量满足：

$\sum_{i=1}^{m}\rho^{2}\left(y_{k}, x_{i}\right)=1$

**规范化变量的总体主成分**<br />不同变量可能有不同的量纲，直接求主成分会产生不合理的结果，所有需要对各随机变量进行规范化，使其均值为0，方差为1，即令<br />$x_{i}^{*}=\frac{x_{i}-E\left(x_{i}\right)}{\sqrt{\operatorname{var}\left(x_{i}\right)}}, \quad i=1,2, \cdots, m$<br />规范化随机变量的协方差矩阵就是相关矩阵$R$，主成分分析通常在$R$上进行。

---

<a name="bcotR"></a>
### 样本主成分分析

实际问题中，需要在观测数据上进行主成分分析，即样本主成分分析。

**主要参数：**
:::info
**样本均值向量：**$\bar{x}=\frac{1}{n}\sum_{j=1}^{n}x_j$<br />**样本协方差矩阵：**$S=[s_{ij}]_{mn},\quad s_{ij}=\frac{1}{n-1}\sum_{k=1}^{n}(x_{ik}-\bar{x_i})(x_{jk}-\bar{x_j})\quad i,j=1,2,...,m$**，**即$S=\frac{1}{n-1}\sum_{}$<br />**样本相关矩阵：**$R=\left[r_{i j}\right]_{m \times m}, \quad r_{i j}=\frac{s_{i j}}{\sqrt{s_{i i} s_{j j}}}, \quad i, j=1,2, \cdots, m$<br />**线性变换**$y_i$**的样本方差：**$\operatorname{var}\left(y_{i}\right)=\frac{1}{n-1} \sum_{j=1}^{n}\left(a_{i}^{\mathrm{T}} \boldsymbol{x}_{j}-a_{i}^{\mathrm{T}} \overline{\boldsymbol{x}}\right)^{2}={a_i}^{\mathrm{T}}Sa_i$<br />$y_i,y_k$**的样本协方差：**$(y_i,y_k)={a_i}^{\mathrm{{T}}}Sa_k$
:::

样本主成分与总体主成分具有同样的性质，只要以样本协方差矩阵$S$替代总体协方差矩阵$\Sigma$即可。使用样本时一般将样本数据规范化，变化如下：<br />$x_{i j}^{*}=\frac{x_{i j}-\bar{x}_{i}}{\sqrt{s_{i i}}}, \quad i=1,2, \cdots, m ; \quad j=1,2, \cdots, n$<br />此时样本协方差矩阵$S$就是样本相关矩阵$R=\frac{1}{n-1}XX^{\mathrm T}$

---

<a name="YYFnB"></a>
### 相关矩阵的算法

**特征值分解算法**
:::info

1. 对$m\times n$样本矩阵$X$进行规范化，然后求样本相关矩阵$R=\frac{1}{n-1}XX^{\mathrm T}$
1. 再求样本相关矩阵$R$的$k$个特征值和对应的单位特征向量，构造正交矩阵$V=(v_1,v_,...,v_k)$
1. $V$的每一列对应一个主成分，得到$k\times n$样本主成分矩阵$Y=V^{\mathrm{T}}X$
:::

**奇异值分解算法：**
:::info

1. 对$m\times n$样本矩阵$X$规范化，然后定义一个新矩阵$X^{\prime}=\frac{1}{\sqrt{n-1}} X^{\mathrm{T}}$
1. 对矩阵$X'$进行截断奇异值分解，保留$k$个奇异值、奇异向量，得到$X'=U\Sigma V^{\mathrm{T}}$
1. $V$每一列对应一个主成分，得到$k \times n$样本主成分矩阵$Y=V^{\mathrm{T}}X$
:::

---

<a name="akVry"></a>
## 本章相关代码

[sklearn.decomposition.PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html?highlight=pca#sklearn.decomposition.PCA)
```python
import numpy as np

X_centered = X - X.mean(axis=0) # 中心化
U, sigma, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0] # 获取第一个主成分单位向量
```
```python
import numpy as np

X_centered = X - X.mean(axis=0) # 中心化
U, sigma, Vt = np.linalg.svd(X_centered) # 获得特征向量矩阵VT

W2 = Vt.T[:, :2] # 降维到2维，取前2个特征向量
X2D = X_centered.dot(W2)
```
```python
from sklearn.decomposition import PCA

# scikit-learn的PCA使用SVD降维
# svd_solver设置为"randomized"时为随机SVD
pca = PCA(n_components=2, svd_solver="randomized") # 保留2个主成分
# pca = PCA(n_components=0.9, svd_solver="randomized") # 保留90%的信息
X2D = pca.fit_transform(X)

X_recovered = pca.inverse_transform(X2D) # 解压缩回原维度

```
```python
pca = PCA() # 不降低维度情况下执行PCA
pca.fit(X_train
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1 #计算保留95%训练集方差所需的最小维度
```

---

<a name="DEBbY"></a>
# 异常检测(Anomaly Detection，也称离群值检测)

<a name="pifPK"></a>
## 问题的动机

:::info
通常数据集汇总的异常数据被认为是异常点、离群点或孤立点，特点是这些数据的特征与大多数数据不一致，呈现出"异常"的特点，检测这些数据的方法称为**“异常检测”(Anomaly detection)**。
:::

例：假设你是一个飞机引擎制造商，当你生产的飞机引擎从生产线上流出时，你需要进行**QA**(质量控制测试)，而作为这个测试的一部分，你测量了飞机引擎的一些特征变量，比如引擎运转时产生的热量，或者引擎的振动等等。

![93d6dfe7e5cb8a46923c178171889747.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656147571449-18254f23-6ea0-483b-82bc-c29d1c7362e2.png#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=88&id=u61363f9f&name=93d6dfe7e5cb8a46923c178171889747.png&originHeight=138&originWidth=600&originalType=binary&ratio=1&rotation=0&showTitle=false&size=40551&status=done&style=none&taskId=ude57f8cf-1f6b-4b57-9a7f-973caf805a4&title=&width=384)

这样一来，你就有了一个数据集，从$x^{(1)}$到$x^{(m)}$，如果你生产了$m$个引擎的话，你将这些数据绘制成图表，看起来就是这个样子：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656147712368-f1fb7577-a9e3-40f3-907f-5575ab66b334.png#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=187&id=udbfbbeb3&name=image.png&originHeight=351&originWidth=459&originalType=binary&ratio=1&rotation=0&showTitle=false&size=20449&status=done&style=none&taskId=udf9ca40c-e99b-416a-9788-7dfd400b215&title=&width=244.75997924804688)

这里的每个点都是你的标签数据。所谓的异常检测问题就是：当生产一个新飞机引擎时，我们希望测试这个新的飞机引擎是否有某种异常，

给定数据集 $x^{(1)},x^{(2)},..,x^{(m)}$，我们假使数据集全部是正常数据，我们希望知道新数据 $x_{test}$ 是不是异常的，即这个测试数据不属于该组数据的概率（密度）多大。

我们所构建的概率密度模型$p(x)$，对于任意一个输入数据$x_i$模型应该能根据该测试数据的位置告诉我们其属于一组数据的可能性。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656147816813-f579db9b-5fde-44ac-b0dc-8c0b3fccf5ef.png#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=239&id=u6a571fc3&name=image.png&originHeight=520&originWidth=1014&originalType=binary&ratio=1&rotation=0&showTitle=false&size=79160&status=done&style=none&taskId=u1483c020-c334-4df7-95ec-bd0972f238d&title=&width=466.32000732421875)

上图中，越是偏远的数据，其属于该组数据的可能性就越低。我们可以其出现的概率与某一个概率值$ϵ$进行对比，假如大于这个概率，我们可以认为其是正常的，反之则是不正常的，这种方法称为**“密度估计”**。

表达式如下：<br />$\begin{cases}p(x)<\varepsilon & \text { anomaly } \\ p(x) \geq \varepsilon & \text { normal }\end{cases}$

---

<a name="eaM8y"></a>
## 应用高斯分布的异常检测算法

在机器学习中样本方差通常只除以$m$而非统计学中的$(m−1)$，因为在实际使用中数据集通常都比较大，那么两者区别并不大。


**异常检测算法：**
:::info
对于给定的数据集 $x^{(1)},x^{(2)},...,x^{(m)}$，我们要进行参数估计

$\hat\mu_j=\frac{1}{m}\sum\limits_{i=1}^{m}x_j^{(i)}$，$\hat\sigma_j^2=\frac{1}{m}\sum\limits_{i=1}^m(x_j^{(i)}-\mu_j)^2$

获得期望和方差的估计值后，给定新的一个训练实例，根据模型计算 $p(x)$：

$p(x)=\prod\limits_{j=1}^np(x_j;\hat\mu_j,\hat\sigma_j^2)=\prod\limits_{j=1}^n\frac{1}{\sqrt{2\pi}\hat\sigma_j}exp(-\frac{(x_j-\hat\mu_j)^2}{2\hat\sigma_j^2})$

当$p(x) < \varepsilon$时，为异常。
:::

下图是一个由两个特征的训练集，以及根据数据估计的特征的分布情况：

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656149161292-630bd923-e233-49c0-98b6-b0772827e22d.jpeg#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=187&id=u8b952683&originHeight=688&originWidth=864&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u293716d2-1167-49b5-899d-03ee86cca5a&title=&width=234.3199920654297)![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656149179435-2f820224-ff47-4910-9bd4-2b0b402858cf.jpeg#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=179&id=u34d8910f&originHeight=902&originWidth=1602&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u65480ca6-d01c-449b-a6c4-3b9bbad4f12&title=&width=317.32000732421875)

下面的三维图表表示的是概率密度估计函数，$z$轴为根据两个特征的值所估计$p(x)$值：

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656148912818-563d1219-4112-436d-9c66-d3e50f8cc817.jpeg#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=247&id=u1f26c699&originHeight=514&originWidth=694&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u6fbe1cea-3e32-43d9-8f14-a717d5fec4a&title=&width=333.3199768066406)

我们选择一个$\varepsilon$，将$p(x) = \varepsilon$作为我们的判定边界，当$p(x) > \varepsilon$时预测数据为正常数据，否则为异常。

本例中，我们令$\varepsilon=0.02$$\varepsilon=0.02$，计算$p(x^{1}_{test})=0.0426≥\varepsilon$$p(x_{test}^1)=0.0426≥\varepsilon$，因此$x^1_{test}$点为正常点；而$p(x^2_{test})=0.0021<\varepsilon$，$p(x_{text}^2)=0.0021<\varepsilon$，因此$x^2_{test}$点为异常点。


**异常检测的评价方法：**<br />对于一组带标签的数据集我们可以分别对不同$\varepsilon$采用交叉验证的方式训练模型，同样可以采用**查全率、查准率、F1度量**作为评价指标，最后选择出性能最好的$\varepsilon$

---

<a name="OUcOf"></a>
## 异常检测与监督学习对比

之前我们构建的异常检测系统也使用了带标记的数据，与监督学习有些相似，下面的对比有助于选择采用监督学习还是异常检测：

两者比较：

| **异常检测** | **监督学习** |
| --- | --- |
| 非常少量的正向类（异常数据 $y=1$）, 大量的负向类（$y=0$） | 同时有大量的正向类和负向类 |
| 许多不同种类的异常，非常难。根据非常 少量的正向类数据来训练算法。 | 有足够多的正向类实例，足够用于训练 算法，未来遇到的正向类实例可能与训练集中的非常近似。 |
| 未来遇到的异常可能与已掌握的异常、非常的不同。 |  |
| 例如： 欺诈行为检测 生产（例如飞机引擎）检测数据中心的计算机运行状况 | 例如：邮件过滤器 天气预报 肿瘤分类 |


对于很多技术公司可能会遇到的一些问题，通常来说，正样本的数量很少，甚至有时候是0，也就是说，出现了太多没见过的不同的异常类型，那么对于这些问题，通常应该使用的算法就是异常检测算法。

---

<a name="KMsOx"></a>
## 特征选择与优化

<a name="JdnJ8"></a>
### 特征选择

对于异常检测算法，我们使用的特征是至关重要的，下面谈谈如何选择特征：

异常检测特征通常符合高斯分布，如果数据的分布不是高斯分布，异常检测算法也能够工作，但是最好还是将数据转换成高斯分布。

例如使用对数函数：$x= log(x+c)$，其中 $c$ 为非负常数； 或者 $x=x^c$，$c$为 0-1 之间的一个分数，等方法。（注：在**python**中，通常用`np.log1p()`函数，$log1p$就是 $log(x+1)$，可以避免出现负数结果，反向函数就是`np.expm1()`)

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656153709315-bd608e83-31d4-44c7-a313-6099b034e25b.jpeg#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=146&id=ucbaac96d&originHeight=610&originWidth=1912&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u242a4af7-dda2-4bb9-a557-db45483aff6&title=&width=456.32000732421875)

**误差分析：**<br />一个常见的问题是一些异常的数据可能也会有较高的$p(x)$值，因而被算法认为是正常的。<br />我们可以分析那些被算法错误预测为正常的数据，观察能否找出一些问题。我们可能能从问题中发现我们需要**增加一些新的特征**，增加这些新特征后获得的新算法能够帮助我们更好地进行异常检测。

**例：**下图左图所示绿色异常点，在$x_1$维度其概率密度$p(x)$依然较大，就会被异常检测算法错误预测为正常数据。这时我们可以分析可否增加特征值，如右图所示新增了特征值$x_2$，此时就会发现在$x_2$这个维度上，绿色异常点的概率密度$p(x)$非常小，那么异常检测算法就能检测到此异常数据了。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656153670762-39b65f6e-f467-4315-9ef8-9346585f5f3d.png#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=0.9893&from=paste&height=279&id=u25c13c5b&name=image.png&originHeight=642&originWidth=1097&originalType=binary&ratio=1&rotation=0&showTitle=false&size=113075&status=done&style=none&taskId=uf0744006-0fee-4ace-922c-eb2717316fc&title=&width=477)


<a name="RCZ93"></a>
### 特征优化

例如我们可以**将一些相关的特征进行组合**，来获得一些新的更好的特征（异常数据的该特征值异常地大或小）

**例：**以计算机异常检测为例，有以下特征值：

- $x1$ = memory use of computer
- $x2$ = number of disk accesses/sec
- $x3$ = CPU load
- $x4$ = network traffic

假如我们的计算机是作为Web服务器使用，那么正常情况下CPU load和network traffic应该是成正比的，这种时候我们就可以定义一个新的特征值$x5=\frac{CPU load}{network traffic}$，假如突然出现了$x5$非常大的情况，那么我们基本上就可以判定计算机CPU出现异常。


1. **特征的简单变换**

**数值特征的变换和组合**
:::info
特征的线性组合( linear combination) 仅适用于决策树及基于决策树的集成学习算法，如Gradient Boosting，随机森林。因为树模型不擅长捕获不同特征之间的相关性。而SVM、线性回归、神经网络等模型自身可以线性组合。常用以下组合：

**多项式特征（ polynomial feature)**<br />**比例特征（ratio feature)：**$\frac{X_1}{X_2}$<br />**绝对值（absolute value)。**<br />$\boldsymbol{\max}(X_1,X_2$**，**$\boldsymbol{\min}(X1,X2)$**，Xl or X2**
:::

**类别特征与数值特征的组合**
:::info
用$N1$和$N2$表示数值特征，用和$C2$表示类别特征，利用** Pandas **的 **groupby **操作可以创造出以下几种有意义的新特征（其中，$C2$还可以是离散化了的$N1$)。

**中位数：**median (N1 ) _by (C1)。<br />**算术平均数：**mean(N1)_by (C1)<br />**众数：**mode (N1 )_by (C1)。<br />**最小值：**min(N1) _by (C1)。<br />**最大值：**max (N1 ）_by (C1)。<br />**标准差：**stdN1） by (C1)。<br />**方差：**var （N1)_by (C1)。<br />**频数：**freq (C2)_ (C1)。

仅仅将已有的类别特征和数值特征进行以上的有效组合，就能增加大量优秀的可用特征。
:::


2. **用决策树创造新特征**
:::info
在决策树系列（单棵决策树、GBDT、随机森林）的算法中，由于每一个样本都会被映射到决策树的一片叶子上，因此我们可以把样本经过每一棵决策树映射后的$index$（自然数）或$one-hot-vector$（哑编码得到的稀疏矢量）作为一项新的特征加入模型中。

具体实现可以采用 **apply() **方法和 **decision_path() **方法，其在 sklearn和 xgboost中都可以用。
:::


3. **特征组合**



**对非线性规律进行编码**
:::info
对非线性问题，可以将特征进行组合得到新特征，然后按照线性模型进行训练。对新的特征组合可以像处理任何其他特征一样来处理。

我们可以创建很多种组合：<br />$[A×B]$：**将两个特征的值相乘形成的特征组合。**<br />$[A× B×C×D×E]$：**将五个特征的值相乘形成的特征组合**<br />$[A×A]$：**对单个特征的值求平方形成的特征组合。**
:::

**组合独热矢量**
:::info
对于文本属性，我们需要将类别转到数字。给每个类别创建一个二进制的属性：假设有两个类别分别为**"C1"**, **"C2"**，当类别是 **"C1" **时，其编码则为** [1, 0]**，当类别 是 **"C2" **时，编码为 **[0, 1]**。这就是独热编码  。

 如果类别属性具有大量可能的类别（例如，国家代码、专业、物种），那么独热编码会导致大量的输入特征，这可能会减慢训练并降低性能。如果发生这种情况，你可 能想要用相关的数字特征代替类别输入。例如，你可以用与海洋的距离来替换 "ocean_proximity" 特征（类似地，可以用该国家的人口和人均GDP来代替国家代码）。或者，你可以用可学习的低维向量（称为嵌入）来替换每个类别。每个类别的表征可以在训练期间学习。  
:::


**使用分桶特征列**
:::info
分桶特征：是以一定方式将连续型数值特征划分到不同的桶（箱）中，可以理解为是对连续型特征的一种离散化处理方式。

例如，我们可以将某地的人口（population）特征分为以下3个分桶：

- bucket_0 (<5000):对应人口分布较少的街区。
- bucket_1 (5000~25000):对应人口分布适中的街区。
- bucket_2 (> 25000):对应人口分布较多的街区。

根据前面的分桶定义，population矢量[[10001]，[ 42004]，[2500]，[18000]] 将变成以下经过分桶的特征矢量[[1], [2], [0], [1]] 。这些特征值会变换为分桶索引，而这些索引被视为离散特征。在通常情况下，这些特征将被进一步转换为独热编码。

:::

---

<a name="C03gu"></a>
## 多元高斯分布

假使我们有两个相关的特征，而且这两个特征的值域范围比较宽，这种情况下，一般的高斯分布模型可能不能很好地识别异常数据。其原因在于，一般的高斯分布模型尝试的是去同时抓住两个特征的偏差，因此创造出一个比较大的判定边界。

下图中是两个相关特征，如果我们分别对$x_1$,$x_2$求解高斯分布并计算绿色异常点的$p(x_1)$,$p(x_1)$，那么结果就是两个概率密度都较大，此时异常检测会把绿色异常点预测为正常。分别求高斯分布的正常范围如下图红色线所示。而采用多元高斯分布将创建像图中蓝色曲线所示的判定边界，此时可以得到正确的预测。z

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656157046749-05b29f9d-d64e-483d-88f4-7c5e68cea33e.png#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=212&id=u4cf7a6ba&name=image.png&originHeight=471&originWidth=546&originalType=binary&ratio=1&rotation=0&showTitle=false&size=108792&status=done&style=none&taskId=u17d53ae0-7903-47d8-b469-72e5418de61&title=&width=245.44000244140625)


多元高斯分布模型中，我们将构建特征的协方差矩阵，用所有的特征一起来计算 $p(x)$。

我们首先计算所有特征的样本期望，然后再计算样本协方差矩阵：

$\mu=\frac{1}{m}\sum_{i=1}^mx^{(i)},\quad\mu \in \mathbb{R}^n$

$\Sigma = \frac{1}{m}\sum_{i=1}^m(x^{(i)}-\mu)(x^{(i)}-\mu)^T=\frac{1}{m}(X-\mu)^T(X-\mu),\quad\Sigma \in \mathbb{R}^{n \times n}$

其中$\mu$ 是一个向量，其每一个值都是原特征矩阵中对应行数据的均值。最后我们计算多元高斯分布的$p\left( x \right)$

$p(x)=\frac{1}{(2\pi)^{\frac{n}{2}}|\Sigma|^{\frac{1}{2}}}exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right)$

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656157168372-3d249502-6edf-4b0b-b022-89e6b2fc5c6e.png#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=292&id=u83923bd0&name=image.png&originHeight=558&originWidth=1008&originalType=binary&ratio=1&rotation=0&showTitle=false&size=403202&status=done&style=none&taskId=ub6f7ab9a-e234-46c1-94c0-f83438bddd1&title=&width=528.1199951171875)<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656157790235-f8afcb9d-22a4-4c71-9c39-8adfe47cd699.png#clientId=ue4861eed-7fc8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=291&id=uff809bcf&name=image.png&originHeight=559&originWidth=999&originalType=binary&ratio=1&rotation=0&showTitle=false&size=395406&status=done&style=none&taskId=u3226c661-af26-428b-8b66-d43d3318232&title=&width=519.3599853515625)



原高斯分布模型和多元高斯分布模型的比较：

| **原高斯分布模型** | **多元高斯分布模型** |
| --- | --- |
| 不能捕捉特征之间的相关性 但可以通过将特征进行组合的方法来解决 | 自动捕捉特征之间的相关性 |
| 计算代价低，能适应大规模的特征 | 计算代价较高 训练集较小时也同样适用 |
| $m$较小时也能顺利运行 | 必须要有 $m>n$，不然的话协方差矩阵$\Sigma$不可逆，通常需要 $m>10n$ ；另外特征冗余也会导致协方差矩阵不可逆 |


原始高斯分布模型使用更为广泛，如果特征之间在某种程度上存在相关性，我们可以通过构造新新特征的方法来捕捉这些相关性。如果训练集不是太大，并且没有太多的特征，我们可以使用多元高斯分布模型。


---

<a name="x08bc"></a>
## 本章相关代码

[sklearn.preprocessing.OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html?highlight=onehotencoder#sklearn.preprocessing.OneHotEncoder)
```python
# 定义特征构造方法
func_dict ={
    'add': lambda x, y : X + y,
    'mins': lambda x, y : X - y,
    'div': lambda x, y : x / (y + epsilon),
    'multi': lambda x, y : × * y
}

# 特征构造函数
def auto_features_make (train_data,test_data,func_dict,col_list):
    train_data, test_data = train_data.copy(, test_data.copy ()
    for col_i in col_ list:
        for col_j in col_list:
            for func_name, fune in func_dict.items O:
                for data in [train_data,test data] :
                    func_features = func (data [col_il,data[co1_j])-
                    col_func_features = '-' .join ([col_i, func_name, col_j1)
                    data [col_func_features]= func_features
    return train data, testdata
```
```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
data_cat_1hot = cat_encoder.fit_transform(data_cat)
data_cat_1hot # 输出为一个SciPy系数矩阵，可以使用toarray()方法转为NumPy数组

cat_encoder.categories_ # 获得类别列表
```
```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# 异常值分析
plt.figure(figsize=(18, 10))
plt.boxplot(x=train_data.values,labels=train_data.columns) # 绘制箱型图
plt.hlines([-7.5, 7.5], 0, 40, colors='r')
plt.show()

# 删除异常值V9
train_data = train_data[train_data['V9']>-7.5]
```
![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1657005980006-ce1c6d9b-04d0-45cf-a198-3687acfd4825.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=392&id=u039359a6&name=image.png&originHeight=392&originWidth=699&originalType=binary&ratio=1&rotation=0&showTitle=false&size=50714&status=done&style=none&taskId=ud3fe4c83-3a32-4a4b-9614-9163c8c0a6e&title=&width=699)

---

<a name="HUdBl"></a>
# 推荐系统(Recommender Systems)和协同过滤(Collaborative Filtering)

<a name="A6szC"></a>
## 推荐系统/协同过滤

:::info
“协同过滤是利用某兴趣相投、拥有共同经验之群体的喜好来推荐用户感兴趣的信息，个人通过合作的机制给予信息相当程度的回应（如评分）并记录下来以达到过滤的目的，进而帮助别人筛选信息，其中回应不一定局限于特别感兴趣的，特别不感兴趣信息的纪录也相当重要。”		——维基百科
:::

<a name="YEGvk"></a>
### 基于内容的推荐系统/协同过滤

在一个**基于内容的推荐系统 (Content-based recommender system)/协同过滤(ItemCF) **算法中，我们假设对于我们希望推荐的东西有一些数据，并且这些数据是有关这些东西的特征。


例：假使我们是一个电影供应商，我们有 5 部电影和 4 个用户，我们要求用户为电影打分。

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656170149149-5dd730c0-5d37-4ed3-9210-a165094a50d4.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=194&id=IRSMt&originHeight=732&originWidth=1718&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u1e837af0-f7f8-47a0-80a2-83cbf99f9af&title=&width=455)

前三部电影是爱情片，后两部则是动作片，我们可以看出**Alice**和**Bob**似乎更倾向与爱情片， 而 **Carol** 和 **Dave** 似乎更倾向与动作片。并且没有一个用户给所有的电影都打过分。我们希望构建一个算法来预测他们每个人可能会给他们没看过的电影打多少分，并以此作为推荐的依据。

我们可以假设每部电影都有两个特征，如$x_1$代表电影的浪漫程度，$x_2$ 代表电影的动作程度。

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656170535408-336c7703-5a16-49f0-bed8-cb4e74a37337.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=176&id=uea084b09&originHeight=766&originWidth=2348&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=uea0d0e58-240d-4fdc-9b0b-6a8eb48125f&title=&width=539.3099975585938)

再加上$x_0=1$则每部电影都有一个特征向量，如第一部电影的特征向量为$x^{(1)}=\begin{bmatrix}
 1\\
 0.9\\0
\end{bmatrix}$。


下面我们要基于这些特征来构建一个推荐系统算法。

假设我们采用**线性回归模型**，我们可以针对每一个用户都训练一个线性回归模型，则有

- $n_u$ 代表用户的数量
- $n_m$ 代表电影的数量
- $r(i, j)$ 如果用户$j$给电影$i$评过分则 $r(i,j)=1$
- $y^{(i, j)}$ 代表用户 $j$ 给电影$i$的评分
- $m_j$代表用户 $j$ 评过分的电影的总数
- $\theta^{(j)}$用户 $j$ 的参数向量
- $x^{(i)}$电影 $i$ 的特征向量
- 对于用户 $j$ 和电影 $i$，我们预测评分为：$(\theta^{(j)})^T x^{(i)}$

假设${{\theta }^{(1)}}$是第一个用户模型的参数，$x^{(3)}$是第三个电影的特征向量，则可以得到第一个用户对第三个电影的评分预测：$(\theta^{(1)})^{\mathrm T}x^{(3)} = 5 \times 0.99 = 4.95$

针对用户 $j$，该线性回归模型的代价函数为预测误差的平方和，加上正则化项，优化目标如下：

$\min_{\theta (j)}\frac{1}{2}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2+\frac{\lambda}{2}\sum_{k=1}^n\left(\theta_{k}^{(j)}\right)^2$

其中 $i:r(i,j)$表示我们只计算那些用户 $j$ 评过分的电影。在一般的线性回归模型中，误差项和正则项应该都是乘以$1/2m$，在这里我们将$m$去掉。并且我们不对方差项$\theta_0$进行正则化处理。

上面的代价函数只是针对单个用户的，对于所有用户的优化目标，我们将所有用户的代价函数求和：（与分别求再求和意义相同）

$\min_{\theta^{(1)},...,\theta^{(n_u)}} \frac{1}{2}\sum_{j=1}^{n_u}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2$

如果我们要用梯度下降法来求解最优解，我们计算代价函数的偏导数后得到梯度下降的更新公式为：

$\left\{\begin{matrix}
\theta_k^{(j)}:=\theta_k^{(j)}-\alpha\sum \limits_  {i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)} \quad (\text{for} \, k = 0)  &k=0 \\
  \theta_k^{(j)}:=\theta_k^{(j)}-\alpha\left(\sum \limits_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)}+\lambda\theta_k^{(j)}\right)&k \ne 0
\end{matrix}\right.$

---

<a name="iQ1xa"></a>
### 基于用户的推荐系统/协同过滤

在之前的基于内容的推荐系统中，对于每一部电影我们都掌握了可用的特征，使用这些特征训练出了每一个用户的参数。基于内容的推荐算法的局限性在于我们必须要知道了每部电影的内容（特征值），但在某些时候我们是难以获取这些特征值的。相反地，如果我们拥有用户的参数，我们可以学习得出电影的特征，即**基于用户的推荐系统 (User-based recommender system)/协同过滤(UserCF)**

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656172603588-445c4f10-bbcf-4392-b95e-389093df50c1.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=225&id=hhH8W&originHeight=954&originWidth=2218&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u957198e6-8106-4e34-974a-9ee12fb347c&title=&width=523)

假如我们没有每一部电影关于内容的特征值，但是我们却有每个用户对于每一类电影的喜爱程度的数据，即$\theta^{(i)}$

则，优化目标变为：已知$\theta^{1},\theta^{2},...,\theta^{n_u}$，学习$x^{(i)}$：

$\mathop{min}\limits_{x^{(1)},...,x^{(n_m)}}\frac{1}{2}\sum_{i=1}^{n_m}\sum_{j:{r(i,j)=1}}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(i)})^2$

同样地，如果是学习所有电影的内容特征值$x^{(1)},x^{(2)},⋯,x^{(n_m)}$，那么只需要对所有电影代价进行求和：

$\min _{x(1), \cdots, x\left(n_{m}\right)} \frac{1}{2} \sum_{j=1}^{n_{m}} \sum_{j: r(i, j)=1}\left(\left(\theta^{(j)}\right)^{T} x^{(i)}-y^{(i, j)}\right)^{2}+\frac{\lambda}{2} \sum_{i=1}^{n_{m}} \sum_{k=1}^{n}\left(x_{k}^{(j)}\right)^{2}$

---

<a name="yYCAr"></a>
### 基于两者的推荐系统/协同过滤

但是如果我们既没有用户的参数，也没有电影的特征，也可以使用协同过滤算法可以同时学习这两者。

我们的优化目标便改为同时针对$x$和$\theta$进行。

$\left\{\begin{array}{l}
\min\limits_{x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)}} J(x^{(1)},...,x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)}) \\
J=\frac{1}{2}\sum_{(i:j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2
\end{array}\right.$


对代价函数求偏导数的结果如下：<br />$\left\{\begin{matrix}
 x_k^{(i)}:=x_k^{(i)}-\alpha\left(\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\theta_k^{j}+\lambda x_k^{(i)}\right)\\
\theta_k^{(i)}:=\theta_k^{(i)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}x_k^{(i)}+\lambda \theta_k^{(j)}\right)
\end{matrix}\right.$

注：在协同过滤算法中，我们通常不使用方差项，如果需要的话，算法会自动学得。

**协同过滤算法使用步骤如下：**
:::info

1.  初始 $x^{(1)},x^{(1)},...x^{(nm)},\ \theta^{(1)},\theta^{(2)},...,\theta^{(n_u)}$为一些随机小值 
1.  使用梯度下降算法最小化代价函数，求的一些列参数
1.  在训练完算法后，我们预测$(\theta^{(j)})^Tx^{(i)}$为用户 $j$ 给电影 $i$ 的评分 
:::

通过这个学习过程获得的特征矩阵包含了有关电影的重要数据，这些数据不总是人能读懂的，但是我们可以用这些数据作为给用户推荐电影的依据。

例如，如果一位用户正在观看电影 $x^{(i)}$，可以依据两部电影的特征向量之间的距离$\left\| {{x}^{(i)}}-{{x}^{(j)}} \right\|$的大小推荐另一部电影$x^{(j)}$。

---

<a name="MzdYS"></a>
### 相似度的角度看待UserCF和ItemCF

除了前面根据回归预测，另一种协同过滤方式是通过直接通过用户/物品相似度的角度来进行推荐。

**基于用户的协同过滤：**
:::info

1. **首先获得上例中打分类似的用户喜好矩阵。**推荐系统收集完用户的行为信息后，还需要对这大量的数据信息进行数据清洗，其中最关键的两步是**减噪**和**归一化**。减噪的目的是为了过滤掉用户行为中的一些失误操作以及数据中的一些噪声，从而使得系统分析可以更加的准确。而归一化的原因在于将这些数据都限制在同一个区间内，但又不能破坏不同数据之间的相对关系，最简单的归一化操作就是将所有数据都进行适当的缩放，使得它们的取值范围为$[0,1]$。
1. **寻找与特定用户相类似的其他用户。**根据这些用户信息来计算每两个用户之间喜好的相似程度，然后对指定用户进行推荐。计算相似度实际就是计算喜好矩阵中每行间的相似度。相似度可选用欧氏距离、Jaccard系数、Pearson相关系数，余弦相似度等。
1. **根据相似度得出预测推荐。**推荐的关键步骤是要找出用户-内容的邻居，而挑选邻居的规则有两种：一是$k$-近邻原则，二是基于阈值的邻居原则，在数据稀疏的情况下基于阈值的方法效果更好。找到邻居后，可以根据相似用户的已有评价对目标用户的偏好进行预测。这里最常用的方式是**利用用户相似度和邻居评价的加权平均获得目标用户的评价预测。**

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656208426538-b7a52cbb-0d1f-4084-bfc4-3587ec65fa3a.png#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=181&id=TowbK&name=image.png&originHeight=275&originWidth=315&originalType=binary&ratio=1&rotation=0&showTitle=false&size=28522&status=done&style=none&taskId=uf805da99-023b-45b3-92c9-6f55beefc78&title=&width=207.59999084472656)![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656208442950-4922412f-1825-4201-ae2d-c90d9e3010e7.png#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=184&id=tSOlE&name=image.png&originHeight=318&originWidth=360&originalType=binary&ratio=1&rotation=0&showTitle=false&size=46067&status=done&style=none&taskId=u92fec11e-988b-44cb-b894-904ce317a72&title=&width=208.39999389648438)
:::

**UserCF示例:**<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656208545016-a5f523ac-f02c-425c-8496-d6f3db755142.png#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=324&id=Q8wuA&name=image.png&originHeight=662&originWidth=751&originalType=binary&ratio=1&rotation=0&showTitle=false&size=282020&status=done&style=none&taskId=u8b00ac57-361d-43cb-abde-88c502aa219&title=&width=367.6399841308594)


**基于内容项的协同过滤（Item-based CF）**与上述UserCF类似，它们的主要区别在于，它是通过计算内容项之间的相似性，而非计算用户间的相似性来得到指定用户的推荐列表。具体步骤为首先利用相似矩阵的列向量，即代表所有用户对内容项的喜好程度，来计算得出内容项间的相似度，然后利用指定用户的历史喜好信息，得出一个排序的相似内容项列表作为推荐预测。

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656208872520-5a0f2e9a-82c7-4d9e-8275-cd67459a2129.png#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=384&id=ub8552813&name=image.png&originHeight=695&originWidth=629&originalType=binary&ratio=1&rotation=0&showTitle=false&size=280333&status=done&style=none&taskId=u4f62deea-6fe3-4917-ad06-a9b3cf5c910&title=&width=347.55999755859375)

---

<a name="zxffo"></a>
## 向量化：低秩矩阵分解

本节主要叙述如何通过向量化的形式实现协同过滤算法，同时也讲解了如何基于学习到的特征进行产品推荐。


我们有五部电影，以及四位用户，那么 这个矩阵 $Y$ 就是一个5行4列的矩阵，它将这些电影的用户评分数据都存在矩阵里：<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656213195133-09818d49-34cf-4881-a3ed-0e353c2b05fb.png#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=180&id=u96aa81ab&name=image.png&originHeight=281&originWidth=785&originalType=binary&ratio=1&rotation=0&showTitle=false&size=28753&status=done&style=none&taskId=ue7a89b69-f514-4140-8ed6-f7b322191fc&title=&width=502.4)

$Y=\begin{bmatrix}
  5&5  &0  &0 \\
  5&?  &0  &0 \\
  ?&4  &0  &? \\
  0&0  &5  &4 \\
  0&0  &5  &0
\end{bmatrix}$

得到评分：<br />$Y=\left[\begin{array}{cccc}
\left(\theta^{(1)}\right)^{T}\left(x^{(1)}\right) & \left(\theta^{(2)}\right)^{T}\left(x^{(1)}\right) & \ldots & \left(\theta^{\left(n_{u}\right)}\right)^{T}\left(x^{(1)}\right) \\
\left(\theta^{(1)}\right)^{T}\left(x^{(2)}\right) & \left(\theta^{(2)}\right)^{T}\left(x^{(2)}\right) & \ldots & \left(\theta^{\left(n_{u}\right)}\right)^{T}\left(x^{(2)}\right) \\
\vdots & \vdots & \vdots & \vdots \\
\left(\theta^{(1)}\right)^{T}\left(x^{\left(n_{m}\right)}\right) & \left(\theta^{(2)}\right)^{T}\left(x^{\left(n_{m}\right)}\right) & \ldots & \left(\theta^{\left(n_{u}\right)}\right)^{T}\left(x^{\left(n_{m}\right)}\right)
\end{array}\right]$

我们同样可以把电影特性值用矩阵表示为$X=\left[\begin{array}{c}
\left(x^{(1)}\right)^{T} \\
\left(x^{(2)}\right)^{T} \\
\vdots \\
\left(x^{\left(n_{m}\right)}\right)^{T}
\end{array}\right]$<br />用户偏好矩阵表示为$\Theta=\left[\begin{array}{c}
\left(\theta^{(1)}\right)^{T} \\
\left(\theta^{(2)}\right)^{T} \\
\vdots \\
\left(\theta^{\left(n_{u}\right)}\right)^{T}
\end{array}\right]$

则预测得分矩阵可以表示为$X\Theta^{\mathrm{T}}$

由于预测得分的矩阵在线性代数中是一个低秩矩阵，所以协同过滤算法也称为**低秩矩阵分解（Low rank matrix factorization）**

对于每个电影$i$，都学习到一个特征向量$x^{(i)}∈\mathbb{R^n}$，那么我们如何找到和电影$i$相关的电影$j$呢？

我们只需要计算电影$i$的特征向量$x^{(i)}$和电影$j$的特征向量$x^{(j)}$之间的距离$\|x^{(i)}−x^{(j)}\|$，如果距离小，则意味着两者比较相似。那么我们就能够以此找到和电影$i$相似的电影推荐给用户即可。

---

<a name="PaVe9"></a>
## 均值归一化

如果我们新增一个用户 **Eve**，并且 **Eve** 没有为任何电影评分，那么我们以什么为依据为**Eve**推荐电影呢？

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656214114514-3158c16e-3834-4228-bbbd-05c9bccaae3a.png#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=197&id=udade41ff&name=image.png&originHeight=308&originWidth=994&originalType=binary&ratio=1&rotation=0&showTitle=false&size=36419&status=done&style=none&taskId=u23d2fb1e-77be-4de6-bae3-788f5071914&title=&width=636.16)

对于一个没有对任何电影进行评分的用户，直接采用协调过滤算法我们会得到用户偏好参数$θ=0$，那么对于每部电影的评分也会是$θ^{\mathrm{T}}x=0$，所以没办法对该用户推荐任何电影。

我们首先需要对结果 $Y$矩阵进行均值归一化处理，将每一个用户对某一部电影的评分减去所有用户对该电影评分的平均值：

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656214230464-373aaa80-a39c-46e5-8bf9-deecc1d652c8.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=117&id=u0a3a15db&originHeight=462&originWidth=2460&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=ud5c49403-a6d6-4d3e-a6fe-818d3a7b91e&title=&width=623)

然后我们利用这个新的 $Y$ 矩阵来训练算法。如果我们要用新训练出的算法来预测评分，则需要将平均值重新加回去，预测$(\theta^{(j)})^T x^{(i)}+\mu_i$，对于**Eve**，我们的新模型会认为她给每部电影的评分都是该电影的平均分。

理论上这种思想还可以应用在没有用户评分的电影上，但是实际上对于没有用户评分的电影，可能根本就不应该将其推荐给用户。比起没有评分的电影，一般更关心没有评分的用户。

---

<a name="i91Nj"></a>
## 协同过滤的应用与不足

**协同过滤不足：**
:::info

1. **冷启动问题。**冷启动问题具体可以分为两类。第一类指的是，当系统建立之初，还未收集足够的用户信息，协同过滤算法不能为指定用户找到合适的邻居，从而无法向用户提供推荐预测。第二类指的是，对于新注册的用户或者新加入的商品，由于系统里没有他们的历史数据信息，所以协同过滤算法也无法为用户预测推荐。
1. **稀疏性问题。**稀疏性问题指的是在实际情况下，用户很少会对每个内容项进行评分，所以真实的用户-内容项的相似矩阵是稀疏的（即矩阵中的很多元素都为0，表示用户对该内容未进行评分），从而降低了计算效率，而且少部分人的错误偏好会降低推荐的准确性。
1. **最初评价问题。**最初评价问题指的是，对于一些从未被评过分的内容，比如新加进的内容或者是比较小众的内容，它们是不可能会被推荐给用户的，而用户可能会对一些冷门内容也感兴趣。
1. **扩展性不足问题。**随着推荐系统的扩展，用户和内容数量的增加，计算用户或者内容项间的相似度时，计算复杂度会大大增加，从而导致系统的性能降低。
1. **流行性偏向问题。**系统会更偏向于为用户推荐比较流行的内容，因为评分覆盖面广。但对于有着独特口味的用户来说，推荐系统不能提供很好的推荐。
:::


为了克服以上的缺点，现今的推荐系统一般会采取**混合的推荐机制**来进行互补，而不是单单只采用某一种推荐策略。现在运用最广的推荐机制混合方法有以下几种：
:::info

1. ** 加权混合。**先用不同的推荐机制对用户进行推荐预测，然后再将它们的结果按照一定的权重加权求和得出最终的推荐预测，具体的权值设置需要根据实际情况决定。
1. ** 转换混合。**在不同的状态和条件下，转换选择最为合适的推荐机制对其进行预测。因为基于不同的情况，推荐机制的选择上可能会有很大的不同，为了充分利用各种推荐机制的优点，我们可以选择转换混合的方式对用户进行推荐预测。
1. **分区混合。**同时采用多种不同的推荐机制，并将产生出的不同结果分成不同的区域推荐给用户。分区混合的方法可以为用户提供更为全面的推荐结果。
1. **分层混合。**和分区混合一样，分层混合也是采用多种不同的推荐机制，但不同的是它是将一个推荐机制的结果作为下一个的输入，这样层层作用下去，最终得到一个推荐预测。分层混合的优点在于可以综合不同推荐机制的优缺点，从提高推荐准确度。
:::


什么时候使用UserCF，什么时候使用ItemCF？

|  | **UserCF** | **ItemCF** |
| --- | --- | --- |
| **性能** | 适用于用户较少的场合，用户多时计算用户相似度矩阵代价太大 | 适用于物品数明显少于用户的场合，物品多时计算用户相似度矩阵代价太大 |
| **领域** | 实时性较高，用户个性化要求不高 | 物品较为丰富，个性化需求强烈 |
| **实时性** | 对于新物品，可以实时进行预测。用户行为往往不能立即体现在推荐结果中。 | 用户行为可以实时改变推荐结果。新物品增加不会立即改变推荐结果。 |
| **冷启动** | 在新用户对少的物品产生行为后，不能立即对他进行个性化推荐，因为用户相似度是离线计算的；新物品上线后一段时间，一旦有用户对物品产生行为，就可以将新物品推荐给其他用户 | 新用户只要对一个物品产生行为，就能推荐相关物品给他，但无法在不离线更新物品相似度表的情况下将新物品推荐给用户（新的item到来也同样是冷启动问题） |


---

<a name="uGPg1"></a>
# 大规模机器学习(Large Scale Machine Learning)

<a name="BVifj"></a>
## 随机梯度下降法 (Stochastic Gradient Descent)

**批量梯度下降法(Batch Grdient Descent)**中，每次更新参数计算全部样本数据<br />$\theta_{j}:=\theta_{j}-\alpha \frac{1}{m} \sum_{i=1}^{m}\left(h_{\theta}\left(x^{(i)}\right)-y^{(i)}\right) x_{j}^{(i)}$

采用大规模数据前应该去检查一个这么大规模的训练集是否真的必要，也许我们只用较少的样本也能获得较好的效果，我们可以绘制学习曲线来帮助判断。

如果我们一定需要一个大规模的训练集，我们可以尝试使用随机梯度下降法来代替**批量梯度下降法(Batch Grdient Descent)**。

在随机梯度下降法中，我们定义代价函数为单一训练实例的代价：

$cost\left(  \theta, \left( {x}^{(i)} , {y}^{(i)} \right)  \right) = \frac{1}{2}\left( {h}_{\theta}\left({x}^{(i)}\right)-{y}^{{(i)}} \right)^{2}$

**随机梯度下降**算法为：
:::info
$while$停止准则未满足：

   - 在随机梯度下降的每次迭代中，我们对数据样本随机均匀采样⼀个索引$i$，其中$i ∈ \{1, . . . , m\}$，并计算梯度以更新参数$\theta$：

$\theta:=\sum_i{\theta}_{j}-\alpha\left( {h}_{\theta}\left({x}^{(i)}\right)-{y}^{(i)} \right){{x}_{j}}^{(i)}$
:::

随机梯度下降算法在每一次计算之后便更新参数 ${{\theta }}$ ，而不需要考虑全部样本，即每次只考虑局部最小而不是全局最小，每次迭代的复杂度从$O(m)$降到$O(1)$。此外随机梯度是对完整梯度的无偏估计。


但是这样的算法存在的问题是：算法虽然会逐渐走向全局最小值的位置，但是可能无法站到那个最小值的那一点，而是在最小值点附近徘徊。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656638686917-6ba93b06-5d9b-4e6a-8b65-db618ae74b96.png#clientId=u44911542-ece2-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=265&id=ua2ce0d60&name=image.png&originHeight=414&originWidth=421&originalType=binary&ratio=1&rotation=0&showTitle=false&size=84929&status=done&style=none&taskId=u45c4a0bd-d811-4c68-885b-6d07b86f4c8&title=&width=269.44)

---

<a name="hL9JA"></a>
## 小批量随机梯度下降(Minibatch **Stochastic** Gradient Descent)

批量梯度下降每次使用完整数据集，随机梯度下降每次使用一个训练样本。每当数据非常相似时，批量梯度下降并不是非常“数据高效”。而由于CPU和GPU无法充分利用向量化，随机梯度下降并不特别“计算高效”。所以采取一种折中办法——小批量随机梯度下降（minibatch Stochastic** **gradient descent）

算法的核心是：梯度是期望，期望可以使用小规模的样本的近似估计。每一步我们从训练集中随机抽取一小批量样本用于梯度计算。

**算法描述：**
:::info
![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656256732394-5745cec3-f31b-490f-bdc4-bd466994b85a.png#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=0.9719&from=paste&height=178&id=XIGAg&name=image.png&originHeight=278&originWidth=804&originalType=binary&ratio=1&rotation=0&showTitle=false&size=60227&status=done&style=none&taskId=u6ca682bc-b3bd-4ee4-81d6-f5a4f1231ea&title=&width=515)
:::

---

<a name="kLNrA"></a>
## 动态学习率

在批量梯度下降中，我们可以令代价函数$J$为迭代次数的函数，绘制图表，根据图表来判断梯度下降是否收敛。但是，在大规模的训练集的情况下，这是不现实的，因为计算代价太大了。

在随机梯度下降中，我们在每一次更新 ${{\theta }}$ 之前都计算一次代价，然后每$x$次迭代后，求出这$x$次对训练实例计算代价的平均值，然后绘制这些平均值与$x$次迭代的次数之间的函数图表。

当我们绘制这样的图表时，可能会得到一个噪声很多的曲线（如下图蓝线），此时如果我们减小学习率$α$，那么也许就会得到一个下降较为缓慢，但是最终代价更小的曲线（如下图红线）。

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656252401301-a1466e0f-3085-4674-ad2b-6ee438716fc2.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=151&id=u88b81fea&originHeight=524&originWidth=886&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u7995cafe-299f-47e1-9e84-57800263ac4&title=&width=255.30999755859375)

如果我们增大计算均值的代价个数，那么我们可能会得到一条更加平滑的曲线（如下图红线）。<br />![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656252455016-941e12f2-ac44-4af9-8cbe-687e8191ea93.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=156&id=u9dd6371c&originHeight=530&originWidth=890&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=uaedece5d-6706-47ff-b606-8a97ebc4cd2&title=&width=261.32000732421875)

假如我们得到的是看不出下降趋势的曲线（如下图蓝线），那么我们可以试着增大计算均值的代价个数，比如把1000个变成5000个，那么我们可能会得到一条更为平滑的，能够看到缓慢下降趋势的曲线（如下图红线），那么至少说明我们的算法是逐渐收敛的；但是，我们也有可能得到一条更为平滑的，但是依然不能够看到任何下降趋势的曲线（如下图品红线），那么说明我们的算法可能是存在问题的。<br />![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656252542102-0507f332-733b-474f-a51b-0f5f9416ba8b.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=155&id=ubec7bc87&originHeight=532&originWidth=866&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=ud1947a9c-b647-4abb-b972-75fefd754c0&title=&width=252.32000732421875)

如果我们得到一个不断上升的曲线（如下图蓝线），那么我们可能需要选择一个较小的学习率$α$。<br />![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656252506601-c74830e8-ca0e-4d18-a7ca-d945493ef480.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=161&id=u2d7e2b40&originHeight=530&originWidth=902&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=ub8bb780a-cbc8-46b0-9cfb-2bb16726956&title=&width=273.32000732421875)

由于梯度的随机性质，即使我们接近最小值，我们仍然受到通过$\alpha\left( {h}_{\theta}\left({x}^{(i)}\right)-{y}^{(i)} \right){{x}_{j}}^{(i)}$的瞬间梯度所注入的不确定性的影响。

也就是说**SGD**中梯度估计引入的噪声源（样本的随机采样）并不会在极小点处消失，所以很难收敛。而批量梯度下降中代价函数的真实梯度会变得很小，最终为0，因此批量梯度可以使用固定的学习率。

保证**SGD**收敛的一个充分条件是：<br />$\sum_{k=1}^{\infin}\epsilon_k=\infin，且\sum_{k=1}^{\infin}\epsilon_k^2<\infin$

我们唯一的选择是：**改变学习率**$\alpha$。但是，如果我们选择的学习率太小，我们⼀开始就不会取得任何有意义的进展。另一方面，如果我们选择的学习率太大， 我们将无法获得⼀个无法近似收敛的方案。解决这些相互冲突的目标的唯一方法就是在优化过程中动态降低学习率。

可以使用与时间相关的学习率$\alpha(t)$取代$\alpha$，以下是一些调整方法：
:::info
$\begin{array}{ll}
\alpha(t)=\alpha_{i} \text { if } t_{i} \leq t \leq t_{i+1} & \text { 分段常数 } \\
\alpha(t)=\alpha_{0} \cdot e^{-\lambda t} & \text { 指数衰减 } \\
\alpha(t)=\alpha_{0} \cdot(a t+1)^{-b} & \text { 多项式衰减 }
\end{array}$
:::

指数衰减会导致收敛前过早停止，$b=0.5$的多项式衰减是一种表现良好的方法。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656252091302-96cdfefb-d6b9-4d86-9ad4-a59d1adc0ee8.png#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=233&id=u4049bb54&name=image.png&originHeight=364&originWidth=507&originalType=binary&ratio=1&rotation=0&showTitle=false&size=31838&status=done&style=none&taskId=ud2effa41-0456-47c3-b31a-c1cc65ca1fa&title=&width=324.48)<br />我们也可以令学习率随着迭代次数的增加而减小，例如令：$\alpha = \frac{const1}{iterationNumber + const2}$

随着我们不断地靠近全局最小值，通过减小学习率，我们迫使算法收敛而非在最小值附近徘徊。

实践中一般会使用线性衰减学习率直到第$\tau$次迭代：$\alpha_k=(1-\epsilon)\alpha_0+\epsilon\alpha_{\tau}$，其中$\epsilon=\frac{k}{\tau}$，在第$\tau$次迭代之后使学习率$\alpha$保持常数。

---

<a name="aiCNy"></a>
## 在线学习

如果有一个由连续用户流引发的连续数据流，可以使用在线学习算法，从数据流中不断学习用户的偏好，然后使用这些信息来优化一些关于网站的决策。

在线学习算法指的是对数据流而非离线的静态数据集的学习。许多在线网站都有持续不断的用户流，对于每一个用户，网站希望能在不将数据存储到数据库中便顺利地进行算法学习。

例：假如用户访问一个运输服务网站，指定出发点和目的地之后，网站会给出一个价格，而用户有时候会选择选择该网站的运输服务$(y=1)$，有时候则不会选择$(y=0)$。特征值$x$代表出发地、目的地、价格等，我们希望通过构建一个模型，来预测用户接受报价使用我们物流服务的可能性$p(y=1|x;θ)$，以此来优化我们的报价。

在线学习的算法与随机梯度下降算法有些类似，我们对单一的实例进行学习，而非对一个提前定义的训练集进行循环。

:::info
Repeat forever (as long as the website is running){<br /> 	get$(x,y)$ corresponding to the current user<br />	$θ:=θ_j−α(hθ(x)−y)x_j\quad(\text{for}\quad j=0:n)$<br />}
:::

一旦对一个数据的学习完成了，我们便可以丢弃该数据，不需要再存储它了。这种方式的好处在于，我们的算法可以很好的适应用户的倾向性，算法可以针对用户的当前行为不断地更新模型以适应该用户。

我们所使用的这个算法与随机梯度下降算法非常类似，唯一的区别的是，我们不会使用一个固定的数据集，我们会做的是获取一个用户样本，从那个样本中学习，然后丢弃那个样本并继续下去，而且如果你对某一种应用有一个连续的数据流，这样的算法可能会非常值得考虑。

在线学习的一个优点就是，如果你有一个特征不断变化的用户群，又或者你在尝试预测的事情，在线学习算法可以根据每一次最新的用户行为慢慢地调整你所学习到的假设。

---

<a name="Y5UGb"></a>
## 映射化简和数据并行 (Map-reduce and data parallelism)

如果我们能够将较大的数据集分配给不多台计算机，让每一台计算机处理数据集的一个子集，然后我们将计所的结果汇总在求和。这样的方法叫做映射化简。

具体而言，如果一个学习算法能够表达为对训练集的函数的求和的形式，那么便能将这个任务分配给多台计算机或者同一台计算机的不同**CPU**核心，以达到加速处理的目的。

例如，我们有400个训练实例，我们可以将批量梯度下降的求和任务分配给4台计算机进行处理，然后最终再合并到一台中心服务器上面进行汇总，这样求出来的结果是一样的，但是计算速度大概提升了4倍：<br />![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656260015240-0a943677-0f49-49a4-a5ee-c26219f976b4.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&id=u8c0e9290&originHeight=1080&originWidth=2149&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u7426ce88-8d27-4031-83fe-15f71b0ec75&title=)

**多台机器**（这不就是分布式吗）<br />![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656260027566-2f43383a-bd38-49d6-bcb3-cc9cac759fb6.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=233&id=uf959ec6a&originHeight=1080&originWidth=1873&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u7ca897f0-e7d8-4512-bfd6-3017ad84a06&title=&width=403.30999755859375)

很多高级的线性代数函数库已经能够利用多核**CPU的多个核心**来并行地处理矩阵运算，这也是算法的向量化实现如此重要的缘故（比调用循环快）<br />![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656260033180-060cdd9e-3574-4d51-961f-dbbc91358270.jpeg#clientId=u35300c43-7da8-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=240&id=ufd3408b7&originHeight=1080&originWidth=1806&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u7a6e8e7d-11e9-49bc-b4b8-091a83be117&title=&width=401.32000732421875)

---

<a name="LzsmV"></a>
## 本章相关代码
```python
import numpy as np

x=2*random.rand(100,1) # 随机生成[0,2)间的数
y=4+3*x+np.random.randn100,1() # 生成一些线性数据
alpha = 0.1 # 学习率

X_b = np.c_[np.ones((100, 1)), X]  #  给每一个样本数据添加 x0 = 1
theta = np.random.randn(2,1) # 随机初始化

m = len(X_b) # 样本特征数
n_epochs = 50 # 算法迭代次数
t0, t1 = 5, 50  # learning schedule hyperparameters

# 学习率随时间（迭代次数）下降
def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2,1)  # random initialization

for epoch in range(n_epochs):
    for i in range(m): # 惯例每轮迭代训练m次
        random_index = np.random.randint(m) # 随机选取一个x
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi) # 计算梯度
        alpha = learning_schedule(epoch * m + i)
        theta = theta - alpha * gradients


plt.show()
```
```python
n_iterations = 50 # 迭代次数
minibatch_size = 20 # 小批量数

theta = np.random.randn(2,1)  # 随机初始化

# 学习率随时间（迭代次数）下降
t0, t1 = 200, 1000
def learning_schedule(t):
    return t0 / (t + t1)

t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size] # 选取i到i+minbatch_size的样本
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
```

---

<a name="fMzZq"></a>
# 
<a name="FU47K"></a>
# 应用实例：图片文字识别(Application Example: Photo OCR)

<a name="Cvg10"></a>
## 问题描述和流程图


图像文字识别应用所作的事是，从一张给定的图片中识别文字。这比从一份扫描文档中识别文字要复杂的多。

为了完成这样的工作，需要采取如下步骤：

1.  文字侦测（**Text detection**）——将图片上的文字与其他环境对象分离开来 
2.  字符切分（**Character segmentation**）——将文字分割成一个个单一的字符 
3.  字符分类（**Character classification**）——确定每一个字符是什么

可以用任务流程图来表达这个问题，每一项任务可以由一个单独的组来负责解决，也可以理解为流水线，把一个完整的机器学习问题划分为不同的模块，分别实现。<br />![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656309610074-176532ab-9998-48d7-a807-6da43278bd58.png#clientId=uf4c47708-0dd9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=54&id=uce54ea49&name=image.png&originHeight=85&originWidth=890&originalType=binary&ratio=1&rotation=0&showTitle=false&size=8111&status=done&style=none&taskId=uf1b3ceed-698c-4dd2-aced-c6306d6f6b8&title=&width=569.6)

---

<a name="zFbRV"></a>
## 滑动窗口

滑动窗口是一项用来从图像中抽取对象的技术。

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656309882859-e2c110f0-78a7-4297-8de8-3577a15fa5ae.jpeg#clientId=uf4c47708-0dd9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=248&id=u835ae0f6&originHeight=1080&originWidth=2069&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u1024f043-44ff-4378-bda9-8c2af743f7e&title=&width=476)<br />例：根据上面的数据集训练出行人检测的监督学习系统之后，我们就需要在一个完整的图像上面进行行人检测。首先，先选好一个固定大小的矩形，从左上角开始，向右进行滑动。每滑动一次，就会切割出一个矩形，然后我们就可以把这个矩形交给模型进行判断，判定该矩形是否为行人。以此类推，直到窗口滑动到图像的右下角，即可结束，这就是滑动窗口的过程。

接着，我们还需要等比例放大矩形（因为行人具有不同的长宽），然后再进行一次滑动窗口的整个流程。<br />滑动窗口中两个窗口的距离称为步长（step-size），理论上步长为1像素自然是最为准确的，但是这样计算量会太大，所以我们通常会选择一个合适的步长，比如4像素或8像素等。

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656309801964-a0124639-ca5f-4923-badf-e7bb3f691408.jpeg#clientId=uf4c47708-0dd9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=232&id=ued75ac67&originHeight=1028&originWidth=1446&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u20db5788-e9f7-4be4-9bc0-90f9abd20dc&title=&width=326)


滑动窗口技术也被用于文字识别，首先训练模型能够区分字符与非字符<br />![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656309936938-18c8aa71-5a4e-476d-aea8-365003c01f3b.jpeg#clientId=uf4c47708-0dd9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=136&id=u514b4c25&originHeight=748&originWidth=2212&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=uedd2bc18-bef7-4dd2-bb97-f9b55ea31b9&title=&width=402)

然后，运用滑动窗口技术识别字符，一旦完成了字符的识别，我们将识别得出的区域进行一些扩展，然后将重叠的区域进行合并。接着我们以宽高比作为过滤条件，过滤掉高度比宽度更大的区域（认为单词的长度通常比高度要大）。下图中绿色的区域是经过这些步骤后被认为是文字的区域，而红色的区域是被忽略的。

![bc48a4b0c7257591643eb50f2bf46db6.jpg](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656310035347-2d794582-9ab3-4717-9d3b-5de6606a3e2c.jpeg#clientId=uf4c47708-0dd9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=177&id=u9300f15a&name=bc48a4b0c7257591643eb50f2bf46db6.jpg&originHeight=250&originWidth=401&originalType=binary&ratio=1&rotation=0&showTitle=false&size=13082&status=done&style=none&taskId=u9cadba84-d9e4-4a81-b638-22795c1225f&title=&width=284.6399841308594)

以上便是文字侦测阶段。

下一步是训练一个模型来完成将文字分割成一个个字符的任务，需要的训练集由单个字符的图片和两个相连字符之间的图片来训练模型。

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656310218923-f0b3f4b3-43a1-430a-a664-5c94452978a7.jpeg#clientId=uf4c47708-0dd9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=211&id=u5ca3a10d&originHeight=1052&originWidth=2154&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u3d0e03e1-e02e-4863-802b-265c28da262&title=&width=433)

模型训练完后，我们仍然是使用滑动窗口技术来进行字符识别。

最后一个阶段是字符分类阶段，利用神经网络、支持向量机或者逻辑回归算法训练一个分类器即可。

---

<a name="Ew5iV"></a>
## 获取大量数据和人工数据

对于低偏差的模型，采用大量的数据进行训练，往往能够得到一个较好的结果的。除了直接获取大量数据之外，还有人工合成的方法。

常见人工数据合成方法：

1.  人工数据合成 
1.  手动收集、标记数据 
1.  众包 （即通过专门的打标签平台把任务发放出去）

以我们的文字识别应用为例，我们可以字体网站下载各种字体，然后利用这些不同的字体配上各种不同的随机背景图片创造出一些用于训练的实例，这让我们能够获得一个无限大的训练集。这是从零开始创造实例。

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656310407414-dae2f2e6-f293-49c1-ba41-635b13e24c93.jpeg#clientId=uf4c47708-0dd9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=210&id=u6c621337&originHeight=1006&originWidth=2102&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=ud3367e29-585b-4c4d-862d-8c8d52d8578&title=&width=439)

另一种方法是，利用已有的数据，然后对其进行修改，例如将已有的字符图片进行一些扭曲、旋转、模糊处理。只要我们认为实际数据有可能和经过这样处理后的数据类似，我们便可以用这样的方法来创造大量的数据。

![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656310425709-baa2152e-8185-4a16-a2f3-399fb94663da.jpeg#clientId=uf4c47708-0dd9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=141&id=u75cf1db6&originHeight=1020&originWidth=1832&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u29780097-8132-4353-afb4-04e77cbe3e7&title=&width=254)


注意，通常将纯随机/无意义的噪声引入数据集对模型是没有帮助的，引入的噪声应该是在测试集中有可能出现的噪声，这才是有助于提升模型效果的。

---

<a name="yfTwG"></a>
## 上限分析：哪部分值得优化

在机器学习的应用中，我们通常需要通过几个步骤才能进行最终的预测，我们如何能够知道哪一部分最值得我们花时间和精力去改善呢？这个问题可以通过上限分析来回答。

回到我们的文字识别应用中，我们的流程图如下：

![](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656309610074-176532ab-9998-48d7-a807-6da43278bd58.png#crop=0&crop=0&crop=1&crop=1&from=url&height=57&id=KPd1k&originHeight=85&originWidth=890&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=595)

流程图中每一部分的输出都是下一部分的输入，因此每一部分的准确率提高了，都会影响整个系统，所以为了判断哪一部分更值得我们花费时间精力，我们需要采用控制变量的思想来进行上限分析。

比如我们另文字检测部分输出的结果100%正确（人工输入100%正确的数据给字符分割），发现系统的总体准确率从72%提升到89%。以此类推，我们分别往字符分割，字符识别模块输入100%准确的数据，然后计算整个系统的准确率，得到下面的表格：

| **Component** | **Accuracy** |
| --- | --- |
| Overall system | 72% |
| Text detection | 89% |
| Character segmentation | 90% |
| Character recognition | 100% |


如果我们令文字侦测部分输出的结果100%正确，发现系统的总体效果从72%提高到了89%。有17%的较大提升，这意味着我们很可能会希望投入时间精力来提高我们的文字侦测部分。

如果我们让字符切分输出的结果100%正确，发现系统的总体效果只提升了1%，这意味着，我们的字符切分部分可能已经足够好了，不需要花费时间精力去提高了。

最后我们让字符分类输出的结果100%正确，系统的总体效果又提升了10%，这意味着我们可能也会应该投入更多的时间和精力来提高应用的总体表现。


再以人脸检测为例：<br />![](https://cdn.nlark.com/yuque/0/2022/jpeg/12563972/1656311054668-f3ccb460-3649-4f41-a2a6-c686729af876.jpeg#clientId=uf4c47708-0dd9-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=235&id=u865edfb7&originHeight=1080&originWidth=2074&originalType=url&ratio=1&rotation=0&showTitle=false&status=done&style=none&taskId=u5c7080c4-1752-4408-a044-98cb159b17e&title=&width=452)

同样地，我们可以按照模块分别进行上限分析，计算系统总体及各部分准确率如下：

| **Component** | **Accuracy** |
| --- | --- |
| Overall system | 85% |
| Preprocess (remove background) | 85.1% |
| Face detection | 91% |
| Eyes segmentation | 95% |
| Nose segmentation | 96% |
| Mouth segmentation | 97% |
| Logistic regression | 100% |

从上表可以发现，我们最不应该花时间提升的就是“Preprocess(remove background)”模块，因为即使这个模块达到了100%的准确率，最终整体效果也只提升了0.1%，对整个模型效果提升不大；相反，最值得研究的部分是“Face detection”和"Mouth segmentation"模块。

上限分析能够让我们找到最应该花费时间精力投入的地方，而不是浪费时间研究一些对总体效果提升不大的模块。

---

<a name="nfdjH"></a>
# 机器学习实践
<a name="T1zds"></a>
## 官方文档链接
[pandas.DataFrame.hist — pandas 1.4.3 documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.hist.html?highlight=hist#pandas.DataFrame.hist)<br />[pandas.DataFrame.plot — pandas 1.4.3 documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html?highlight=plot#pandas.DataFrame.plot)

[seaborn.boxplot — seaborn 0.11.2 documentation](https://seaborn.pydata.org/generated/seaborn.boxplot.html?highlight=boxplot#seaborn.boxplot)

[scipy.stats.probplot — SciPy v1.8.1 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.probplot.html?highlight=probplot#scipy.stats.probplot)

[sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html?highlight=stand#sklearn.preprocessing.StandardScaler)<br />[sklearn.model_selection.GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV)<br />[sklearn.model_selection.RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html?highlight=searchcv#sklearn.model_selection.RandomizedSearchCV)<br />[sklearn.preprocessing.MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html?highlight=minmax#sklearn.preprocessing.MinMaxScaler)<br />[sklearn.preprocessing.Normalizer](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html?highlight=norma#sklearn.preprocessing.Normalizer)

```python
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model selection import RandomizedSearchcv

%matplot inline # 使用Jupyter时需要加上此条
```

---

<a name="DHU9T"></a>
## 数据特征观察
```python
# 查看训练集特征变量信息
train_data.info()

#查看训练数据统计信息（count,min,max,std等）
train_data.describe()

#查看前5行数据信息
train_data.head()
```
```python
# 方式一：pandas
train_data.hist(bins=50 ,figsize=(20,15))
plt.show() # jupyter中可选

#方式二：seaborn
sns.histplot(train_data['V0'],stat='count')

#方式三：seaborn
sns.distplot(train_data['V0'],fit=stats.norm) # 以密度绘制并显示正态分布曲线
```
示例![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656993143160-8ab76505-bf7b-49d5-967b-723d0effcaf4.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=225&id=yICJH&name=image.png&originHeight=327&originWidth=481&originalType=binary&ratio=1&rotation=0&showTitle=true&size=11329&status=done&style=none&taskId=u6a636919-46e9-4318-8397-5e22ec510e2&title=pandas&width=331 "pandas")![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656993216933-d2acb792-9b69-4748-8d15-c06e7fd93577.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=237&id=FZZKr&name=image.png&originHeight=391&originWidth=539&originalType=binary&ratio=1&rotation=0&showTitle=true&size=15700&status=done&style=none&taskId=u061118e2-d797-44c8-87b3-90a46d1e25c&title=seanborn-1&width=327 "seanborn-1")![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656993622476-8110a4e7-cf5a-4d93-8970-721d25d265f7.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=341&id=zoLcC&name=image.png&originHeight=372&originWidth=716&originalType=binary&ratio=1&rotation=0&showTitle=true&size=37631&status=done&style=none&taskId=ua679e6c8-6e37-41f4-a108-bcd422270e1&title=seaborn-2&width=656 "seaborn-2")

```python
sns.boxplot(train_data['V0'],orient="v", width=0.5) # 绘制特征V0的箱型图
```
示例![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656999076877-ee957db8-c395-4d72-bf4c-aca181b9f14d.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=257&id=nXMEb&name=image.png&originHeight=322&originWidth=434&originalType=binary&ratio=1&rotation=0&showTitle=false&size=9086&status=done&style=none&taskId=u5abf90e4-44a7-4c84-a64d-f4c3665b5c6&title=&width=347)

```python
# Q-Q图是数据分位数和正态分布分位数对比参照图
res = stats.probplot(train_data['V0'], plot=plt) #检验数据是否近似符合正态分布
```
示例![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656999169099-61f84bb4-ec2b-4743-8940-e070d4a551e1.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=386&id=jOBe6&name=image.png&originHeight=386&originWidth=341&originalType=binary&ratio=1&rotation=0&showTitle=false&size=24916&status=done&style=none&taskId=u39ebf2ba-c6a5-43d2-9467-e2187990f3e&title=&width=341)
```python
# 对比同特征下，训练集和测试集数据分布是否一致
ax = sns.kdeplot(train_data['V0'], color="Red", shade=True)
ax = sns.kdeplot(test_data['V0'], color="Blue", shade=True)
ax = ax.legend(["train","test"]) # 图例
```
示例![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656999817241-993720cd-79f3-4e8d-bbfd-428444f9bc5c.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=288&id=u20da92aa&name=image.png&originHeight=288&originWidth=454&originalType=binary&ratio=1&rotation=0&showTitle=false&size=22958&status=done&style=none&taskId=u9f8bbe54-d293-4661-9792-cacacfc6592&title=&width=454)
```python
# 方式一
# 查看特征变量‘V0’与'target'变量的线性回归关系
plt.figure(figsize=(8,4))
ax=plt.subplot(1,2,1)
sns.regplot(x='V0', y='target', data=train_data, ax=ax, 
            scatter_kws={'marker':'.','s':3,'alpha':0.3},
            line_kws={'color':'k'}); # apha为透明度
plt.xlabel('V0')
plt.ylabel('target')
plt.show()


#方式二
plt.figure(figsize=(4,4))
ax=plt.subplot(1,1,1)
train_data.plot(kind="scatter", x='V0', y='target', alpha=0.1,ax=ax)
plt.show()
```
示例![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1657000941495-b0b39342-38a9-4637-8dd3-852eb5f16399.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=319&id=u2e3d4cd3&name=image.png&originHeight=319&originWidth=309&originalType=binary&ratio=1&rotation=0&showTitle=true&size=45873&status=done&style=none&taskId=u9967480c-0882-49ee-bbac-7e263c4e0b8&title=seaborn&width=309 "seaborn")![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1657003617462-1b29a9dd-c8d1-42dc-bbdd-58b08be9c0b7.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=324&id=u3cea0b77&name=image.png&originHeight=324&originWidth=330&originalType=binary&ratio=1&rotation=0&showTitle=true&size=49920&status=done&style=none&taskId=uc4bd850a-6f96-4dbb-b33c-0ec0f8d83a8&title=pandas&width=330 "pandas")

```python
train_corr = train_data.corr()
```
```python
# 画出相关性热力图
ax = plt.subplots(figsize=(20, 16))#调整画布大小
ax = sns.heatmap(train_corr, vmax=.8, square=True, annot=True)#画热力图   annot=True 显示系数
```
示例![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1657001366520-7d34a31a-436b-44a6-9163-cba9a8ef4162.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=632&id=u48e7d153&name=image.png&originHeight=632&originWidth=716&originalType=binary&ratio=1&rotation=0&showTitle=false&size=357794&status=done&style=none&taskId=u204a49b0-99ed-4459-befd-57d9591712f&title=&width=716)

```python
k = 10 # number of variables for heatmap
cols = train_corr.nlargest(k, 'target')['target'].index
# 然后仍可绘出热力图，或相关性散点图
```
```python
k = 10 # number of variables for heatmap
cols = train_corr.nlargest(k, 'target')['target'].index # 只绘制出最相关的10个特征

# scatter_matrix函数绘制与其他属性间的相关性
# 每个属性自身与自身的相关图替换为其直方图
scatter_matrix(train_data[cols], figsize=(12, 8))
save_fig("scatter_matrix_plot")
```
示例![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1657002367731-eedbed39-4525-44e7-8ae9-913d3b390391.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=444&id=ufe601600&name=image.png&originHeight=444&originWidth=680&originalType=binary&ratio=1&rotation=0&showTitle=false&size=201531&status=done&style=none&taskId=ud913c62d-1611-40ba-9cf7-32ee7db571f&title=&width=680)

```python
#使用Scikit-Learn划分训练集和数据集
train_set, test_set = train_test_split(X_data, test_size=0.2, random_state=42)
```
```python
from sklearn.model_selection import StratifiedShuffleSplit

# 测试集占总数据20%
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 

# 根据V0分层抽样
for train_index, test_index in split.split(housing, X_data["V0"]): 
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```

---

<a name="d9A6A"></a>
## 特征处理
```python
# 异常值分析
plt.figure(figsize=(18, 10))
plt.boxplot(x=train_data.values,labels=train_data.columns)
plt.hlines([-7.5, 7.5], 0, 40, colors='r')
plt.show()

# 删除特征值
train_data = train_data[train_data['V9']>-7.5]
```
示例![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1657006117180-9370538d-1cb7-4e39-bb4c-dcc5c89e6809.png#clientId=u95b06e87-f250-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=396&id=u7aab20f2&name=image.png&originHeight=396&originWidth=720&originalType=binary&ratio=1&rotation=0&showTitle=false&size=53251&status=done&style=none&taskId=ufbc319f7-998d-43b3-ad9e-587130d2979&title=&width=720)
```python
s=StandardScaler()
s.fit_tranform(train_data) # 返回标准化后的数据
```
```python
m = MinMaxScaler(feature_range=(0,1)) 
m.fit_transform(train_data) # 返回缩放到区间（0,1）的数据
```
```python
# 归一化是将样本数据转化到同一量纲下，区间缩放是归一化的一种
n = Normalizer() 
n.fit_transform()
```
scikit-learn中相关类总结

| **类** | **功能** | **说明** |
| --- | --- | --- |
| StandardScaler | 无量纲化 | 标准化，基于特征矩阵的列，将特征值转换为服从标准正态分布 |
| MinMaxScaler | 无量纲化 | 区间缩放，基于最大值或最小值，将特征值转换到[o,1]区间内 |
| Normalizcr | 归一化 | 基于特征矩阵的行，将样本向量转换为单位向量 |
| Binarizer | 定量特征二值化 | 基于给定阈值，将定量特征按阈值划分 |
| OncHotEncoder | 定性特征哑编码 | 将定性特征编码为定量特征 |
| Imputer | 缺失值处理 | 计算缺失值，缺失值可填充为均值等 |
| PolynomialFeatures | 多项式数据转换 | 多项式数据转换 |
| FunctionTransformer | 自定义单元数据转换 | 使用单变元函数转换数据 |


降维部分见前文相关章节

---

<a name="jXR7y"></a>
## 调参（参数选择）
```python
# 先评估第一个dict中的3×4种组合，再设置bootsrap为false并尝试第二个中2×3种组合
param grid = [
    {'n_estimators' : [3,10,30], 'max_features' : [2,4,6,8]}，
    {'bootstrap' : [False], 'n_estimators': [3,10], 'max_features ': [2,3,4]},]

forest_reg = RandomForestRegressor () # 以随机森林为例

# 进行网格搜索所有参数组合
grid_search = Gridsearchcv( forest_reg,param_grid, cv=5,
                            scoring= 'neg_mean_squared_error',
                            return_train_score=True)

grid_search.fit(train_data,train_target)
```
```python
param grid = [
    {'n_estimators' : [3,10,30], 'max_features' : [2,4,6,8]}，
    {'bootstrap' : [False], 'n_estimators': [3,10], 'max_features ': [2,3,4]},]

forest_reg = RandomForestRegressor () # 以随机森林为例

# 进行随机搜索
grid_search = RandomizedsearchCV ( forest_reg,param_grid, cv=5,
                                   scoring= 'neg_mean_squared_error',
                                   return_train_score=True)

grid_search.fit(train_data,train_target)
```
