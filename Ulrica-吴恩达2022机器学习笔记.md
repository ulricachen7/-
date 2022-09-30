# 机器学习课程
视频网址：【【吴恩达-2022-中英字幕】令人醍醐灌顶的机器学习（我愿称之为人工智能AI教程天花板）-哔哩哔哩】 https://b23.tv/LjTVvTr
**很新，但是是机翻**

【[中英字幕]吴恩达机器学习系列课程-哔哩哔哩】 https://b23.tv/prP2CSL
**系统性**

python3.6版的课后题作业代码实现以及各种格式的笔记、数学资料梳理等等，感谢大佬，Github：https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes

## 1 基础概念
机器学习包括监督学习和非监督学习


### 1.1 监督学习
包括回归和分类两种


### 1.2 无监督学习
1⃣️聚类算法（Clustering），因为地方无法将数据分为不同的族。常见应用：1）找到相似的部分并且分为一类，可以用于新闻、视频、文章的推荐；2）可用于对DNA不同序列进行分类，将有相似DNA序列的分为某一种type的人；3）将客户分为不同的市场。2⃣️异常探测（Anomaly detection），用于探测异常事件，对于金融系统的检测十分重要，可以检测出是否可能是欺诈的标志。3⃣️降维（Dimensionality reduction）


## 2 单变量线性回归（linear regreession with one variable）
### 2.1 模型的表示
![image](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ad0718d6e5218be6e6fce9dc775a38e6.png)

回归问题的训练集(Training Set)如下表所示：  
𝑚 代表训练集中实例的数量  
𝑥 代表特征/输入变量  
𝑦 代表目标变量/输出变量  
(𝑥,𝑦) 代表训练集中的实例  
(𝑥(𝑖),𝑦(𝑖)) 代表第𝑖 个观察实例  
h 代表学习算法的解决方案或函数也称为假设(hypothesis)  

### 2.2 代价函数
需要为模型选择合适的参数(parameters)，模型所预测的值和实际值之间的差距就是建模误差，如下图： 

![](https://github.com/ulricachen7/markdown_image/blob/main/吴恩达机器学习/2.2建模误差.png?raw=true)

目标是选择出可以是建模误差的平方和最小的模型参数，及使得代价函数最小。

$J \left( \theta_0, \theta_1 \right) = \frac{1}{2m}\sum\limits_{i=1}^m \left( h_{\theta}(x^{(i)})-y^{(i)} \right)^{2}$


绘制一个等高线图，三个坐标分别为$\theta_{0}$和$\theta_{1}$ 和$J(\theta_{0}, \theta_{1})$：

> 代价函数分母为什么有2，是后面要用梯度下降法，要求导，这样求导多出的乘2就和二分之一抵消了，一个简化后面计算的技巧

![image](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/27ee0db04705fb20fab4574bb03064ab.png).

在三维空间中存在一个使得$J(\theta_{0}, \theta_{1})$最小的点。
代价函数也被称作**平方误差函数**，有时也被称为**平方误差代价函数**。

### 2.3 代价函数的直观理解I
$h(x)$:假设函数。$J(x)$:代价函数。
![image](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/2c9fe871ca411ba557e65ac15d55745d.png)

### 2.4 代价函数的直观理解II
2.3只改变了单变量线性方程的一个值，这一节$\theta_{0}$, $\theta_{1}$两个参数都改变从而找到$J(\theta_{0}, \theta_{1})$最小的值

![image](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/0b789788fc15889fe33fb44818c40852.png)

三维图不够直观，所以选择等高线图，下图右：

![image](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/86c827fe0978ebdd608505cd45feb774.png)

### 2.5 梯度下降
用于求函数最小值的算法，思想是开始时随机选择一个参数的组合$\left( {\theta_{0}},{\theta_{1}},......,{\theta_{n}} \right)$，计算代价函数，然后寻找下一个能让代价函数值下降最多的参数组合。直到找到一个局部最小值（local minimum），因为我们并没有尝试完所有的参数组合，所以不能确定我们得到的局部最小值是否便是全局最小值（global minimum），选择不同的初始参数组合，可能会找到不同的局部最小值。


![image](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/db48c81304317847870d486ba5bb2015.jpg)

批量梯度下降（batch gradient descent）算法的公式为：

> :=是指赋值

![image](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/7da5a5f635b1eb552618556f1b4aac1a.png)

其中$a$是学习率（learning rate），**$a$决定了我们沿着能让代价函数下降程度最大的方向向下迈出的步子有多大**，在批量梯度下降中，我们每一次都同时让所有的参数减去学习速率乘以代价函数的导数。

![image](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ef4227864e3cabb9a3938386f857e938.png)

**需要同时更新$J(\theta_{0}, \theta_{1})$**

### 2.6 梯度下降的直观理解
梯度下降的过程导数也在减小越来越接近0，在梯度下降法中，当我们接近局部最低点时，梯度下降法会自动采取更小的幅度，这是由于在接近局部最低点时导数值会变小，从而导致梯度下降的幅度减小，因此不需要额外减小$a$

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/4668349e04cf0c4489865e133d112e98.png)

### 2.7 梯度下降的线性回归

梯度下降法和线性回归算法比较如图：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/5eb364cc5732428c695e2aa90138b01b.png)

对我们之前的线性回归问题运用梯度下降法，关键在于求出代价函数的导数，即：

$\frac{\partial }{\partial {{\theta }{j}}}J({{\theta }{0}},{{\theta }{1}})=\frac{\partial }{\partial {{\theta }{j}}}\frac{1}{2m}{{\sum\limits_{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}^{2}}$

$j=0$ 时：$\frac{\partial }{\partial {{\theta }{0}}}J({{\theta }{0}},{{\theta }{1}})=\frac{1}{m}{{\sum\limits{i=1}^{m}{\left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}}}$

$j=1$ 时：$\frac{\partial }{\partial {{\theta }{1}}}J({{\theta }{0}},{{\theta }{1}})=\frac{1}{m}\sum\limits{i=1}^{m}{\left( \left( {{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

则算法改写成：

Repeat {

​ ${\theta_{0}}:={\theta_{0}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{ \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)}$

​ ${\theta_{1}}:={\theta_{1}}-a\frac{1}{m}\sum\limits_{i=1}^{m}{\left( \left({{h}_{\theta }}({{x}^{(i)}})-{{y}^{(i)}} \right)\cdot {{x}^{(i)}} \right)}$

​ }
​ 
> ​批量梯度下降：在梯度下降的每一步当中都使用到了所有的训练样本

> 正规方程（normal equations）：不需要走很多步梯度下降就可以求解出代价函数最小值

> 在数据量较大的情况下，梯度下降法比正规方程更为适用

## 3 线性代数回顾

### 3.1 矩阵和向量

> 矩阵的维数是行数*列数

> 讲义中的向量一般都是列向量

### 3.2 加法和标量乘法

标量乘法值要乘以矩阵中所有的元素

### 3.3 矩阵向量乘法

矩阵和向量的乘法如图：$m×n$的矩阵乘以$n×1$的向量，得到的是$m×1$的向量

### 3.4 矩阵乘法

$m×n$矩阵乘以$n×o$矩阵，变成$m×o$矩阵。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/1a9f98df1560724713f6580de27a0bde.jpg)

### 3.5 矩阵乘法的性质

矩阵乘法的性质：  
矩阵的乘法不满足交换律：$A×B≠B×A$  
矩阵的乘法满足结合律。即：$A×(B×C)=(A×B)×C$  

单位矩阵：在矩阵的乘法中，有一种矩阵起着特殊的作用，如同数的乘法中的1,我们称这种矩阵为单位矩阵．它是个方阵，一般用 $I$ 或者 $E$ 表示，本讲义都用 $I$ 代表单位矩阵，从左上角到右下角的对角线（称为主对角线）上的元素均为1以外全都为0。如：  
$A{{A}^{-1}}={{A}^{-1}}A=I$  
对于单位矩阵，有$AI=IA=A$

### 3.6 逆、转置

矩阵的逆：如**矩阵$A$是一个$m×m$矩阵（方阵）**，如果有逆矩阵，则：$A{{A}^{-1}}={{A}^{-1}}A=I$

一般在**OCTAVE或者MATLAB中**进行计算矩阵的逆矩阵。

矩阵的转置：设$A$为$m×n$阶矩阵（即$m$行$n$列），第$i $行$j $列的元素是$a(i,j)$，即：$A=a(i,j)$

定义$A$的转置为这样一个$n×m$阶矩阵$B$，满足$B=a(j,i)$，即 $b (i,j)=a(j,i)$（$B$的第$i$行第$j$列元素是$A$的第$j$行第$i$列元素），记${{A}^{T}}=B$。(有些书记为A'=B）

直观来看，将$A$的所有元素绕着一条从第1行第1列元素出发的右下方45度的射线作镜面反转，即得到$A$的转置。

矩阵的转置基本性质:

$ {{\left( A\pm B \right)}^{T}}={{A}^{T}}\pm {{B}^{T}} $ ${{\left( A\times B \right)}^{T}}={{B}^{T}}\times {{A}^{T}}$ ${{\left( {{A}^{T}} \right)}^{T}}=A $ ${{\left( KA \right)}^{T}}=K{{A}^{T}} $

matlab中矩阵转置：直接打一撇，x=y'。

## 4 多变量线性回归

### 4.1 多维特征

支持多变量的假设函数 $h(x)$ 表示为：$h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$

公式可以简化为：$h_{\theta} \left( x \right)={\theta^{T}}X$

### 4.2 多变量梯度下降

多变量线性回归的批量梯度下降算法为：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/41797ceb7293b838a3125ba945624cf6.png)

即：
![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/6bdaff07783e37fcbb1f8765ca06b01b.png)

求导数后得到：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/dd33179ceccbd8b0b59a5ae698847049.png)

### 4.3 梯度下降法实践1-特征缩放

以房价问题为例，假设我们使用两个特征：房屋的尺寸和房间的数量，尺寸的值为 0-2000平方英尺，而房间数量的值则是0-5，以两个参数分别为横纵坐标，绘制代价函数的等高线图能，看出图像会显得很扁，梯度下降算法需要非常多次的迭代才能收敛。

解决的方法是尝试将所有特征的尺度都尽量缩放到-1到1之间。如图：  
![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/b8167ff0926046e112acf789dba98057.png)  
最简单的方法是均值归一化(mean normalization)：${x_{n}}=\frac{{x_{n}}-{\mu_{n}}}{{s_{n}}}$，其中 ${\mu_{n}}$是平均值，${s_{n}}$是标准差。

### 4.4 梯度下降法实践2-学习率

梯度下降算法的每次迭代受到学习率的影响，**如果学习率$a$过小，则达到收敛所需的迭代次数会非常高；如果学习率$a$过大，每次迭代可能不会减小代价函数，可能会越过局部最小值导致无法收敛**。

![](https://github.com/ulricachen7/markdown_image/blob/main/吴恩达机器学习/4.4%20梯度下降法学习率.png?raw=true)

通常可以考虑尝试些学习率：  
**$\alpha=0.01，0.03，0.1，0.3，1，3，10$**

### 4.5 特征和多项式回归

线性回归并不适用于所有数据，有时我们需要曲线来适应我们的数据，比如一个二次方模型：$h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}^2}$ 或者三次方模型： $h_{\theta}\left( x \right)={\theta_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}^2}+{\theta_{3}}{x_{3}^3}$

根据函数图形特性，我们还可以使：

${{{h}}{\theta}}(x)={{\theta }{0}}\text{+}{{\theta }{1}}(size)+{{\theta}{2}}{{(size)}^{2}}$

或者:

${{{h}}{\theta}}(x)={{\theta }{0}}\text{+}{{\theta }{1}}(size)+{{\theta }{2}}\sqrt{size}$

注：如果我们采用多项式回归模型，在运行梯度下降算法前，特征缩放非常有必要。

### 4.6 正规方程

正规方程是通过求解下面的方程来找出使得代价函数最小的参数的：$\frac{\partial}{\partial{\theta_{j}}}J\left( {\theta_{j}} \right)=0$ 。 假设我们的训练集特征矩阵为 $X$（包含了 ${{x}_{0}}=1$）并且我们的训练集结果为向量 $y$，则利用正规方程解出向量 $\theta ={{\left( {X^T}X \right)}^{-1}}{X^{T}}y$ 。 上标T代表矩阵转置，上标-1 代表矩阵的逆。设矩阵$A={X^{T}}X$，则：${{\left( {X^T}X \right)}^{-1}}={A^{-1}}$ 以下表示数据为例：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/261a11d6bce6690121f26ee369b9e9d1.png)

即：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/c8eedc42ed9feb21fac64e4de8d39a06.png)

运用正规方程方法求解参数：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/b62d24a1f709496a6d7c65f87464e911.jpg)

注：对于那些不可逆的矩阵（通常是因为特征之间不独立，如同时包含英尺为单位的尺寸和米为单位的尺寸两个特征，也有可能是特征数量大于训练集的数量），正规方程方法是不能用的。

梯度下降与正规方程的比较：

|  梯度下降   | 正规方程  |
|  ----  | ----  |
|  需要选择学习率$\alpha$  |  不需要  |  
|  需要多次迭代  |  一次运算得出  |
|  当特征数量$n$大时也能较好适用	|  需要计算${{\left( {{X}^{T}}X \right)}^{-1}}$ 如果特征数量n较大则运算代价大，因为矩阵逆的计算时间复杂度为$O\left( {{n}^{3}} \right)$，通常来说当$n$小于10000 时还是可以接受的
|  适用于各种类型的模型  |	只适用于线性模型，不适合逻辑回归模型等其他模型  |

python实现

```
import numpy as np
    
 def normalEqn(X, y):
    
   theta = np.linalg.inv(X.T@X)@X.T@y #X.T@X等价于X.T.dot(X)
    
   return theta
```

### 4.7 正规方程及不可逆
不可逆矩阵也称为奇异矩阵  

矩阵不可逆的两个原因  
1. 特征值中有一些多余的特征，这些特征存在线性相关，互为线性函数，可以删除其中一个。 
2. 特征值太多，但是数据集很少，删除一些特征用较少的特征来反映更多的内容，否者考虑使用**正规化方法**。

>**增加内容:$\theta ={{\left( {X^{T}}X \right)}^{-1}}{X^{T}}y$ 的推导过程：**
>$J\left( \theta \right)=\frac{1}{2m}\sum\limits_{i=1}^{m}{{{\left( {h_{\theta}}\left( {x^{(i)}} \right)-{y^{(i)}} \right)}^{2}}}$ 其中：${h_{\theta}}\left( x \right)={\theta^{T}}X={\theta_{0}}{x_{0}}+{\theta_{1}}{x_{1}}+{\theta_{2}}{x_{2}}+...+{\theta_{n}}{x_{n}}$
>
>将向量表达形式转为矩阵表达形式，则有$J(\theta )=\frac{1}{2}{{\left( X\theta -y\right)}^{2}}$ ，其中$X$为$m$行$n$列的矩阵（$m$为样本个数，$n$为特征个数），$\theta$为$n$行1列的矩阵，$y$为$m$行1列的矩阵，对$J(\theta )$进行如下变换
>
>$J(\theta )=\frac{1}{2}{{\left( X\theta -y\right)}^{T}}\left( X\theta -y \right)$
>​ $=\frac{1}{2}\left( {{\theta }^{T}}{{X}^{T}}-{{y}^{T}} \right)\left(X\theta -y \right)$
>$=\frac{1}{2}\left( {{\theta }^{T}}{{X}^{T}}X\theta -{{\theta}^{T}}{{X}^{T}}y-{{y}^{T}}X\theta -{{y}^{T}}y \right)$
>
>接下来对$J(\theta )$偏导，需要用到以下几个矩阵的求导法则:
>
>$\frac{dAB}{dB}={{A}^{T}}$
>$\frac{d{{X}^{T}}AX}{dX}=2AX$
>
>所以有:
>
>$\frac{\partial J\left( \theta \right)}{\partial \theta }=\frac{1}{2}\left(2{{X}^{T}}X\theta -{{X}^{T}}y -{}({{y}^{T}}X )^{T}-0 \right)$
>$=\frac{1}{2}\left(2{{X}^{T}}X\theta -{{X}^{T}}y -{{X}^{T}}y -0 \right)$
>$={{X}^{T}}X\theta -{{X}^{T}}y$
>
令$\frac{\partial J\left( \theta \right)}{\partial \theta }=0$,
>则有$\theta ={{\left( {X^{T}}X \right)}^{-1}}{X^{T}}y$
>

## 5 （logistics）逻辑回归
### 5.1 假说表示
因变量(dependent variable)可能属于的两个类分别称为负向类（negative class）和正向类（positive class），则因变量y∈{0,1}，其中 0 表示负向类，1 表示正向类  

逻辑回归模型的假设是： $h_\theta \left( x \right)=g\left(\theta^{T}X \right)$ 其中： $X$ 代表特征向量 $g$ 代表逻辑函数（logistic function)是一个常用的逻辑函数为S形函数（Sigmoid function），公式为： $g\left( z \right)=\frac{1}{1+{{e}^{-z}}}$。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/1073efb17b0d053b4f9218d4393246cc.jpg)

```
import numpy as np
    
def sigmoid(z):
    
   return 1 / (1 + np.exp(-z))

```

### 5.2 判定边界
![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/f71fb6102e1ceb616314499a027336dc.jpg)

### 5.3 逻辑回归的代价函数(cost function)
将${h_\theta}\left( x \right)=\frac{1}{1+{e^{-\theta^{T}x}}}$带入到这样定义了的代价函数中时，得到的代价函数将是一个非凸函数（non-convexfunction）

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/8b94e47b7630ac2b0bcb10d204513810.jpg)

线性回归的代价函数为：

$J\left( \theta \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{\frac{1}{2}{{\left( {h_\theta}\left({x}^{\left( i \right)} \right)-{y}^{\left( i \right)} \right)}^{2}}}$  

我们重新定义逻辑回归的代价函数为：

$J\left( \theta \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{{Cost}\left( {h_\theta}\left( {x}^{\left( i \right)} \right),{y}^{\left( i \right)} \right)}$


![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/54249cb51f0086fa6a805291bf2639f1.png)

${h_\theta}\left( x \right)$

与 

$Cost\left( {h_\theta}\left( x \right),y \right)$之间的关系如下图所示：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ffa56adcc217800d71afdc3e0df88378.jpg)

```
import numpy as np
    
def cost(theta, X, y):
    
  theta = np.matrix(theta)
  X = np.matrix(X)
  y = np.matrix(y)
  first = np.multiply(-y, np.log(sigmoid(X* theta.T)))
  second = np.multiply((1 - y), np.log(1 - sigmoid(X* theta.T)))
  return np.sum(first - second) / (len(X))
  
```

### 5.4 简化的逻辑回归的代价函数(cost function)

式子可以合并成：

$Cost\left( {h_\theta}\left( x \right),y \right)=-y\times log\left( {h_\theta}\left( x \right) \right)-(1-y)\times log\left( 1-{h_\theta}\left( x \right) \right)$ 

### 5.5 过拟合（overfit）

underfit 欠拟合 bias
just right 泛化 generalization 
overfit 过拟合 high variance

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/be39b497588499d671942cc15026e4a2.jpg)

### 5.6 解决过拟合的方法

1. 获取更多的数据
2. 选择较少的特征
3. 选择正则化函数

### 5.7 正则化代价函数

$J\left( \theta \right)=\frac{1}{2m}[\sum\limits_{i=1}^{m}{{{({h_\theta}({{x}^{(i)}})-{{y}^{(i)}})}^{2}}+\lambda \sum\limits_{j=1}^{n}{\theta_{j}^{2}}]}$

其中\lambda 又称为正则化参数（Regularization Parameter）。 注：根据惯例，我们不对{\theta_{0}} 进行惩罚。经过正则化处理的模型与原模型的可能对比如下图所示：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ea76cc5394cf298f2414f230bcded0bd.jpg)

为什么增加的一项\lambda =\sum\limits_{j=1}^{n}{\theta_j^{2}} 可以使\theta 的值减小呢？ 因为如果我们令 \lambda 的值很大的话，为了使Cost Function 尽可能的小，所有的 \theta  的值（不包括{\theta_{0}}）都会在一定程度上减小。 但若\lambda 的值太大了，那么\theta （不包括{\theta_{0}}）都会趋近于0

## 6 （Neural Networks）神经网络
### 6.1 神经网络的构建
为了构建神经网络模型，我们需要首先思考大脑中的神经网络是怎样的？每一个神经元都可以被认为是一个处理单元/神经核（processing unit/Nucleus），它含有许多输入/树突（input/Dendrite），并且有一个输出/轴突（output/Axon）。神经网络是大量神经元相互链接并通过电脉冲来交流的一个网络。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/3d93e8c1cd681c2b3599f05739e3f3cc.jpg)

设计出了类似于神经元的神经网络，效果如下：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/fbb4ffb48b64468c384647d45f7b86b5.png)

其中x_1, x_2, x_3是输入单元（input units），我们将原始数据输入给它们。 a_1, a_2, a_3是中间单元，它们负责将数据进行处理，然后呈递到下一层。 最后是输出单元，它负责计算{h_\theta}\left( x \right)。

神经网络模型是许多逻辑单元按照不同层级组织起来的网络，每一层的输出变量都是下一层的输入变量。下图为一个3层的神经网络，第一层成为输入层（Input Layer），最后一层称为输出层（Output Layer），中间一层成为隐藏层（Hidden Layers）。我们为每一层都增加一个偏差单位（bias unit）：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/8293711e1d23414d0a03f6878f5a2d91.jpg)

下面引入一些标记法来帮助描述模型： a_{i}^{\left( j \right)} 代表第j 层的第 i 个激活单元。{{\theta }^{\left( j \right)}}代表从第 j 层映射到第 j+1 层时的权重的矩阵，例如{{\theta }^{\left( 1 \right)}}代表从第一层映射到第二层的权重的矩阵。其尺寸为：以第 j+1层的激活单元数量为行数，以第 j 层的激活单元数加一为列数的矩阵。例如：上图所示的神经网络中{{\theta }^{\left( 1 \right)}}的尺寸为 3*4。

对于上图所示的模型，激活单元和输出分别表达为：

$a_{1}^{(2)}=g(\Theta {10}^{(1)}{{x}{0}}+\Theta {11}^{(1)}{{x}{1}}+\Theta {12}^{(1)}{{x}{2}}+\Theta {13}^{(1)}{{x}{3}})$ 

$a_{2}^{(2)}=g(\Theta {20}^{(1)}{{x}{0}}+\Theta {21}^{(1)}{{x}{1}}+\Theta {22}^{(1)}{{x}{2}}+\Theta {23}^{(1)}{{x}{3}})$ 

$a_{3}^{(2)}=g(\Theta {30}^{(1)}{{x}{0}}+\Theta {31}^{(1)}{{x}{1}}+\Theta {32}^{(1)}{{x}{2}}+\Theta {33}^{(1)}{{x}{3}})$ 

${{h}_{\Theta }}(x)=g(\Theta {10}^{(2)}a{0}^{(2)}+\Theta {11}^{(2)}a{1}^{(2)}+\Theta {12}^{(2)}a{2}^{(2)}+\Theta {13}^{(2)}a{3}^{(2)})$

### 6.2 向前传播工作过程
![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654483441239-fb5eebe0-105b-4e90-b40e-6ee6c22bef08.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=300&id=u2ed29dea&name=image.png&originHeight=590&originWidth=1129&originalType=binary&ratio=1&rotation=0&showTitle=false&size=215204&status=done&style=none&taskId=u394f580c-7038-41a4-8eb8-e3b88a929e2&title=&width=574#crop=0&crop=0&crop=1&crop=1&height=316&id=dPnTT&originHeight=590&originWidth=1129&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=&width=605)

相对于使用循环来编码，利用向量化的方法会使得计算更为简便。以上面的神经网络为例，计算第二层的值：

我们令 ${{z}^{\left( 2 \right)}}={{\Theta }^{\left( 1 \right)}}x$，则 ${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$ ，计算后添加 $a_{0}^{\left( 2 \right)}=1$。 计算输出的值为：

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1654484060366-ff859e57-c74d-46c6-84d2-0410e4d6c583.png#clientId=u46ff1742-ab27-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=84&id=ua0d0a48a&name=image.png&originHeight=85&originWidth=573&originalType=binary&ratio=1&rotation=0&showTitle=false&size=11481&status=done&style=none&taskId=u0e351ef3-47f8-4b36-8f56-4347c8f5477&title=&width=563.727294921875#crop=0&crop=0&crop=1&crop=1&id=dAdTF&originHeight=85&originWidth=573&originalType=binary&ratio=1&rotation=0&showTitle=false&status=done&style=none&title=)

我们令 ${{z}^{\left( 3 \right)}}={{\Theta }^{\left( 2 \right)}}{{a}^{\left( 2 \right)}}$，则 $h_\theta(x)={{a}^{\left( 3 \right)}}=g({{z}^{\left( 3 \right)}})$。

这只是针对训练集中一个训练实例所进行的计算。如果我们要对整个训练集进行计算，我们需要将训练集特征矩阵进行转置，使得同一个实例的特征都在同一列里。即：<br />${z}^{(2)}={{\Theta }^{\left( 1 \right)}}\times {{X}^{T}}\\$<br />${{a}^{\left( 2 \right)}}=g({{z}^{\left( 2 \right)}})$

### 6.2 如何选择激活函数
**输出层**
二分类 用 sigmoid
预测 用 linear activation function
预测值都是正数时 用 ReLU

**隐藏层**
很少 用 sigmoid，学习速度很慢  
最常 用 ReLU ，计算简单，效率更高

### 6.3 多类分类

输入向量$x$有三个维度，两个中间层，输出层4个神经元分别用来表示4类，也就是每一个数据在输出层都会出现${{\left[ a\text{ }b\text{ }c\text{ }d \right]}^{T}}$，且$a,b,c,d$中仅有一个为1，表示当前类。下面是该神经网络的可能结构示例：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/f3236b14640fa053e62c73177b3474ed.jpg)

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/685180bf1774f7edd2b0856a8aae3498.png)

神经网络算法的输出结果为四种可能情形之一：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/5e1a39d165f272b7f145c68ef78a3e13.png)

### 6.4 随机初始化
任何优化算法都需要一些初始的参数。如果初始所有参数都为 0，这样的初始方法对于逻辑回归来说是可行的，但是对于神经网络来说是不可行的。如果我们令所有的初始参数都为 0，这将意味着我们第二层的所有激活单元都会有相同的值。同理，如果我们初始所有的参数都为一个非 0 的数，结果也是一样的。

**通常初始参数为正负𝜀之间的随机值**。

>训练神经网络：

>1. 参数的随机初始化
>2. 利用正向传播方法计算所有的$h_{\theta}(x)$
>3. 编写计算代价函数 $J$ 的代码
>4. 利用反向传播方法计算所有偏导数
>5. 利用数值检验方法检验这些偏导数
>6. 使用优化算法来最小化代价函数



## 7 机器学习优化
当我们运用训练好了的模型来预测未知数据的时候发现有较大的误差，我们下一步可以做什么？

> 1.  获得更多的训练样本——解决高方差 
> 1.  尝试减少特征的数量——解决高方差 、
> 1.  尝试获得更多的特征——解决高偏差 
> 1.  尝试增加多项式特征——解决高偏差 
> 1.  尝试减少正则化程度λ——解决高偏差 
> 1.  尝试增加正则化程度λ——解决高方差 

我们不应该随机选择上面的某种方法来改进我们的算法，而是运用一些机器学习诊断法来帮助我们知道上面哪些方法对我们的算法是有效的。

### 7.2 诊断偏差和方差
将训练集和交叉验证集的代价函数误差与多项式的次数绘制在同一张图表上来帮助分析：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/bca6906add60245bbc24d71e22f8b836.png)

对于训练集，当d较小时，模型拟合程度更低，误差较大；随着d的增长，拟合程度提高，误差减小。对于交叉验证集，当 d 较小时，模型拟合程度低，误差较大；但是随着 d的增长，误差呈现先减小后增大的趋势，转折点是我们的模型开始过拟合训练数据集的时候。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/25597f0f88208a7e74a3ca028e971852.png)

**训练集误差和交叉验证集误差近似时：偏差/欠拟合**

**交叉验证集误差远大于训练集误差时：方差/过拟合**

### 7.3 正则化和偏差/方差

选择$\lambda$的方法为：

使用训练集训练出12个不同程度正则化的模型
用12个模型分别对交叉验证集计算的出交叉验证误差
选择得出交叉验证误差最小的模型
运用步骤3中选出模型对测试集计算得出推广误差，我们也可以同时将训练集和交叉验证集模型的代价函数误差与λ的值绘制在一张图表上：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/38eed7de718f44f6bb23727c5a88bf5d.png)

**当 λ较小时，训练集误差较小（过拟合）而交叉验证集误差较大**

**随着 λ的增加，训练集误差不断增加（欠拟合），而交叉验证集误差则是先减小后增加**

### 7.4 学习曲线

学习曲线是学习算法的一个很好的合理检验（sanity check）。学习曲线是将训练集误差和交叉验证集误差作为训练集样本数量（$m$）的函数绘制的图表。

即，如果我们有100行数据，我们从1行数据开始，逐渐学习更多行的数据。思想是：当训练较少行数据的时候，训练的模型将能够非常完美地适应较少的训练数据，但是训练出来的模型却不能很好地适应交叉验证集数据或测试集数据。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/969281bc9b07e92a0052b17288fb2c52.png)

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/973216c7b01c910cfa1454da936391c6.png)

如何利用学习曲线识别高偏差/欠拟合：作为例子，我们尝试用一条直线来适应下面的数据，可以看出，无论训练集有多么大误差都不会有太大改观：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/4a5099b9f4b6aac5785cb0ad05289335.jpg)

也就是说在高偏差/欠拟合的情况下，增加数据到训练集不一定能有帮助。

如何利用学习曲线识别高方差/过拟合：假设我们使用一个非常高次的多项式模型，并且正则化非常小，可以看出，当交叉验证集误差远大于训练集误差时，往训练集增加更多数据可以提高模型的效果。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/2977243994d8d28d5ff300680988ec34.jpg)

也就是说在高方差/过拟合的情况下，增加更多数据到训练集可能可以提高算法效果。

---

在高偏差/欠拟合的情况下，增加数据到训练集不一定能有帮助  
在高方差/过拟合的情况下，增加更多数据到训练集可能可以提高算法效果

---

1. 获得更多的训练样本——解决高方差/过拟合
2. 尝试减少特征的数量——解决高方差/过拟合
3. 尝试增加正则化程度λ——解决高方差/过拟合
4. 尝试获得更多的特征——解决高偏差/欠拟合
5. 尝试增加多项式特征——解决高偏差/欠拟合
6. 尝试减少正则化程度λ——解决高偏差/欠拟合

神经网络的方差和偏差：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/c5cd6fa2eb9aea9c581b2d78f2f4ea57.png)

> 使用较小的神经网络，类似于参数较少的情况，容易导致高偏差和欠拟合，但计算代价较小；使用较大的神经网络，类似于参数较多的情况，容易导致高方差和过拟合，虽然计算代价比较大，但是可以通过正则化手段来调整而更加适应数据。
> 
> 通常选择**较大的神经网络并采用正则化处理**会比采用较小的神经网络效果要好。
> 
> 对于神经网络中的隐藏层的层数的选择，通常从一层开始逐渐增加层数，为了更好地作选择，可以把数据分为训练集、交叉验证集和测试集，针对不同隐藏层层数的神经网络训练神经网络， 然后选择交叉验证集代价最小的神经网络。

## 8 机器学习系统设计

### 8.1 误差分析

构建一个学习算法的推荐方法为：
> 1. 从一个简单的能快速实现的算法开始，实现该算法并用交叉验证集数据测试这个算法
> 2. 绘制学习曲线，决定是增加更多数据，或者添加更多特征，还是其他选择
> 3. 进行误差分析：人工检查交叉验证集中我们算法中产生预测误差的样本，看看这些样本是否有某种系统化的趋势

### 8.2 添加数据

修正已有的训练数据以创造一个新的训练数据，比如在图像数据中将图像数据进行选择、放大、缩小以及模糊处理以增加的目的；或者在语音识别的时候添加一些背景音

### 8.3 迁移学习
包括两个步骤：
1. 下载应用程序相同输入带参数的神经网络，并且已经在大型数据集进行了预训练，输入类型包括图像、音频、文本或者其他内容
2. 根据需求用自己的数据进行训练

1. 只训练输出层的参数
2. 训练所有包括隐藏层的参数 

比如将一个用于识别动物、车辆等的模型用于识别数字

### 8.4 机器学习项目的完整周期

scope project >> collect data >> train model >> deploy in production 
define project >> define and collect data >> traing, error analysis & iterative improvement >> deploy moniter and maintain system


### 8.5 类偏斜的误差度量

查准率（Precision）和查全率（Recall） 我们将算法预测的结果分成四种情况：

正确肯定（True Positive,TP）：预测为真，实际为真
2.正确否定（True Negative,TN）：预测为假，实际为假

3.错误肯定（False Positive,FP）：预测为真，实际为假

4.错误否定（False Negative,FN）：预测为假，实际为真

则：查准率（Precision）=TP/(TP+FP)。例，在所有我们预测有恶性肿瘤的病人中，实际上有恶性肿瘤的病人的百分比，越高越好。

查全率（Recall）=TP/(TP+FN)。例，在所有实际上有恶性肿瘤的病人中，成功预测有恶性肿瘤的病人的百分比，越高越好。

这样，对于我们刚才那个总是预测病人肿瘤为良性的算法，其查全率是0。

|  | | 预测值 | |
|  ----  | ----  | ----  | ----  |
|  | | Positive | Negtive |
| 实际值 | Positive | TP | FN |
|  | Negtive | FP | TN |

### 8.6 查准率和查全率之间的权衡

假使，我们的算法输出的结果在0-1 之间，我们使用阀值0.5 来预测真和假。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ad00c2043ab31f32deb2a1eb456b7246.png)

查准率**(Precision)=TP/(TP+FP)** 例，在所有我们预测有恶性肿瘤的病人中，实际上有恶性肿瘤的病人的百分比，越高越好。

查全率**(Recall)=TP/(TP+FN)**例，在所有实际上有恶性肿瘤的病人中，成功预测有恶性肿瘤的病人的百分比，越高越好。

如果我们希望只在非常确信的情况下预测为真（肿瘤为恶性），即我们希望更高的查准率，我们可以使用比0.5更大的阀值，如0.7，0.9。这样做我们会减少错误预测病人为恶性肿瘤的情况，同时却会增加未能成功预测肿瘤为恶性的情况。

如果我们希望提高查全率，尽可能地让所有有可能是恶性肿瘤的病人都得到进一步地检查、诊断，我们可以使用比0.5更小的阀值，如0.3。

我们可以将不同阀值情况下，查全率与查准率的关系绘制成图表，曲线的形状根据数据的不同而不同：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/84067e23f2ab0423679379afc6ed6caf.png)

我们希望有一个帮助我们选择这个阀值的方法。一种方法是计算F1 值（F1 Score），其计算公式为：

${{F}_{1}}Score:2\frac{PR}{P+R}$

## 9 决策树
### 9.1 决策树概述

决策树是一种基本的分类与回归方法。决策树学习通常包括3个步骤：特征选择、决策树的生成、剪枝。

用决策树分类，**从根结点开始**，对实例的某特征进行测试（决策中每个判定问题都是对某个特征的“测试”），根据测试结果，将实例分配到其子结点；这时，每个子结点对应着该特征的一个取值。如此递归地对实例进行测试并分配，直至达到叶结点。最后将实例分到叶结点的类中。

由决策树的个别结点到叶结点的每一条路径构建一条规则；路径上的内部结点对应着规则的条件，而叶结点的类对应着规则的结论，即每个测试的结果或是导出最终结论，或是导出进一步的判定问题。

决策树学习的损失函数通常是正则化的极大似然函数。决策树的学习目标是一损失函数为目标函数的最小化。

### 9.2 特征选择

通常随着划分过程的进行，我们希望决策树的分支结点所包含的样本尽可能属于同一类，即结点的“纯度”(purity) 越来越高。

常用的选择准则有：**信息增益、信息增益率、基尼系数**

 “信息增益”(information gain)表示得知X的信息而使Y的信息不确定性减少的程度。
信息增益表示由于特征a而使得对数据集D的分类不确定性减少的程度。信息增益越大则意味着使用属性a来进行划分所获得的纯度提升越大。

“信息增益率”(information gain ratio)Grainratio(D,a)定义为其信息增益Grain(D,a)与训练数据集D关于特征a的熵IV(a)之比。

“基尼系数”(Gini index)反映了从数据集D随机抽取两个样本，其类别标记不一致的概率，基尼系数越小，则数据集D纯度越高。

### 9.3 独热编码

两个以上的编码就选用onehot

### 9.4 决策树生成

![image.png](https://cdn.nlark.com/yuque/0/2022/png/12563972/1656351066169-a1c5cecc-0aa0-4345-a73d-c882e08215e0.png#clientId=ubf71af2a-0695-4&crop=0&crop=0&crop=1&crop=1&from=paste&height=360&id=u9ed5d274&name=image.png&originHeight=562&originWidth=800&originalType=binary&ratio=1&rotation=0&showTitle=false&size=196710&status=done&style=none&taskId=u42662646-5712-4718-bf99-f478d1059e6&title=&width=512)<br />显然决策树生成是一种递归算法，而划分的关键是第8行，即如何选择划分属性。

有三种情况会导致递归返回：

1. 当前结点包含的样本全属于同一类别。
1. 当前特征（属性）集为空，或是所有样本在所有属性上取值相同，无法划分。此时我们标记当前结点为叶结点，并将其类别设定为该结点所含样本最多的类别。
1. 当前节点包含的样本集合为空，不能划分。此时把当前结点标记为叶结点，并将其类别设定为其父节点所含样本最多的类别。

### 9.5 替换程序
放回再抽样

### 9.6 随机森林

### 9.7 XGBoost(eXtreme Gradient Boosting)

### 9.10 什么时候选用决策树

**decison trees and tree ensembles**
1. works well on tabular (structured) data
2. not recommended for unstructured data(images,audio,text)
3. fast
4. small decision trees may be human interpretable

**neural networks**
1. works well on all types of data, including tabular (structured) and unstructured data
2. may be slower than a decision tree
3. works with transfer learning 
4. when building a system of multiple models working together, it might be easier to string together multiple neural networks

# 10 无监督学习

1. unsupervised learning 无监督学习
	1. clustering 聚类
	2. anomaly detection 异常检测 
2. recommender systems 推荐系统
3. reinforcement learning 强化学习

## 10.1 k-means聚类

K-均值是最普及的聚类算法，算法接受一个未标记的数据集，然后将数据聚类成不同的组。K-均值是一个迭代算法，假设我们想要将数据聚类成n个组，其方法为:

1. 首先随机选择$K$个随机的点，称为聚类中心（cluster centroids）；

2. 对于数据集中的每一个数据，按照距离$K$个中心点的距离，将其与距离最近的中心点关联起来，与同一个中心点关联的所有点聚成一类。

3. 计算每一个组的平均值，将该组所关联的中心点移动到平均值的位置。
 
4. 重复步骤2-4直至中心点不再变化。 

```
Repeat {

for i = 1 to m

c(i) := index (form 1 to K) of cluster centroid closest to x(i)

for k = 1 to K

μk := average (mean) of points assigned to cluster k

}
```

## 10.2 优化目标

K-均值最小化问题，是要最小化所有的数据点与其所关联的聚类中心点之间的距离之和，因此 K-均值的代价函数（又称畸变函数 Distortion function）为：

$J(c^{(1)},...,c^{(m)},μ_1,...,μ_K)=\dfrac {1}{m}\sum^{m}{i=1}\left| X^{\left( i\right) }-\mu{c^{(i)}}\right| ^{2}$

回顾刚才给出的: K-均值迭代算法，我们知道，第一个循环是用于减小ci引起的代价，而第二个循环则是用于减小mu_i引起的代价。迭代的过程一定会是每一次迭代都在减小代价函数，不然便是出现了错误。

## 10.3 随机初始化

在运行K-均值算法的之前，我们首先要随机初始化所有的聚类中心点，下面介绍怎样做：

1. 我们应该选择K<m，即聚类中心点的个数要小于所有训练集实例的数量

2. 随机选择K个训练实例，然后令K个聚类中心分别与这K个训练实例相等

K-均值的一个问题在于，它有可能会停留在一个局部最小值处，而这取决于初始化的情况。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/d4d2c3edbdd8915f4e9d254d2a47d9c7.png)

为了解决这个问题，我们通常需要多次运行K-means算法，每一次都重新进行随机初始化，最后再比较多次运行K-means的结果，选择代价函数最小的结果。这种方法在K较小的时候（2--10）还是可行的，但是如果K较大，这么做也可能不会有明显地改善。

## 10.4 选择聚类数量

没有所谓最好的选择聚类数的方法，通常是需要根据不同的问题，人工进行选择的。选择的时候思考我们运用K-均值算法聚类的动机是什么，然后选择能最好服务于该目的标聚类数。

当人们在讨论，选择聚类数目的方法时，有一个可能会谈及的方法叫作“肘部法则”。关于“肘部法则”，我们所需要做的是改变K值，也就是聚类类别数目的总数。我们用一个聚类来运行K均值聚类方法。这就意味着，所有的数据都会分到一个聚类里，然后计算成本函数或者计算畸变函数J。K代表聚类数字。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/f3ddc6d751cab7aba7a6f8f44794e975.png)

们可能会得到一条类似于这样的曲线。像一个人的肘部。这就是“肘部法则”所做的，让我们来看这样一个图，看起来就好像有一个很清楚的肘在那儿。好像人的手臂，如果你伸出你的胳膊，那么这就是你的肩关节、肘关节、手。这就是“肘部法则”。你会发现这种模式，它的畸变值会迅速下降，从1到2，从2到3之后，你会在3的时候达到一个肘点。在此之后，畸变值就下降的非常慢，看起来就像使用3个聚类来进行聚类是正确的，这是因为那个点是曲线的肘点，畸变值下降得很快，K=3之后就下降得很慢，那么我们就选K=3。当你应用“肘部法则”的时候，如果你得到了一个像上面这样的图，那么这将是一种用来选择聚类个数的合理方法。

例如，我们的 T-恤制造例子中，我们要将用户按照身材聚类，我们可以分成3个尺寸S,M,L，也可以分成5个尺寸XS,S,M,L,XL，这样的选择是建立在回答“聚类后我们制造的T-恤是否能较好地适合我们的客户”这个问题的基础上作出的。

## 11 异常检测
### 11.1 什么是异常检测

什么是异常检测呢？为了解释这个概念，让我举一个例子吧：

假想你是一个飞机引擎制造商，当你生产的飞机引擎从生产线上流出时，你需要进行QA(质量控制测试)，而作为这个测试的一部分，你测量了飞机引擎的一些特征变量，比如引擎运转时产生的热量，或者引擎的振动等等。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/93d6dfe7e5cb8a46923c178171889747.png)

这样一来，你就有了一个数据集，从x1到xm，如果你生产了m个引擎的话，你将这些数据绘制成图表，看起来就是这个样子：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/fe4472adbf6ddd9d9b51d698cc750b68.png)

这里的每个点、每个叉，都是你的无标签数据。这样，异常检测问题可以定义如下：我们假设后来有一天，你有一个新的飞机引擎从生产线上流出，而你的新飞机引擎有特征变量x_test。所谓的异常检测问题就是：我们希望知道这个新的飞机引擎是否有某种异常，或者说，我们希望判断这个引擎是否需要进一步测试。因为，如果它看起来像一个正常的引擎，那么我们可以直接将它运送到客户那里，而不需要进一步的测试。

给定数据集 x1,x2,xm，我们假使数据集是正常的，我们希望知道新的数据 
x_test是不是异常的，即这个测试数据不属于该组数据的几率如何。我们所构建的模型应该能根据该测试数据的位置告诉我们其属于一组数据的可能性 p(x)。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/65afdea865d50cba12d4f7674d599de5.png)

欺诈检测：

用户的第个活动特征x(i)=用户的第i个活动特征

模型p(x)为我们其属于一组数据的可能性，通过p(x) < \varepsilon检测非正常用户。

异常检测主要用来识别欺骗。例如在线采集而来的有关用户的数据，一个特征向量中可能会包含如：用户多久登录一次，访问过的页面，在论坛发布的帖子数量，甚至是打字速度等。尝试根据这些特征构建一个模型，可以用这个模型来识别那些不符合该模式的用户。

再一个例子是检测一个数据中心，特征可能包含：内存使用情况，被访问的磁盘数量，CPU的负载，网络的通信量等。根据这些特征可以构建一个模型，用来判断某些计算机是不是有可能出错了。

### 11.2 高斯分布

通常如果我们认为变量 x 符合高斯分布
$ x \sim N(\mu, \sigma^2) $

则其概率密度函数为：
$p(x,\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

我们可以利用已有的数据来预测总体中的μ和σ^2的计算方法如下：
$\mu=\frac{1}{m}\sum\limits_{i=1}^{m}x^{(i)}$

$\sigma^2=\frac{1}{m}\sum\limits_{i=1}^{m}(x^{(i)}-\mu)^2$

高斯分布样例：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/fcb35433507a56631dde2b4e543743ee.png)

注：机器学习中对于方差我们通常只除以m而非统计学中的(m-1)。这里顺便提一下，在实际使用中，到底是选择使用1/m还是1/(m-1)其实区别很小，只要你有一个还算大的训练集，在机器学习领域大部分人更习惯使用1/m这个版本的公式。这两个版本的公式在理论特性和数学特性上稍有不同，但是在实际使用中，他们的区别甚小，几乎可以忽略不计。

### 11.3 算法
下图是一个由两个特征的训练集，以及特征的分布情况：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/ba47767a11ba39a23898b9f1a5a57cc5.png)

下面的三维图表表示的是密度估计函数，z轴为根据两个特征的值所估计p(x)值：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/82b90f56570c05966da116c3afe6fc91.jpg)

我们选择一个\varepsilon，将p(x) = \varepsilon作为我们的判定边界，当p(x) > \varepsilon时预测数据为正常数据，否则为异常。

在这段视频中，我们介绍了如何拟合p(x)，也就是 x的概率值，以开发出一种异常检测算法。同时，在这节课中，我们也给出了通过给出的数据集拟合参数，进行参数估计，得到参数 μ 和 σ，然后检测新的样本，确定新样本是否是异常。

### 11.4 开发和评价一个异常检测系统 
异常检测算法是一个非监督学习算法，意味着我们无法根据结果变量  y 的值来告诉我们数据是否真的是异常的。我们需要另一种方法来帮助检验算法是否有效。当我们开发一个异常检测系统时，我们从带标记（异常或正常）的数据着手，我们从其中选择一部分正常数据用于构建训练集，然后用剩下的正常数据和异常数据混合的数据构成交叉检验集和测试集。

例如：我们有10000台正常引擎的数据，有20台异常引擎的数据。 我们这样分配数据：

第一种分配方法
训练集：6000台正常引擎的数据
交叉检验集：2000台正常引擎(y = 0)和10台异常引擎(y = 1)的数据
测试集：2000台正常引擎(y = 0)和10台异常引擎(y = 1)的数据

第二种分配方法
训练集：6000台正常引擎的数据
交叉检验集：4000台正常引擎(y = 0)和10台异常引擎(y = 1)的数据
没有测试集

具体的评价方法如下：

根据测试集数据，我们估计特征的平均值和方差并构建p(x)函数

对交叉检验集，我们尝试使用不同的\varepsilon值作为阀值，并预测数据是否异常，根据F1值或者查准率与查全率的比例来选择 ε选出 ε 后，针对测试集进行预测，计算异常检验系统的F1值，或者查准率与查全率之比。

### 11.5 异常检测和监督学习的对比

之前我们构建的异常检测系统也使用了带标记的数据，与监督学习有些相似，下面的对比有助于选择采用监督学习还是异常检测：

|  异常检测   | 监督学习  |
|  ----  | ----  |
| 非常少量的正向类（异常数据 y=1）, 大量的负向类（y=0）  | 同时有大量的正向类和负向类 |
| 许多不同种类的异常，非常难。根据非常 少量的正向类数据来训练算法。  | 有足够多的正向类实例，足够用于训练 算法，未来遇到的正向类实例可能与训练集中的非常近似。 |
| 许多不同种类的异常，非常难。根据非常 少量的正向类数据来训练算法。  | 有足够多的正向类实例，足够用于训练 算法，未来遇到的正向类实例可能与训练集中的非常近似。 |
| 未来遇到的异常可能与已掌握的异常、非常的不同。  |  |
| 例如： 欺诈行为检测 生产（例如飞机引擎）检测数据中心的计算机运行状况  | 例如：邮件过滤器 天气预报 肿瘤分类 |

### 11.6 选择特征

异常检测假设特征符合高斯分布，如果数据的分布不是高斯分布，异常检测算法也能够工作，但是最好还是将数据转换成高斯分布，例如使用对数函数：x= log(x+c)，其中 c为非负常数； 或者 x=x^c，c为 0-1 之间的一个分数，等方法。(编者注：在python中，通常用np.log1p()函数，log1p就是 log(x+1)，可以避免出现负数结果，反向函数就是np.expm1())

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/0990d6b7a5ab3c0036f42083fe2718c6.jpg)

误差分析：

一个常见的问题是一些异常的数据可能也会有较高的$p(x)$值，因而被算法认为是正常的。这种情况下误差分析能够帮助我们，我们可以分析那些被算法错误预测为正常的数据，观察能否找出一些问题。我们可能能从问题中发现我们需要增加一些新的特征，增加这些新特征后获得的新算法能够帮助我们更好地进行异常检测。

异常检测误差分析：

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/f406bc738e5e032be79e52b6facfa48e.png)

我们通常可以通过将一些相关的特征进行组合，来获得一些新的更好的特征（异常数据的该特征值异常地大或小），例如，在检测数据中心的计算机状况的例子中，我们可以用CPU负载与网络通信量的比例作为一个新的特征，如果该值异常地大，便有可能意味着该服务器是陷入了一些问题中。

在这段视频中，我们介绍了如何选择特征，以及对特征进行一些小小的转换，让数据更像正态分布，然后再把数据输入异常检测算法。同时也介绍了建立特征时，进行的误差分析方法，来捕捉各种异常的可能。希望你通过这些方法，能够了解如何选择好的特征变量，从而帮助你的异常检测算法，捕捉到各种不同的异常情况。

## 12 推荐系统

### 12.1 基于内容的推荐系统

在我们的例子中，我们可以假设每部电影都有两个特征，如x_1代表电影的浪漫程度，x_2 代表电影的动作程度。

![](https://github.com/fengdu78/Coursera-ML-AndrewNg-Notes/raw/master/images/747c1fd6bff694c6034da1911aa3314b.png)

则每部电影都有一个特征向量，如x^{(1)}是第一部电影的特征向量为[0.9 0]。

下面我们要基于这些特征来构建一个推荐系统算法。 假设我们采用线性回归模型，我们可以针对每一个用户都训练一个线性回归模型，如{{\theta }^{(1)}}是第一个用户的模型的参数。 于是，我们有：θ(j)用户 j的参数向量x(i)电影 i 的特征向量对于用户 j 和电影 i，我们预测评分为：(\theta^{(j)})^T x^{(i)}

代价函数，针对用户 j，该线性回归模型的代价为预测误差的平方和，加上正则化项：
$ \min_{\theta (j)}\frac{1}{2}\sum_{i:r(i,j)=1}\left((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\right)^2+\frac{\lambda}{2}\left(\theta_{k}^{(j)}\right)^2 $

为了学习所有用户，我们将所有用户的代价函数求和，如果我们要用梯度下降法来求解最优解，我们计算代价函数的偏导数后得到梯度下降的更新公式为：

$\theta_k^{(j)}:=\theta_k^{(j)}-\alpha\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)} \quad (\text{for} , k = 0)$

$\theta_k^{(j)}:=\theta_k^{(j)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})x_{k}^{(i)}+\lambda\theta_k^{(j)}\right) \quad (\text{for} , k\neq 0)$

### 12.2 协同过滤

在之前的基于内容的推荐系统中，对于每一部电影，我们都掌握了可用的特征，使用这些特征训练出了每一个用户的参数。相反地，如果我们拥有用户的参数，我们可以学习得出电影的特征。但是如果我们既没有用户的参数，也没有电影的特征，这两种方法都不可行了。协同过滤算法可以同时学习这两者。我们的优化目标便改为同时针对x和\theta进行。

$$ J(x^{(1)},...x^{(n_m)},\theta^{(1)},...,\theta^{(n_u)})=\frac{1}{2}\sum_{(i:j):r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2+\frac{\lambda}{2}\sum_{i=1}^{n_m}\sum_{k=1}^{n}(x_k^{(j)})^2+\frac{\lambda}{2}\sum_{j=1}^{n_u}\sum_{k=1}^{n}(\theta_k^{(j)})^2 $$

对代价函数求偏导数的结果如下：

$x_k^{(i)}:=x_k^{(i)}-\alpha\left(\sum_{j:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\theta_k^{j}+\lambda x_k^{(i)}\right)$

$\theta_k^{(i)}:=\theta_k^{(i)}-\alpha\left(\sum_{i:r(i,j)=1}((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}x_k^{(i)}+\lambda \theta_k^{(j)}\right)$

注：在协同过滤从算法中，我们通常不使用方差项，如果需要的话，算法会自动学得。 协同过滤算法使用步骤如下：

1. 初始 x(1),x(1),...x(nm),θ(1),θ(2),...θ(nu)为一些随机小值使用梯度下降算法最小化代价函数在训练完算法后，我们预测(\theta^{(j)})^Tx^{(i)}为用户 j 给电影 i 的评分。

## 13 强化学习