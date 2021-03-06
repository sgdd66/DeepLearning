# 深度学习
这个库用于存储在学习*吴恩达老师的《深度学习》*中编写的代码，不定期更新
## logistics regression
文件名Logistics.py。测试数据有两个特征，方便展示。使用高斯分布产生两类数据，存在一定程度的重合。具体算法参考吴恩达老师的教程
接口与scikit-learn库中的使用方法相似，应用fit学习样本，transform预测数据
## 单隐层神经网络
文件名SHLNN.py(Single Hiden Layer Neural Network)。训练方法采用累积修正。接口与scikit-learn库中的使用方法相似，应用fit学习样本，transform预测数据
对于调参过程详细说明一下：
### 测试样例
首先说一下自己的测试样例。我的测试样例有两组，都是二分类问题。第一组，由中心在（1,1）和（2,2），方差都为0.4的正态分布构成，每个分布都有100个采样点，中心在（1,1)的采样点标记为0，中心在（2,2）标记为1。很明显这是一个简单的二分类问题。第二组有三个分布，中心在（1,1），（2,2），（3,3）。其中中心在（1,1）和（3,3）的采样点各有50个，标记为0。中心在（2,2）的采样点有100个，标记为1。这是一个非线性分类问题
### 程序功能
然后说明一下我的程序实现的功能。这个程序只针对单隐层神经网络，但是各层神经结点数据可变，且每层结点的激活函数可选。目前实现了sigmoid函数，tanh函数，ReLU函数和leaky ReLU函数。
### 可调参数
那么对于这样的测试环境与样例，可以调整的参数有，隐藏层神经节点数目，每次改进的步进常数，目标准确率，最大迭代次数，隐藏层的激活函数。
>对于隐藏层的神经节数目，刚开始预估样本空间的复杂程度，设置尽可能少的数目。例如对于一个线性可分的问题，两个就可以了。至于再复杂的可以安排8个或者以上。

>建议将步进常数设定为0.01或者更小，防止参数不收敛。

>目标准确率一开始可以设定为0.9

>迭代次数可以设置为1000或者更高，这个要结合样本量具体考量。总之在可以接受的时间范围内，将迭代次数设置的大一点。

>我个人建议先使用tanh函数作为隐藏层的激活函数。虽然ReLU函数收敛快，但如果步进常数或者隐藏层结点数目一开始设置不好，很容易产生震荡，无法收敛。相反tanh函数就稳定一点。

### 调整顺序
在调参中，我们先调整步进常数。就是观察成本函数下降的速率，修正步进常数，一方面快速下降，另一方面防止不收敛。调整合适之后再调整隐藏层结点数目。逐渐增加，准确率理论上应该是先增大，后减小。找到合适的结点数目。最后逐步调高目标准确率。若一切稳定可以将激活函数换为ReLU函数在细调一下。

## 多层神经网络
对应文件DNN.py。在SHLNN.py的基础上，拓展了隐藏层的数目。具体的使用方法可以参考程序中的执行代码

## 正则化，归一化，参数初始化
对应文件Regularization.py。在DNN.py的基础上添加正则化，归一化以及参数初始化。该部分内容对应《深度学习》课程第二周第一部分。注意这部分内容没有实现梯度检验，因为个人认为这部分是在编写神经网络程序中使用的检验技术

## mini batch算法
这个算法也写在Regularization.py中，函数fit_miniBatch()就是这个方法的实现

## 动量梯度下降法
这个算法对应Regularization.py中的fit_momentum()函数

## Adam optimization algorithm
算法对应Regularization.py中的fit_Adam()函数

## Batch Normalize
Batch Normalize算法于2015年提出，通过对每一层白化，使得前几层的变化不会影响本层的数据分布，由此加速神经网络的收敛。代码位于BatchNormal.py中

## SoftMax
为使神经网络模型应对多分类任务，将最后一层的激活函数修改为softmax函数。程序在SoftMax.py