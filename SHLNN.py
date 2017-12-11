#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""这个程序用于实现单隐层的BP算法"""

import numpy as np
import matplotlib.pyplot as plt

class SLNN(object):
    """单隐层神经网络"""
    def __init__(self,n0,n1,n2):
        """n2:输出层神经元个数
        n1:隐藏层神经元数目
        n0：输入层神经元数目，也就是样本维度"""
        #隐层-输出层链接矩阵
        self.w1=np.random.uniform(size=(n1,n0))*0.1
        #输入层-隐层链接矩阵
        self.w2=np.random.uniform(size=(n2,n1))*0.1
        #输出层阈值
        self.b2=np.zeros((n2,1))
        #隐层阈值
        self.b1=np.zeros((n1,1))
        #各层的激活函数，输入层不存在激活函数，设定为0
        self.actFun=np.array([0,3,1])


    def fit(self,X,Y):
        """学习样本，学习算法是累积BP算法
        X,样本特征n0*m，n0是样本维度，m是样本数目，与输入层神经元数目对应
        Y,样本对应的标签n2*m，m是样本数目，n2与输出层神经元数目对应，对于二分类问题n2=1"""
        # 修正步长
        eta = 0.5
        m=X.shape[1]
        maxGen=1000
        gen=0
        w1=self.w1
        b1=self.b1
        w2=self.w2
        b2=self.b2
        maxRatio=0.95
        while gen<maxGen:
            gen+=1
            Z1=np.dot(w1,X)+b1
            A1=self.Activation(Z1,self.actFun[1])
            Z2=np.dot(w2,A1)+b2
            A2=self.Activation(Z2,self.actFun[2])
            J=-np.sum(Y*np.log(A2)+(1-Y)*np.log(1-A2))/m

            ratio=A2//0.5
            ratio[ratio==2]=1
            ratio=(m-np.sum(np.abs(ratio-Y)))/m
            if ratio>maxRatio:
                break



            dZ2=A2-Y
            dW2=np.dot(dZ2,A1.T)/m
            db2=np.sum(dZ2,axis=1,keepdims=True)/m
            dZ1=np.dot(w2.T,dZ2)*self.derivative(Z1,kind=self.actFun[1])
            dW1=np.dot(dZ1,X.T)/m
            db1=np.sum(dZ1,axis=1,keepdims=True)/m

            w2-=eta*dW2
            b2-=eta*db2
            w1-=eta*dW1
            b1-=eta*db1
        self.w1=w1
        self.w2=w2
        self.b1=b1
        self.b2=b2
        print(gen,ratio)


    def transform(self,x):
        z1 = np.dot(self.w1, x)
        a1 = self.Activation(z1,self.actFun[1])
        z2 = np.dot(self.w2,a1)
        a2 = self.Activation(z2,self.actFun[2])
        return a2//0.5

    def Activation(self,X,kind=1):
        """激活函数：
        kind=1:sigmoid函数
        kind=2:tanh函数
        kind=3:ReLU函数
        kind=4:leaky ReLU函数"""
        x=np.zeros(X.shape)
        x[:,:]=X[:,:]
        if kind==1:
            return 1/(1+np.exp(-x))
        elif kind==2:
            return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        elif kind==3:
            x[x<0]=0
            return x
        elif kind==4:
            x[x<0]=x[x<0]*0.01
            return np.max(x,axis=1,keepdims=True)

    def derivative(self,X,kind=1):
        """激活函数导数：
        kind=1:sigmoid函数
        kind=2:tanh函数
        kind=3:ReLU函数
        kind=4:leaky ReLU函数"""
        x=np.zeros(X.shape)
        x[:,:]=X[:,:]
        if kind==1:
            a=1/(1+np.exp(-x))
            return a*(1-a)
        elif kind==2:
            a=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
            return 1-a**2
        elif kind==3:
            a=np.zeros(x.shape)
            a[x>=0]=1
            return a
        elif kind==4:
            a=np.zeors(x.shape)
            a[x>=0]=1
            a[x<0]=0.01
            return a



if __name__=='__main__':
    nn=SLNN(2,10,1)
    theta=0.4
    x1=np.random.normal(1,theta,50)
    x2 = np.random.normal(1, theta, 50)
    X1=np.row_stack((x1,x2))

    x1=np.random.normal(3,theta,50)
    x2 = np.random.normal(3, theta, 50)
    X2=np.row_stack((x1,x2))

    X1=np.column_stack((X1,X2))

    x1=np.random.normal(2,theta,100)
    x2 = np.random.normal(2, theta, 100)
    X2=np.row_stack((x1,x2))
    X=np.column_stack((X1,X2))
    Y=np.zeros((1,200))
    Y[0,100:200]+=1
    # plt.scatter(X[0,0:100],X[1,0:100])
    # plt.scatter(X[0,100:200],X[1,100:200])
    # plt.show()
    nn.fit(X,Y)