#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""这个程序实现了batch-normal方法，辅助训练更加深层的神经网络
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class DNN(object):
    """多层神经网络，使用batch-normal方法训练"""
    def __init__(self,L):
        """L：描述每层神经结点的数目。
        L[0]:表示输入层，也就是样本的维度
        L[n]:表示第n层的神经结点数目"""
        self.L=L
        N=L.shape[0]

        #链接系数
        self.W=[None]*N
        self.gamma=[None]*N
        self.beta=[None]*N


        for i in range(1,N):
            self.W[i]=np.random.uniform(size=(L[i],L[i-1]))*np.sqrt(2/(L[i-1]))
            self.gamma[i]=np.random.uniform(0,1,(L[i],1))
            self.beta[i]=np.zeros((L[i],1))
        self.gamma[0]=np.zeros((L[0],1))+1      #np.random.uniform(0,1,(L[0],1))
        self.beta[0]=np.zeros((L[0],1))
        self.B=np.zeros((L[N-1],1))


        self.actFun=np.zeros(N)+2
        self.actFun[N-1]=1
        self.actFun[0]=0
        # 正则化参数
        self.lambd=0.001
        # 开关1：用于确定是否启用正则化
        self.switch1=True
        # 开关2：用于确定是否启用mini_batch
        self.switch2=True

        #用于Adam方法
        self.beta1=0.9
        self.beta2=0.99
        self.epsilon=10**-8

    def fit_miniBatch(self,X,Y):
        """学习样本，学习算法是累积BP算法加mini_batch
         X,样本特征n0*m，n0是样本维度，m是样本数目，与输入层神经元数目对应
         Y,样本对应的标签n2*m，m是样本数目，n2与输出层神经元数目对应，对于二分类问题n2=1"""

        X_batch, Y_batch = self.mini_batch(X, Y)

        # 修正步长
        eta = 0.1

        N = self.L.shape[0]
        maxGen = 1000
        ratio = np.zeros(maxGen + 1)
        gen = 0
        maxRatio = 0.95

        Z   = [None]*N
        A   = [None]*(N+1)
        Z1  = [None]*N
        Z2  = [None]*N
        dZ  = [None]*N
        dA  = [None]*N
        dZ1 = [None]*N
        dZ2 = [None]*N
        dW  = [None]*N
        dMu = [None]*N
        mu  = [None]*N
        var = [None]*N
        dGamma  =[None]*N
        dBeta   =[None]*N
        dVar    =[None]*N

        beta1=0.9
        beta2=0.9


        while gen < maxGen:
            index=gen%len(X_batch)
            m = X_batch[index].shape[1]
            gen += 1
            for i in range(0, N-1):
                if i==0:
                    Z[i]=X_batch[index]
                else:
                    Z[i] = np.dot(self.W[i], A[i-1])

                if mu[i] is None:
                    mu[i]=np.mean(Z[i],axis=1,keepdims=True)
                    var[i]=np.var(Z[i],axis=1,keepdims=True)
                else:
                    mu[i]=mu[i]*beta1+(1-beta1)*np.mean(Z[i],axis=1,keepdims=True)
                    var[i]=var[i]*beta2+(1-beta2)*np.var(Z[i],axis=1,keepdims=True)


                Z1[i]=(Z[i]-mu[i])/np.sqrt(var[i]+self.epsilon)
                Z2[i]=self.gamma[i]*Z1[i]+self.beta[i]
                A[i] = self.Activation(Z2[i], self.actFun[i])

            Z[N-1]= np.dot(self.W[N-1], A[N-2])+self.B
            A[N-1]= self.Activation(Z[N-1],self.actFun[N-1])

            J = -np.sum(Y_batch[index] * np.log(A[N - 1]) + (1 - Y_batch[index]) * np.log(1 - A[N - 1])) / m

            out = A[N - 1] // 0.5
            out[out == 2] = 1
            ratio[gen] = (m - np.sum(np.abs(out - Y_batch[index]))) / m
            if ratio[gen] > maxRatio:
                break

            dZ[N - 1] = A[N - 1] - Y_batch[index]
            dW[N-1]=np.dot(dZ[N-1], A[N-2].T) / m
            dB=np.sum(dZ[N-1], axis=1, keepdims=True) / m
            self.B -= eta * dB
            for i in range(N - 2, 0, -1):
                dA[i]=np.dot(self.W[i+1].T,dZ[i+1])
                dZ2[i]=dA[i]*self.derivative(Z2[i], self.actFun[i])
                dGamma[i]=np.sum(dZ2[i]*Z1[i],axis=1,keepdims=True)
                dBeta[i]=np.sum(dZ2[i],axis=1,keepdims=True)
                dZ1[i]=dZ2[i]*self.gamma[i]
                dVar[i]=np.sum(dZ1[i]*(Z[i]-mu[i]),axis=1,keepdims=True)*(-1/2*(var[i]+self.epsilon)**-3/2)
                dMu[i]=np.sum(dZ1[i]*(-1/np.sqrt(var[i]+self.epsilon)))+\
                       dVar[i]*np.sum(-2*(Z[i]-mu[i]),axis=1,keepdims=True)/m
                dZ[i]=dZ1[i]/np.sqrt(var[i]+self.epsilon)+dMu[i]/m+dVar[i]*2*(Z[i]-mu[i])/m
                dW[i] = np.dot(dZ[i], A[i - 1].T) / m

                self.W[i] -= eta * dW[i]
                self.beta[i] -= eta * dBeta[i]
                self.gamma[i] -= eta * dGamma[i]



        self.mu=mu
        self.var=var
        np.set_printoptions(threshold=np.inf)
        print(ratio)

    def mini_batch(self,X,Y):
        """X默认的样本数目在2000以上"""

        m=X.shape[1]
        dim=X.shape[0]
        n=50
        batchNum=m//n
        if(n*batchNum!=m):
            batchNum+=1
        batchX=[]
        batchY=[]
        for i in range(0,batchNum-1):
            array=range(0,m)
            r=random.sample(array,n)

            x=np.zeros((dim,n))
            y=np.zeros((1,n))
            x[:,:]=X[:,r]
            y[:,:]=Y[:,r]
            X=np.delete(X,r,1)
            Y=np.delete(Y,r,1)
            m=X.shape[1]
            batchX.append(x)
            batchY.append(y)

        batchX.append(X)
        batchY.append(Y)
        return batchX,batchY

    def transform(self,x):
        A=x
        N=self.L.shape[0]
        for i in range(0, N - 1):
            if i == 0:
                Z = A
            else:
                Z = np.dot(self.W[i], A)
            Z1 = (Z - self.mu[i]) / np.sqrt(self.var[i] + self.epsilon)
            Z2 = self.gamma[i] * Z1 + self.beta[i]
            A = self.Activation(Z2, self.actFun[i])
        Z=np.dot(self.W[N-1], A)
        A=self.Activation(Z, self.actFun[N-1])

        return A//0.5

    def Activation(self,X,kind=1):
        """激活函数：
        kind=1:sigmoid函数
        kind=2:tanh函数
        kind=3:ReLU函数
        kind=4:leaky ReLU函数"""
        x=np.zeros(X.shape)
        x[:,:]=X[:,:]
        if kind==0:
            return X
        elif kind==1:
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
        if kind==0:
            return np.zeros(X.shape)+1
        elif kind==1:
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
    L=np.array([2,4,2,1])
    nn=DNN(L)
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
    nn.fit_miniBatch(X,Y)