#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""在DNN的基础上添加正则化,参数初始化
该部分内容对应《深度学习》课程第二周第一部分
注意这部分内容没有实现梯度检验，因为个人认为这部分是在编写神经网络程序中使用的检验技术"""

import numpy as np
import matplotlib.pyplot as plt

class DNN(object):
    """多层神经网络"""
    def __init__(self,L):
        """L：描述每层神经结点的数目。
        L[0]:表示输入层，也就是样本的维度
        L[n]:表示第n层的神经结点数目"""
        self.L=L
        N=L.shape[0]

        #链接系数
        self.W=[None]*N

        # 各层阈值
        self.B =[None]*N

        for i in range(1,N):
            self.W[i]=np.random.uniform(size=(L[i],L[i-1]))*np.sqrt(2/(L[i-1]))
            self.B[i]=np.zeros((L[i],1))

        self.actFun=np.zeros(N)+2
        self.actFun[N-1]=1
        self.actFun[0]=0
        self.lambd=0.001
        # 开关1：用于确定是否启用正则化
        self.switch1=True
        # 开关2：用于确定是否启用mini_batch
        self.switch2=True
        #由于momentum梯度下降方法
        self.beta=0.9
        #用于Adam方法
        self.beta1=0.9
        self.beta2=0.99

    def fit_miniBatch(self,X,Y):
        """学习样本，学习算法是累积BP算法加mini_batch
         X,样本特征n0*m，n0是样本维度，m是样本数目，与输入层神经元数目对应
         Y,样本对应的标签n2*m，m是样本数目，n2与输出层神经元数目对应，对于二分类问题n2=1"""

        X_batch, Y_batch = self.mini_batch(X, Y)

        # 修正步长
        eta = 0.5

        N = self.L.shape[0]
        maxGen = 2000
        ratio = np.zeros(maxGen + 1)
        gen = 0
        maxRatio = 0.95

        Z = [None] * (N)
        A = [None] * (N)
        dZ = [None] * (N)
        dW = [None] * (N)
        dB = [None] * (N)

        m = X.shape[1]

        while gen < maxGen:
            index=gen%len(X_batch)
            m = X_batch[index].shape[1]
            A[0] = X_batch[index]
            gen += 1
            for i in range(1, N):
                Z[i] = np.dot(self.W[i], A[i - 1]) + self.B[i]
                A[i] = self.Activation(Z[i], self.actFun[i])

            J = -np.sum(Y * np.log(A[N - 1]) + (1 - Y) * np.log(1 - A[N - 1])) / m

            out = A[N - 1] // 0.5
            out[out == 2] = 1
            ratio[gen] = (m - np.sum(np.abs(out - Y))) / m
            if ratio[gen] > maxRatio:
                break

            dZ[N - 1] = A[N - 1] - Y
            for i in range(N - 1, 0, -1):
                if self.switch1:
                    dW[i] = np.dot(dZ[i], A[i - 1].T) / m + self.lambd / m * self.W[i]
                else:
                    dW[i] = np.dot(dZ[i], A[i - 1].T) / m
                dB[i] = np.sum(dZ[i], axis=1, keepdims=True) / m
                self.W[i] -= eta * dW[i]
                self.B[i] -= eta * dB[i]
                if i > 1:
                    dZ[i - 1] = np.dot(self.W[i].T, dZ[i]) * self.derivative(Z[i - 1], self.actFun[i - 1])
        np.set_printoptions(threshold=np.inf)
        print(ratio)


    def fit(self,X,Y):
        """学习样本，学习算法是累积BP算法
        X,样本特征n0*m，n0是样本维度，m是样本数目，与输入层神经元数目对应
        Y,样本对应的标签n2*m，m是样本数目，n2与输出层神经元数目对应，对于二分类问题n2=1"""

        # 修正步长
        eta = 0.5
        m=X.shape[1]
        N=self.L.shape[0]
        maxGen=2000
        ratio=np.zeros(maxGen+1)
        gen=0
        maxRatio=0.95

        Z=[None]*(N)
        A=[None]*(N)
        dZ=[None]*(N)
        dW=[None]*(N)
        dB=[None]*(N)

        A[0]=X
        while gen<maxGen:
            gen+=1
            for i in range(1,N):
                Z[i]=np.dot(self.W[i],A[i-1])+self.B[i]
                A[i]=self.Activation(Z[i],self.actFun[i])

            J=-np.sum(Y*np.log(A[N-1])+(1-Y)*np.log(1-A[N-1]))/m

            out=A[N-1]//0.5
            out[out==2]=1
            ratio[gen]=(m-np.sum(np.abs(out-Y)))/m
            if ratio[gen]>maxRatio:
                break

            dZ[N-1]=A[N-1]-Y
            for i in range(N-1,0,-1):
                if self.switch1:
                    dW[i]=np.dot(dZ[i],A[i-1].T)/m+self.lambd/m*self.W[i]
                else:
                    dW[i] = np.dot(dZ[i], A[i - 1].T) / m
                dB[i]=np.sum(dZ[i],axis=1,keepdims=True)/m
                self.W[i]-=eta*dW[i]
                self.B[i]-=eta*dB[i]
                if i>1:
                    dZ[i-1]=np.dot(self.W[i].T,dZ[i])*self.derivative(Z[i-1],self.actFun[i-1])
        np.set_printoptions(threshold=np.inf)
        print(ratio)

    def fit_momentum(self,X,Y):
        """学习样本，学习算法是动量梯度下降法
        X,样本特征n0*m，n0是样本维度，m是样本数目，与输入层神经元数目对应
        Y,样本对应的标签n2*m，m是样本数目，n2与输出层神经元数目对应，对于二分类问题n2=1"""

        # 修正步长
        eta = 0.5
        m=X.shape[1]
        N=self.L.shape[0]
        maxGen=2000
        ratio=np.zeros(maxGen+1)
        gen=0
        maxRatio=0.95

        Z=[None]*(N)
        A=[None]*(N)
        dZ=[None]*(N)
        dW=[None]*(N)
        dB=[None]*(N)

        A[0]=X
        while gen<maxGen:
            gen+=1
            for i in range(1,N):
                Z[i]=np.dot(self.W[i],A[i-1])+self.B[i]
                A[i]=self.Activation(Z[i],self.actFun[i])

            J=-np.sum(Y*np.log(A[N-1])+(1-Y)*np.log(1-A[N-1]))/m

            out=A[N-1]//0.5
            out[out==2]=1
            ratio[gen]=(m-np.sum(np.abs(out-Y)))/m
            if ratio[gen]>maxRatio:
                break

            dZ[N-1]=A[N-1]-Y
            for i in range(N-1,0,-1):
                if self.switch1:
                    dw=np.dot(dZ[i],A[i-1].T)/m+self.lambd/m*self.W[i]
                else:
                    dw = np.dot(dZ[i], A[i - 1].T) / m
                db=np.sum(dZ[i],axis=1,keepdims=True)/m
                if(dW[i] is None):
                    dW[i]=(1-self.beta)*dw
                else:
                    dW[i]=self.beta*dW[i]+(1-self.beta)*dw
                if dB[i] is None:
                    dB[i]=(1-self.beta)*db
                else:
                    dB[i]=self.beta*dB[i]+(1-self.beta)*db
                self.W[i]-=eta*dW[i]
                self.B[i]-=eta*dB[i]
                if i>1:
                    dZ[i-1]=np.dot(self.W[i].T,dZ[i])*self.derivative(Z[i-1],self.actFun[i-1])
        np.set_printoptions(threshold=np.inf)
        print(ratio)

    def fit_Adam(self,X,Y):
        """学习样本，学习算法是动量梯度下降法
        X,样本特征n0*m，n0是样本维度，m是样本数目，与输入层神经元数目对应
        Y,样本对应的标签n2*m，m是样本数目，n2与输出层神经元数目对应，对于二分类问题n2=1"""

        # 修正步长
        eta = 0.005
        m=X.shape[1]
        N=self.L.shape[0]
        maxGen=2000
        ratio=np.zeros(maxGen+1)
        gen=0
        maxRatio=0.95

        Z=[None]*(N)
        A=[None]*(N)
        dZ=[None]*(N)
        V_dw=[None]*(N)
        V_db=[None]*(N)
        S_dw=[None]*(N)
        S_db=[None]*(N)


        A[0]=X
        while gen<maxGen:
            gen+=1
            for i in range(1,N):
                Z[i]=np.dot(self.W[i],A[i-1])+self.B[i]
                A[i]=self.Activation(Z[i],self.actFun[i])

            J=-np.sum(Y*np.log(A[N-1])+(1-Y)*np.log(1-A[N-1]))/m

            out=A[N-1]//0.5
            out[out==2]=1
            ratio[gen]=(m-np.sum(np.abs(out-Y)))/m
            if ratio[gen]>maxRatio:
                break

            dZ[N-1]=A[N-1]-Y
            for i in range(N-1,0,-1):
                if self.switch1:
                    dw=np.dot(dZ[i],A[i-1].T)/m+self.lambd/m*self.W[i]
                else:
                    dw = np.dot(dZ[i], A[i - 1].T) / m
                db=np.sum(dZ[i],axis=1,keepdims=True)/m

                if(V_dw[i] is None):
                    V_dw[i]=(1-self.beta1)*dw
                    V_db[i]=(1-self.beta1)*db
                    S_dw[i]=(1-self.beta2)*dw**2
                    S_db[i]=(1-self.beta2)*db**2
                else:
                    V_dw[i]=self.beta1*V_dw[i]+(1-self.beta1)*dw
                    V_db[i]=self.beta1*V_db[i]+(1-self.beta1)*db
                    S_dw[i]=self.beta2*S_dw[i]+(1-self.beta2)*dw**2
                    S_db[i]=self.beta2*S_db[i]+(1-self.beta2)*db**2
                v_dw_cor=V_dw[i]/(1-self.beta1**(gen+1))
                v_db_cor=V_db[i]/(1-self.beta1**(gen+1))
                s_dw_cor=S_dw[i]/(1-self.beta2**(gen+1))
                s_db_cor=S_db[i]/(1-self.beta2**(gen+1))

                self.W[i]-=eta*v_dw_cor/(s_dw_cor+10**-8)**0.5
                self.B[i]-=eta*v_db_cor/(s_db_cor+10**-8)**0.5

                if i>1:
                    dZ[i-1]=np.dot(self.W[i].T,dZ[i])*self.derivative(Z[i-1],self.actFun[i-1])
        np.set_printoptions(threshold=np.inf)
        print(ratio)


    def mini_batch(self,X,Y):
        """X默认的样本数目在2000以上"""

        m=X.shape[1]
        dim=X.shape[0]
        n=1024
        batchNum=m//n
        if(n*batchNum!=m):
            batchNum+=1
        batchX=[None]*batchNum
        batchY=[None]*batchNum
        for i in range(0,batchNum-1):
            r=np.random.randint(0,m,n)
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
        for i in range(1,N):
            Z=np.dot(self.W[i],A)+self.B[i]
            A=self.Activation(Z,self.actFun[i])

        return A//0.5

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
    nn.fit_Adam(X,Y)