#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Logistic Regression"""

import numpy as np
import matplotlib.pyplot as plt

class LR(object):

    def __init__(self):
        self.alpha=0.1
        self.maxTime=200

    def fit(self,X,Y):
        w = np.random.uniform(size=(X.shape[0],1))
        b = np.random.uniform()

        #学习步长
        alpha=self.alpha
        index=0
        while index<self.maxTime:

            Z=np.dot(w.T,X)+b
            A=self.sigmoid(Z)
            dz=A-Y
            dw=np.dot(X,dz.T)/X.shape[1]
            db=np.sum(dz)/X.shape[1]
            w-=alpha*dw
            b-=alpha*db
            index+=1
        self.w=w
        self.b=b



    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def transform(self,x):
        return sigmoid(np.dot(self.w.T,x)+b)

if __name__=='__main__':
    x1=np.random.normal(4,1,100)
    x2=np.random.normal(2,1,100)
    X1=np.row_stack((x1,x2))
    x1=np.random.normal(2,1,100)
    x2=np.random.normal(4,1,100)
    X2=np.row_stack((x1,x2))
    X=np.column_stack((X1,X2))
    Y=np.zeros((1,200))
    Y[0,100:200]+=1
    lr=LR()
    lr.fit(X,Y)
    print(lr.w)
    print(lr.b)
