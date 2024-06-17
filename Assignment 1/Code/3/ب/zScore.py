# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 01:49:43 2022

@author: hosei
"""
import numpy as np
#import pandas as pd


def Mean(x):
    return sum(x)/len(x)

def StandardDeviation(x, m):
    t = 0
    for item in x:
        t += ((item - m)**2)
    return (t/x.shape[0])**0.5


def zScore(x,y):
    tempX = np.zeros(x.shape)
    tempY = np.zeros(y.shape)
    for i in range(x.shape[1]):
        m = Mean(x[:,i])
        std = StandardDeviation(x[:,i], m)
        tempY[i] = (y[i]-m)/std
        for j in range(x.shape[0]):
            tempX[j,i] = (x[j,i]-m)/std
    return tempX,tempY

def Euclidean(x,y):
    return ( (x[0]-y[0])**2 + (x[1]-y[1])**2 )**0.5


#%%
X = np.array([[1.5,1.7],[2,1.9],[1.6,1.8],[1.2,1.5],[1.5,1]])
q = np.array([1.4,1.6])
#%%
X,q = zScore(X,q)
EuclideanDistance = list()
for i in range(X.shape[0]):
    EuclideanDistance.append(round(Euclidean(X[i,:], q),3))
del i