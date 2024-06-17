# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 22:18:52 2022

@author: hosei
"""
import numpy as np
import pandas as pd

X = np.array([[1.5,1.7],[2,1.9],[1.6,1.8],[1.2,1.5],[1.5,1]])
q = np.array([1.4,1.6])

def Euclidean(x,y):
    return ( (x[0]-y[0])**2 + (x[1]-y[1])**2 )**0.5

def Manhattan(x,y):
    return ( abs(x[0]-y[0]) + abs(x[1]-y[1]) )

def Supremum(x,y):
    return max(abs(x[0]-y[0]),abs(x[1]-y[1]))

def VecSize(x):
    return (x[0]**2 + x[1]**2)**0.5

def Cosine(x,y):
    return np.dot(x,y)/(VecSize(x)*VecSize(y))

def Distance(func,x,y,n=5):
    d = list()
    for i in range(n):
        d.append(func(x[i,:],y))
    return d


funcs = [Euclidean,Manhattan,Supremum,Cosine]
FuncNames = ['Euclidean','Manhattan','Supremum','Cosine']
dist = pd.DataFrame()
for i in range(len(funcs)):
    dist[FuncNames[i]] = Distance(funcs[i], X, q)
del i