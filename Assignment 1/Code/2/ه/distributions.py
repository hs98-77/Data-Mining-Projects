# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 20:57:18 2022

@author: hosei
"""
import numpy as np 
import pylab 
import scipy.stats as stats
A = [11.7, 75.3, 6.21, 265.6, 0.31, 43.02, 111.87, 145, 121.2, 95, 111, 42.5, 0.04, 72, 58.1, 659.98, 110.02, 0.78, 598.23, 311.54]
B = [17.20, 12.90, 0.53, 34.80, 4.20, 1.50, 3.90, 6.70, 7.43, 8.60, 11.30, 12.10, 5.6, 13.20, 15.80, 19.80, 23.90, 34.17, 2.30, 7.80]
#%%
stats.probplot(np.array(A), dist=stats.expon, plot=pylab)
#%%
stats.probplot(np.array(B), dist=stats.expon, plot=pylab)
pylab.show()
#%%
