#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 15:08:51 2021

@author: thosvarley
"""
import numpy as np 
from opyn import embed_series
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

X = np.random.randn(1000)
X = np.cumsum(X)

#%%

from opyn import false_nearest_neighbors

print(false_nearest_neighbors(X, 0.35, 1, metric="euclidean"))