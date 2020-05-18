#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:26:12 2019

@author: thosvarley
"""
import cython
cimport cython
import numpy as np
cimport numpy as np 
import igraph as ig 
from itertools import product
from scipy.stats import entropy, pearsonr
from libc.math cimport log2
from scipy.signal import argrelextrema

@cython.boundscheck(False)
@cython.initializedcheck(False)
def allowed_transition(a, b):
    
    a_star = a[1:]
    b_star = b[:-1]
    
    phi_a = np.argsort(np.argsort([int(x) for x in a_star]))
    phi_b = np.argsort(np.argsort([int(x) for x in b_star]))
    
    if False not in (phi_a == phi_b):
        return True
    
    else:
        return False

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def perm_embedding(double[:] ts, int dim, int lag):
    
    cdef float[:,:] timestep = np.zeros((ts.shape[0]-lag*(dim-1), dim), dtype="single")
    cdef object[:] series = np.zeros((ts.shape[0]-lag*(dim-1))-1, dtype=object)
    cdef int t, d, s, i, x
    
    for t in range(ts.shape[0]-lag*(dim-1)):
        for d in range(dim):
            timestep[t][d] = ts[t + d*lag]
    
    sort = np.argsort(np.argsort(timestep, axis=1).astype("int32"))
    unique = np.unique(sort, axis=0)
        
    cdef dict arr2string = {unique[i].tobytes() : "".join([str(x) for x in unique[i]]) for i in range(unique.shape[0])}
    
    for s in range(sort.shape[0]-1):
        series[s] = arr2string[sort[s].tobytes()]

    return series

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def OPN(double[:] ts, int dim, int lag):
        
    cdef float[:,:] timestep = np.zeros((ts.shape[0]-lag*(dim-1), dim), dtype="single")
    cdef object[:] series = np.zeros((ts.shape[0]-lag*(dim-1))-1, dtype=object)
    cdef int t, d, s, i, x
    
    for t in range(ts.shape[0]-lag*(dim-1)):
        for d in range(dim):
            timestep[t][d] = ts[t + d*lag]
    
    sort = np.argsort(np.argsort(timestep, axis=1).astype("int32"))
    unique = np.unique(sort, axis=0)
        
    cdef dict arr2string = {unique[i].tobytes() : "".join([str(x) for x in unique[i]]) for i in range(unique.shape[0])}
        
    G = ig.Graph(directed=True)
    G.add_vertices(list(arr2string.values()))
    
    cdef object[:] edges = np.zeros(sort.shape[0]-1, dtype=object)
    cdef int[:] weights = np.ones(sort.shape[0]-1, dtype="int32")
    
    for s in range(sort.shape[0]-1):
        
        edges[s] = (arr2string[sort[s].tobytes()], 
                   arr2string[sort[s+1].tobytes()])
        
        series[s] = arr2string[sort[s].tobytes()]
    
    G.add_edges(edges)
    G.es["weight"] = weights
    G.es["source_target"] = edges
    G = G.simplify(loops = False, combine_edges = dict(weight="sum", source_target="first"))
    
    return G, series 


@cython.initializedcheck(False)
@cython.boundscheck(False)
def optimal_lag(double[:] X, int lrange = 25):
    
    cdef double[:] autocorr = np.ones(lrange-1)
    cdef int i 
    
    for i in range(1,lrange-1):
        autocorr[i] = pearsonr(X[:-i], X[i:])[0]
    
    cdef long[:] extrema = argrelextrema(np.abs(autocorr), np.less)[0]
    
    return extrema[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def optimal_dim(double[:] X, int lag, int drange = 20):
    
    cdef double[:] var = np.zeros(drange)
    cdef int i 
    
    for i in range(2, drange):
        G, s = OPN(X, i, lag)
        var[i] = np.var(G.degree())
        
    return np.argmax(var)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def constrained_walk(G, int lag, int steps):
    
    cdef object[:] walk = np.zeros(steps, dtype="object")
    cdef int i
    
    for i in range(lag):
        walk[i] = str(np.random.choice(G.vs["name"]))
    
    cdef int ptr = lag 
    cdef set vset = {x for x in range(G.vcount())}
    cdef list blk_n = [set() for x in range(steps)]
    cdef str curr
    cdef set u, comp_blk 
    cdef list neighbors, weights, probs
    
    while ptr < steps:
        
        curr = walk[ptr-1]
        
        u = set()
        neighbors = G.neighbors(G.vs["name"].index(walk[ptr-1]), mode="out")
        weights = [G.es.find(_source = G.vs["name"].index(walk[ptr-1]), _target=x)["weight"] for x in neighbors]
        probs = [x / sum(weights) for x in weights]
        
        u = u.union({x for x in neighbors if allowed_transition(walk[ptr-lag], G.vs[x]["name"]) == True})
        
        comp_blk = vset - blk_n[ptr]
        u = u.intersection(comp_blk)
        
        if len(u) > 0:
            choice = np.random.choice(neighbors, p = probs)
            walk[ptr] = str(G.vs[choice]["name"])
            ptr += 1
        else:
            ptr -= 1
            blk_n[ptr] = blk_n[ptr].union({curr})
            
        if ptr < lag:
            print("Error! Could not find a random walk on G with the given initial conditions. Try again!")
            return None
        
    return walk
