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
from itertools import product, permutations
from scipy.stats import entropy, pearsonr
from libc.math cimport log2
from scipy.signal import argrelextrema

@cython.boundscheck(False)
@cython.initializedcheck(False)
def allowed_transition(a, b):
    """
    Tests whether a walker on the OPN is allowed to transition from node a to b. 
    """
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
    """
    Constructs the permutation embedding from a given time-series.
    
    Arguments:
        ts:
            A one-dimensional Numpy array with dtype = "double." The time-series. 
        dim:
            The integer embedding dimension.
        lag:
            The integer embedding lag.
    
    Returns:
        series:
            The permutation embedding of the original series ts. 
    """
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
    """
    Constructs the time-lagged, ordinal partition network from a given time-series.
    
    Arguments:
        ts:
            A one-dimensional Numpy array with dtype = "double." The time-series. 
        dim:
            The integer embedding dimension.
        lag:
            The integer embedding lag.
    
    Returns:
        G:
            A python-iGraph network.
        series:
            The permutation embedding of the original series ts. 
        
    """
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
    """
    Returns the embedding lag corresponding to the first zero of the autocorrelation function. See:
        McCullough, M., Small, M., Stemler, T., & Iu, H. H.-C. (2015). 
        Time lagged ordinal partition networks for capturing dynamics of continuous dynamical systems. 
        Chaos: An Interdisciplinary Journal of Nonlinear Science, 25(5), 053101. 
        https://doi.org/10.1063/1.4919075
    
    Arguments:
        X:
            A one-dimensional Numpy array with dtype = "double." The time-series.
        lrange:
            The range of possible embedding lags to try, from 1...drange.
    
    Returns:
        lag:
            The lag that gives the first zero of the autocorrelation function.. 
    """
    
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
    """
    Returns the embedding dimension that maximizes the variange in the degree distribution of the resulting OPN.
    A proxy for the optimal embedding dimension. See:
        McCullough, M., Small, M., Stemler, T., & Iu, H. H.-C. (2015). 
        Time lagged ordinal partition networks for capturing dynamics of continuous dynamical systems. 
        Chaos: An Interdisciplinary Journal of Nonlinear Science, 25(5), 053101. 
        https://doi.org/10.1063/1.4919075
    
    Arguments:
        X:
            A one-dimensional Numpy array with dtype = "double." The time-series.
        lag:
            The integer embedding lag.
        drange:
            The range of possible embedding dimensions to try, from 1...drange.
    
    Returns:
        dim:
            The embedding dimension that maximized the variance in the degree distribution of the resulting network. 
    """
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
    """
    Returns a constrained, random walk on the OPN. 
    This corresponds to a surrogate time-series, which preserves higher-order dynamics.
    
    Based on the algorithm in:
        McCullough, M., Sakellariou, K., Stemler, T., & Small, M. (2017). 
        Regenerating time series from ordinal networks. 
        Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(3), 035814. 
        https://doi.org/10.1063/1.4978743
    
    Arguments:
        G:
            A python-iGraph network returned from the OPN() function.
        lag:
            The integer temporal lag used to construct G.
        steps:
            The integer number of steps to run the walk.
    
    Returns:
        If the walk is successful, returns an array of strings corresponding to steps in the walk.
        If the walk is unnsucessful, returns None. 
    """
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

#Make sure both OPNs have the same vertex sets, assigned to the same indices (this was hard!)

def sort_networks(G1, G2):
    
    cdef int i
    cdef list G1_nodes = [x["name"] for x in G1.vs]
    cdef list G2_nodes = [x["name"] for x in G2.vs]
    cdef np.ndarray union = np.union1d(G1_nodes, G2_nodes)
    cdef np.ndarray G1_diff = np.setdiff1d(union, G1_nodes)
    cdef np.ndarray G2_diff = np.setdiff1d(union, G2_nodes)
    
    G1.add_vertices(G1_diff)
    G2.add_vertices(G2_diff)
    
    cdef dict G1_dict = { G1.vs[i]["name"] : i for i in range(G1.vcount()) }
    cdef dict G2_dict = { G2.vs[i]["name"] : i for i in range(G2.vcount()) }
    
    cdef list keys1 = list(G1_dict.keys())
    cdef list keys2 = list(G2_dict.keys())
    keys1.sort()
    keys2.sort()
    
    cdef list sort1 = [G1_dict[i] for i in keys1]
    cdef list sort2 = [G2_dict[i] for i in keys2]
    cdef list perm1 = [i for i in np.argsort(sort1)]
    cdef list perm2 = [i for i in np.argsort(sort2)]
        
    G1 = G1.permute_vertices(perm1)
    G2 = G2.permute_vertices(perm2)
    
    return G1, G2


def full_OPN_space(G):
    """
    Adds all the missing states to an OPN as isolated nodes (in/out-degree = 0)
    Use with care - for OPNs with high embedding dimensions, the space explodes with N!
    Don't use above D ~ 8 or so.
    """
    cdef list G_nodes = [x["name"] for x in G.vs]
    cdef list perms = ["".join(x) for x in permutations(G.vs[0]["name"])]
    cdef np.ndarray diff = np.setdiff1d(perms, G_nodes)
    
    cdef int i
    G.add_vertices(diff)
    cdef dict G_dict = { G.vs[i]["name"] : i for i in range(G.vcount()) }
    cdef list keys = list(G_dict.keys())
    keys.sort()
    cdef list sort = [G_dict[x] for x in keys]
    cdef list perm = [x for x in np.argsort(sort)]
        
    G = G.permute_vertices(perm)
    
    return G