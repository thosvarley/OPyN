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
from scipy.stats import pearsonr
from libc.math cimport log2
from scipy.signal import argrelextrema
from sklearn.neighbors import KDTree
from collections import Counter


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
def embed_series(double[:] X, int dim, int lag):
    """
    Constructs a time-delay embedding of a continuous timeseries.
    
    Arguments:
        X:
            A one-dimensional Numpy array with dtype = "double."
        dim:
            The integer embedding dimension.
        lag:
            The integer embedding lag.
    
    Returns:
        embedding:
            The embedded series as a 2-dimensional array.
    """
    cdef float[:,:] timestep = np.zeros((X.shape[0]-lag*(dim-1), dim), dtype="single")
    cdef int t, d, s, i, x
    
    for t in range(X.shape[0]-lag*(dim-1)):
        for d in range(dim):
            timestep[t][d] = X[t + d*lag]
    
    return timestep

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def perm_embedding(double[:] X, int dim, int lag):
    """
    Constructs the permutation embedding from a given time-series.
    
    Arguments:
        X:
            A one-dimensional Numpy array with dtype = "double." 
        dim:
            The integer embedding dimension.
        lag:
            The integer embedding lag.
    
    Returns:
        series:
            The permutation embedding of the original series ts. 
    """
    cdef float[:,:] timestep = np.zeros((X.shape[0]-lag*(dim-1), dim), dtype="single")
    cdef object[:] series = np.zeros((X.shape[0]-lag*(dim-1))-1, dtype=object)
    cdef int t, d, s, i, x
    
    for t in range(X.shape[0]-lag*(dim-1)):
        for d in range(dim):
            timestep[t][d] = X[t + d*lag]
    
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
def optimal_lag(double[:] X, int lrange, int step):
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
        step:
            Whether to sample the entire lrange, or instead take it in steps of a given size. 
            Computationally faster, but returns a less precise lag. 
    
    Returns:
        lag:
            The lag that gives the first zero of the autocorrelation function.. 
    """
    cdef double[:] autocorr = np.ones((lrange)//step)
    cdef bint pos = True
    cdef int extrema = -1
    cdef int i
    
    for i in range(1, autocorr.shape[0]):
        if pos == True:
            autocorr[i] = pearsonr(X[:-step*i], X[step*i:])[0]
            if autocorr[i] < 0 and autocorr[i-1] > 0:
                pos = False
                extrema = i-1   
                
    return extrema*step

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
def false_nearest_neighbors(double[:] X, float threshold, int lag, int drange=20, str metric="euclidean"):    
    """
    Returns the minimum embedding dimension d such that the proportion of false nearest neighbors
    from d / d-1 embeddings is below the threshold. 
    
    Arguments:
        X:
            A one-dimensional Numpy array with dtype = "double". The time-series.
        threshold:
            A floating point value - the threshold of false nearest neighbors.
        lag:
            The integer embedding lag.
        drange:
            The range of possible embedding dimensions to try, from 1...drange.
        metric:
            The distance metric used to find the nearest neighbors. 
    Returns:
        dim:
            The smallest embedding dimension that satisfies the false nearest neighbors criteria.
    """
    assert threshold > 0 and type(threshold) == float, "The threshold must be a positive interger" 
    
    queries = np.zeros((X.shape[0]-lag*(drange-1), drange-1))
    
    cdef int i
    cdef float[:,:] series
    
    for i in range(1,drange):
        series = embed_series(X, i, 1)
        kdt = KDTree(series, metric=metric)
        queries[:,i-1] = kdt.query(series, k=2, return_distance=False)[:queries.shape[0],-1]
        
        if i > 1:
            if (queries[:,i-2] != queries[:,i-1]).sum() / queries.shape[0] < threshold:
                return(i)
    

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

##################################
### BASIC ANALYSES OF THE NETWORKS
##################################

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def entropy_production(G):
    """
    Returns the entropy produced by the process, estimated as the degree to which the transition 
    probability matrix breaks the detailed balance critera. For an example of the application, see:
        Lynn, C., Cornblath, E., Papadopoulos, L., Bertolero, M., & Bassett, D. (2020). 
        Broken detailed balance and entropy production in the human brain,
        arXiv:2005.02526
    
    Arguments:
        G:
            The OPN, in the form of a python-igraph graph.
    
    Returns:
        entropy:
            The entropy, in bits.
    
    """
    
    cdef double[:,:] transmat = np.array(list(G.get_adjacency(attribute="weight")))
    cdef double entropy = 0.0
    cdef int i, j
    cdef double[:] sums = np.sum(transmat, axis=1)
    
    for i in range(transmat.shape[0]):
        if sums[i] > 0.0:
            for j in range(transmat.shape[0]):
                transmat[i][j] = transmat[i][j] / sums[i]
    
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[0]):
            if transmat[i][j] != 0 and transmat[j][i] != 0:
                entropy += transmat[i][j] * log2(transmat[i][j] / transmat[j][i])
    
    return entropy 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def determinism(G, bint norm = False):
    """
    Measures the degree to which the future of a system modeled by an OPN can be predicted given
    the present. In determinism(G) = 1, then the future is perfectly predictable given the present.
    
    Based on work presented in:
        Brennan Klein, Erik Hoel, "The Emergence of Informative Higher Scales in Complex Networks", 
        Complexity, vol. 2020, Article ID 8932526, 12 pages, 2020. 
        https://doi.org/10.1155/2020/8932526
    
    Arguments:
        G:
            The OPN, in the form of a python-igraph.
        norm:
            If true, normalizes the determinism onto the range 0-1
    
    Returns:
        Determinism:
            The determinism in bits (if norm == False).
    """
    cdef double[:,:] transmat = np.array(list(G.get_adjacency(attribute="weight")))
    cdef double N = transmat.shape[0]
    cdef int i, j
    cdef double[:] sums = np.sum(transmat, axis=1)

    for i in range(transmat.shape[0]):
        if sums[i] > 0:
            for j in range(transmat.shape[0]):
                transmat[i][j] = transmat[i][j] / sums[i]
        
    cdef double avg_ent = 0.0
    cdef double row_ent
    
    for i in range(transmat.shape[0]):
        row_ent = 0
        for j in range(transmat.shape[0]):
            if transmat[i][j] > 0:
                row_ent += transmat[i][j] * log2(transmat[i][j])
        avg_ent += -1*row_ent
        
    avg_ent /= N
    
    if norm == False:
        return log2(N) - avg_ent
    elif norm == True:
        return (log2(N) - avg_ent) / log2(N)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def degeneracy(G, bint norm = False):
    """
    Measures the degree to which information about the past is lost when different trajectories "run together". 
    Closely related to the entropy production.
    
    Based on work presented in:
        Brennan Klein, Erik Hoel, "The Emergence of Informative Higher Scales in Complex Networks", 
        Complexity, vol. 2020, Article ID 8932526, 12 pages, 2020. 
        https://doi.org/10.1155/2020/8932526
    
    Arguments:
        G:
            The OPN, in the form of a python-igraph.
        norm:
            If true, normalizes the degeneracy onto the range 0-1
    
    Returns:
        Degeneracy:
            The degeneracy in bits (if norm == False).
    """
    cdef double[:,:] transmat = np.array(list(G.get_adjacency(attribute="weight")))
    cdef double N = transmat.shape[0]
    cdef int i, j
    cdef double[:] sums = np.sum(transmat, axis=1)

    for i in range(transmat.shape[0]):
        if sums[i] > 0:
            for j in range(transmat.shape[0]):
                transmat[i][j] = transmat[i][j] / sums[i]  
    
    cdef double[:] avg_row = np.mean(transmat, axis=0)
    
    cdef double ent = 0
    
    for i in range(avg_row.shape[0]):
        ent += avg_row[i] * log2(avg_row[i])
    ent *= -1
    
    if norm == False:
        return log2(N) - ent
    elif norm == True:
        return (log2(N) - ent) / log2(N)


@cython.initializedcheck(False)
def effective_information(G, norm=False):
    """
    Measures the total amount of information the OPN encodes in it's causal structure.
    
    Based on work presented in:
        Brennan Klein, Erik Hoel, "The Emergence of Informative Higher Scales in Complex Networks", 
        Complexity, vol. 2020, Article ID 8932526, 12 pages, 2020. 
        https://doi.org/10.1155/2020/8932526
    
    Arguments:
        G:
            The OPN, in the form of a python-igraph.
        norm:
            If true, normalizes the degeneracy onto the range 0-1
    
    Returns:
        EI:
            The EI in bits (if norm == False).
    """
    return determinism(G, norm=norm) - degeneracy(G, norm=norm)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def permutation_entropy(series, norm=False):
    """

    
    """
    
    cdef dict counts = dict(Counter(series))
    cdef float N = series.shape[0]
    cdef str key
    cdef dict probs = {key : float(counts[key]) / N for key in counts.keys()}
    
    cdef float ent = 0.0
    cdef int i
    
    for key in probs.keys():
        ent += probs[key] * log2(probs[key])
    
    if norm == False:
        return -1*ent
    elif norm == True:
        return (-1*ent) / log2(len(probs))

def shannon_entropy_rate(G, series):
    """
    
        Presented in:
        Crutchfield, J. P. (2012). Between order and chaos. 
        Nature Physics, 8(1), 17–24. 
        https://doi.org/10.1038/nphys2190
    """
    return None