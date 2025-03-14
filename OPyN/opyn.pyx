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
from scipy.stats import pearsonr, entropy
from libc.math cimport log2
from scipy.signal import argrelextrema
from sklearn.neighbors import KDTree
from collections import Counter


@cython.boundscheck(False)
@cython.initializedcheck(False)
def allowed_transition(a, b):
    """
    Tests whether a given transition from embedded 
    vector a to b is allowed given the constraints
    of the ordinal partition embedding.

    Parameters
    ----------
    a : iterable
        A given ordinal partition vector.
    b : iterable
        A given ordinal partition vector.

    Returns
    -------
    bool
        Whether a transition from A to B is allowed.

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
def embed_series(
        np.ndarray[np.float64_t, ndim=1] X, 
        int dim, 
        int lag) -> np.ndarray:
    """

    Parameters
    ----------
    X : np.ndarray
        A one-dimensional Numpy array with dtype = "double.".
    dim : int
        The embedding dimension.
    lag : int
        The embedding lag.

    Returns
    -------
    embedded : np.ndarray
        A two dimensional array. Each row is an embedded vector
        constructed from X.

    """
    cdef np.ndarray[np.float64_t, ndim=2] embedded = np.zeros((X.shape[0]-lag*(dim-1), dim), 
                                        dtype=np.float64)
    cdef int t, d
    
    for t in range(X.shape[0]-lag*(dim-1)):
        for d in range(dim):
            embedded[t][d] = X[t + d*lag]
    
    return embedded

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def ordinal_partition_embedding(
        np.ndarray[np.float64_t, ndim=1] X, 
        int dim, 
        int lag) -> np.ndarray:
    """
    Embeds the time series X and then converts the real-valued vectors
    into ordinal partitions. 
    
    Parameters
    ----------
    X : np.ndarray
        A one-dimensional Numpy array with dtype = "double.".
    dim : int
        The embedding dimension.
    lag : int
        The embedding lag.

    Returns
    -------
    series : np.ndarray
        The series of embedded vectors.
        A numpy object of dim-length strings.

    """
    
    cdef np.ndarray[np.float64_t, ndim=2] embedded = embed_series(X, dim, lag)
    cdef np.ndarray[object, ndim=1] series = np.zeros(embedded.shape[0], dtype=object)
    cdef int d, s, i, x
    
    cdef np.ndarray[np.int64_t, ndim=2] sort = np.argsort(np.argsort(embedded, axis=1))
    cdef np.ndarray[np.int64_t, ndim=2] unique = np.unique(sort, axis=0)
    
    cdef dict arr2string = {unique[i].tobytes() : "".join(
        [str(x) for x in unique[i]]) 
        for i in range(unique.shape[0])}
    
    for s in range(sort.shape[0]):
        series[s] = arr2string[sort[s].tobytes()]
    
    return series


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def permutation_entropy(
        np.ndarray[np.float64_t, ndim=1] X, 
        int dim, 
        int lag,
        int base = 2) -> float:
    """
    This function computes the permutation entropy as an omnibus function 
    (going from the time series to the entropy), but if one wants to 
    save time on the emebdded, entropy can be computed directly from the 
    embedded series() object returned by ordinal_partition_embedding().
    
    
    Parameters
    ----------
    X : np.ndarray
        A one-dimensional Numpy array with dtype = "double.".
    dim : int
        The embedding dimension.
    lag : int
        The embedding lag.
    base : int, optional.
        The base of the logarithm used to compute the entropy.
        The default value is 2 (bits).

    Returns
    -------
    float.
        The permutation entropy

    """
    
    cdef np.ndarray[object, ndim=1] series = ordinal_partition_embedding(X, dim, lag)
    cdef np.ndarray[np.int64_t, ndim=1] counts
    
    _, counts = np.unique(series, return_counts=True)
    
    return entropy(counts, base = base)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def ordinal_partition_network(
        np.ndarray[np.float64_t, ndim=1] X, 
        int dim, 
        int lag):
    """
    Constructs the ordinal partition network (OPN). The OPN is a finite-state 
    machine that represents the structure of a continuous time series.
    
    See:
        McCullough, M., Small, M., Stemler, T., & Iu, H. H.-C. (2015). 
        Time lagged ordinal partition networks for capturing dynamics of 
        continuous dynamical systems. 
        Chaos: An Interdisciplinary Journal of Nonlinear Science, 25(5), Article 5. 
        https://doi.org/10.1063/1.4919075


    Parameters
    ----------
    X : np.ndarray
        A one-dimensional Numpy array with dtype = "double.".
    dim : int
        The embedding dimension.
    lag : int
        The embedding lag.

    Returns
    -------
    G : ig.Graph
        A python-igraph graph representing the 
        ordinal partition network.
    series : np.ndarray
        The series of embedded vectors.
        A numpy object of dim-length strings. 
    """
    
    cdef np.ndarray[object, ndim=1] series = ordinal_partition_embedding(X, dim, lag)
    cdef int N = series.shape[0]
    
    cdef list unq = np.unique(series).tolist()

    cdef list edges = [(series[i], series[i+1]) for i in range(N - 1)]
    cdef np.ndarray[np.float64_t, ndim=1] weights = np.ones(N - 1)

    G = ig.Graph(directed=True)
    G.add_vertices(unq)
    G.add_edges(edges)
    G.es["weight"] = weights
    
    G = G.simplify(loops=False,
                   combine_edges = dict(weight="sum"))
    
    return G, series
    

@cython.initializedcheck(False)
@cython.boundscheck(False)
def optimal_lag(
        np.ndarray[np.float64_t, ndim=1] X, 
        int lrange, 
        int step):
    """
    Returns the embedding lag corresponding to the first zero of the 
    autocorrelation function. See:
        
        McCullough, M., Small, M., Stemler, T., & Iu, H. H.-C. (2015). 
        Time lagged ordinal partition networks for capturing dynamics of 
        continuous dynamical systems. 
        Chaos: An Interdisciplinary Journal of Nonlinear Science, 
        25(5), 053101. 
        https://doi.org/10.1063/1.4919075
    
    Parameters
    ----------
    X : np.ndarray
        A one-dimensional Numpy array with dtype = "double."
    lrange : int
        The range of possible embedding lags to try, from 1...drange.
    step :
        Whether to sample the entire lrange, or instead take it in steps 
        of a given size. 
        Computationally faster, but returns a less precise lag. 
    
    Returns
    -------
    lag : int
        The lag that gives the first zero of the autocorrelation function.. 
    """
    cdef np.ndarray[np.float64_t, ndim=1] autocorr = np.ones((lrange)//step)
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
def optimal_dim(
        np.ndarray[np.float64_t, ndim=1] X, 
        int lag, 
        int drange = 20):
    """
    Returns the embedding dimension that maximizes the variange in the degree 
    distribution of the resulting OPN.
    A proxy for the optimal embedding dimension. See:
        McCullough, M., Small, M., Stemler, T., & Iu, H. H.-C. (2015). 
        Time lagged ordinal partition networks for capturing dynamics of 
        continuous dynamical systems. 
        Chaos: An Interdisciplinary Journal of Nonlinear Science, 
        25(5), 053101. 
        https://doi.org/10.1063/1.4919075
    
    Arguments
    ---------
    X : np.ndarray
        A one-dimensional Numpy array with dtype = "double." 
    lag : int
        The integer embedding lag.
    drange : int
        The range of possible embedding dimensions to try, 
        from 1...drange.
    
    Returns
    -------
    dim : int
        The embedding dimension that maximized the variance in the degree 
        distribution of the resulting network. 
    """
    cdef np.ndarray[np.float64_t, ndim=1] var = np.zeros(drange, dtype=np.float64)
    cdef int i 

    for i in range(2, drange):
        G, s = ordinal_partition_network(X, i, lag)
        var[i] = np.var(G.degree())

    return np.argmax(var)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def false_nearest_neighbors(
        np.ndarray[np.float64_t, ndim=1] X, 
        float threshold, 
        int lag, 
        int drange=20, 
        str metric="chebyshev",
        int k=1):    
    """
    Returns the minimum embedding dimension d such that the proportion of 
    false nearest neighbors
    from d / d-1 embeddings is below the threshold. 
    
    Parameters
    ----------
    X : np.ndarray
        A one-dimensional Numpy array of dtype=double.
    threshold : float
        The threshold of false nearest neighbors.
    lag : int
        The embedding lag.
    drange : int, optional.
        The range of possible embedding dimensions to try, from 1...drange.
        The default drange is 20
    metric : str, optional
        The distance metric used to find the nearest neighbors.
        Any distance implemented in scipy.spatial is allowed. 
        The default distance is the Chebyshev distance.
    k : int, optional
        The kth-nearest neighbor. Defaults to first nearest neighbor. 
    
    Returns
    -------
    dim : int
        The smallest embedding dimension that satisfies the false 
        nearest neighbors criteria.
    """
    assert threshold > 0 and type(threshold) == float, "The threshold must be \
        a positive interger" 
    
    queries = np.zeros((X.shape[0]-lag*(drange), drange))
    
    cdef np.ndarray[np.float64_t, ndim=2] series
    
    cdef int i
    for i in range(1,drange+1):
        series = embed_series(X, i, lag)
        kdt = KDTree(series, metric=metric)
        queries[:,i-1] = kdt.query(series, 
                                   k=(k+1), 
                                   return_distance=False)[:queries.shape[0],-1]
        
        if i >= 2:
            prop = (queries[:,i-2] != queries[:,i-1]).sum() / queries.shape[0]
            print(f"Proportion of FNN: {prop}")
            if prop <= threshold:
                return(i-1)
    print("You need a longer drange or a more relaxed threshold.")
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
def constrained_walk(G, int lag, int steps):
    """
    Returns a constrained, random walk on the OPN. 
    This corresponds to a surrogate time-series, which preserves 
    higher-order dynamics.
    
    Based on the algorithm in:
        McCullough, M., Sakellariou, K., Stemler, T., & Small, M. (2017). 
        Regenerating time series from ordinal networks. 
        Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(3), 035814. 
        https://doi.org/10.1063/1.4978743
    
    Parameters
    -----------
    G : ig.Graph
        A python-iGraph network returned from the OPN() function.
    lag : int
        The integer temporal lag used to construct G.
    steps : int
        The integer number of steps to run the walk.
    
    Returns
    -------
        list.
            If the walk is successful, a list of states. If the walk 
            is unscessful, nothing is returned.
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
        weights = [G.es.find(_source = G.vs["name"].index(walk[ptr-1]), 
                             _target=x)["weight"] for x in neighbors]
        probs = [x / sum(weights) for x in weights]
        
        u = u.union({x for x in neighbors if \
                     allowed_transition(walk[ptr-lag], G.vs[x]["name"]) == True})
        
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
            print("Error! Could not find a random walk on G with the \
                  given initial conditions. Try again!")
            return None
        
    return np.array(walk)


def sort_networks(G1, G2):
    """
    Expands networks G1 and G2 so that they have the same vertex set
    and then sorts the vertices so the names all have the same
    indices. 
    
    WARNING: Modifies the graphs in place. 

    Parameters
    ----------
    G1 : ig.Graph
        The first ordinal partition network.
    G2 : ig.Graph
        The second ordinal partition network.

    Returns
    -------
    G1 : ig.Graph
        The first ordinal partition network.
    G2 : ig.Graph
        The second ordinal partition network.
    """
    
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
    Use with care - for OPNs with high embedding dimensions, the space explodes 
    with N!
    Don't use above D ~ 8 or so.
    
    WARNING: Modifies G in-place. 

    Parameters
    ----------
    G : ig.Graph
        The ordinal partition network.

    Returns
    -------
    G : ig.Graph
        The ordinal partition network.

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
def transition_probability_matrix(G):
    """
    Returns the transition-probability matrix. 
    Each row and column correspond to a state, and 
    the entry i,j gives the probability of seeing 
    state j conditional on being in state i.

    Parameters
    ----------
    G : ig.Graph
        The ordinal partition network.

    Returns
    -------
    np.ndarray
        The transition probability matrix.
    list
        The embeddings associated with each row.

    """
    
    cdef np.ndarray[np.float64_t, ndim=2] transmat = np.array(
        list(
            G.get_adjacency(attribute="weight")
            )
        
        )
    cdef np.ndarray[np.float64_t, ndim=1] sums = transmat.sum(axis=1)

    return transmat / sums[:,np.newaxis], G.vs["name"]




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def entropy_production(G):
    """
    Returns the entropy produced by the process, estimated as the degree to 
    which the transition probability matrix breaks the detailed balance critera. 
    
    See:
        Lynn, C. W., Cornblath, E. J., Papadopoulos, L., 
        Bertolero, M. A., & Bassett, D. S. (2021). 
        Broken detailed balance and entropy production in the human brain. 
        Proceedings of the National Academy of Sciences, 118(47). 
        https://doi.org/10.1073/pnas.2109889118


    Parameters
    ----------
    G : ig.Graph
        The ordinal partition network.

    Returns
    -------
    entropy : float
        The entropy production.
        The higher the number, the greater the irreversibility of the 
        process. 
    """
    
    cdef np.ndarray[np.float64_t, ndim=2] transmat
    transmat, _ = transition_probability_matrix(G)
    
    cdef double entropy = 0.0
    cdef int i, j
    
    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[0]):
            if (transmat[i][j]) != 0 and (transmat[j][i] != 0):
                entropy += transmat[i][j] * log2(transmat[i][j] / transmat[j][i])
    
    return entropy 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def determinism(G, bint norm = False):
    """
    Measures the degree to which the future of a system modeled by an OPN can
    be predicted given
    the present. In determinism(G) = 1, then the future is perfectly 
    predictable given the present.
    
    See:
        Brennan Klein, Erik Hoel, 
        "The Emergence of Informative Higher Scales in Complex Networks", 
        Complexity, vol. 2020, Article ID 8932526, 12 pages, 2020. 
        https://doi.org/10.1155/2020/8932526

    Parameters
    ----------
    G : ig.Graph
        The ordinal partition network.
    norm : bool, optional
        Whether to normalize the measure by log2(N). 
        The default is False.

    Returns
    -------
    float
        The determinism.
        The higher the number, the more deterministically (on average)
        state A -> state B. 
    """
    
    cdef np.ndarray[np.float64_t, ndim=2] transmat
    transmat, _ = transition_probability_matrix(G)
    
    cdef int N = transmat.shape[0] 
    cdef double Nf = float(N)
    
    cdef double avg_ent = 0.0
    cdef double row_ent
    
    cdef int i, j
    for i in range(N):
        row_ent = 0
        for j in range(N):
            if transmat[i,j] > 0:
                row_ent += transmat[i,j] * log2(transmat[i,j])
        avg_ent += -1*row_ent
        
    avg_ent /= Nf
    
    if norm == False:
        return log2(Nf) - avg_ent
    elif norm == True:
        return (log2(Nf) - avg_ent) / log2(Nf)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def degeneracy(G, bint norm = False):
    """
    Measures the degree to which information about the past is lost 
    when different trajectories "run together". 
    Closely related to the entropy production.
    
    See:
        Brennan Klein, Erik Hoel, 
        "The Emergence of Informative Higher Scales in Complex Networks", 
        Complexity, vol. 2020, Article ID 8932526, 12 pages, 2020. 
        https://doi.org/10.1155/2020/8932526

    Parameters
    ----------
    G : ig.Graph
        The ordinal partition network.
    norm : bool, optional
        Whether to normalize the measure by log2(N). 
        The default is False.

    Returns
    -------
    float
        The degeneracy.
        The higher the number, the more states tend to all run together. 

    """
    
    cdef np.ndarray[np.float64_t, ndim=2] transmat
    transmat, _ = transition_probability_matrix(G) 
    
    cdef int N = transmat.shape[0]
    cdef double Nf = float(N)
    
    cdef np.ndarray[np.float64_t, ndim=1] avg_row = np.mean(transmat, axis=0)
    
    cdef double ent = 0
    
    cdef int i 
    for i in range(N):
        ent += avg_row[i] * log2(avg_row[i])
    ent *= -1
    
    if norm == False:
        return log2(Nf) - ent
    elif norm == True:
        return (log2(Nf) - ent) / log2(Nf)


@cython.initializedcheck(False)
def effective_information(G, norm=False):
    """
    Measures the total amount of information the OPN encodes
    in its causal structure.
    
    See:
        Brennan Klein, Erik Hoel, 
        "The Emergence of Informative Higher Scales in Complex Networks", 
        Complexity, vol. 2020, Article ID 8932526, 12 pages, 2020. 
        https://doi.org/10.1155/2020/8932526

    Parameters
    ----------
    G : ig.Graph
        The ordinal partition network.
    norm : bool, optional
        Whether to normalize the measure by log2(N). 
        The default is False.

    Returns
    -------
    float
        The effective information
    """
    
    return determinism(G, norm=norm) - degeneracy(G, norm=norm)
