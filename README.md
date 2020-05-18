# OPyN
A Python package for construction and analysis of ordinal partition networks from time-series data.

### Dependencies:
 - Python-iGraph (v. 0.7.1 or newer)
 - Numpy (v. 1.13.3 or newer)
 - Scipy (v. 1.3.1 or newer)
 - Itertools
 - Cython 

### Installation:
- Download OPyN to your target directory. 
- Run ```python setup.py install```


## Functions:

```perm_embedding(ts, dim, lag)```

Peforms a permutaiton embedding on the given timeseries (```ts```) using an embedding dimension ```dim``` and lag of ```lag```. The time-series must be a one-dimensional Numpy array with dtype of "double." The embedding dimension and lag must both be integers.  

```OPN(ts, dim, lag)```

Constructs the ordinal partition network from the given timeseries, embedding dimension, and lag. Returns ```G```, a Python-iGraph network, as well as the embedded timeseries. The type-constraints are the same as in the permutation embedding function.

```optimal_lag(ts, lrange=25)```

Given some time-series, this function finds the optimal lag as the first zero of the pearson autocorrelation function. By default it sweeps lags 1-25. 

```optimal_dim(ts, lag, lrange=20)```

Given some time-series and a pre-selected lag, returns the embedding dimension where the variance of the resulting degree distribution is the largest. This is generally considered to be a reasonable approximation of the optimal embedding dimension. 

```constrained_walk(G, lag, steps)```

Given an OPN ```G```, constructed with a given lag, this will peform a constrained random walk on the network to produce surrogate discrete time-series with a pre-defined number of steps. If such a walk cannot be found, returns a ```None``` type. 
Algorithm taken from:

```McCullough, M., Sakellariou, K., Stemler, T., & Small, M. (2017). Regenerating time series from ordinal networks. Chaos: An Interdisciplinary Journal of Nonlinear Science, 27(3), 035814. https://doi.org/10.1063/1.4978743```

