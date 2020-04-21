"""Common statistical modelling code."""


import numpy as np


def multivariate_normal_pdf(x, mean, cov):
    """Unnormalized multivariate normal probability density function."""
    # Convert to ndarray
    x = np.asanyarray(x)
    mean = np.asanyarray(mean)
    cov = np.asanyarray(cov)
    
    # Deviation from mean
    dev = x - mean
    if np.all(np.ma.getmaskarray(dev)):
        return np.ones(dev.shape[:-1])
    
    # Broadcast cov, if needed
    if cov.ndim <= dev.ndim:
        extra_dim = 1 + dev.ndim - cov.ndim 
        cov = np.broadcast_to(cov, (1,) * extra_dim + cov.shape)
    
    exponent = -0.5 * np.ma.inner(dev, np.linalg.solve(cov, dev))
    return np.exp(exponent) / np.sqrt(np.linalg.det(cov))


def multivariate_normal_rvs(mean, cov):
    std = svd_sqrt(cov)
    nw = std.size[-1]
    w = np.random.randn(*np.shape(mean), nw)
    return mean + np.einsum('...ij,...j', std, w)


def svd_sqrt(a):
    u, s, vh = np.linalg.svd(a)
    return u * s

