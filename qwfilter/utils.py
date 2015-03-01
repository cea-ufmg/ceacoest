'''General utility and convenience functions.'''

import numpy as np


def central_diff(f, x, min_step=1e-7, rel_step=1e-7):
    '''Vectorized central finite difference.
    
    Parameters
    ----------
    f : callable
        Function to be differentiated.
    x : array_like
        Point at which the derivative will be evaluated.

        
    Examples
    --------
    >>> f = lambda x: [np.sin(x[0]), np.cos(x[1]) + np.exp(x[0])]
    >>> x = [1, 2]
    >>> numerical_diff = central_diff(f, x)
    >>> analytical_diff = [[np.cos(x[0]), 0], [np.exp(x[0]), -np.sin(x[1])]]
    >>> np.allclose(numerical_diff, analytical_diff)
    True
    
    '''
    x = np.asarray(x, float)
    h = np.maximum(x*rel_step, min_step)

    diff = None
    for i, hi in np.ndenumerate(h):
        xplus = x.copy()
        xminus = x.copy()
        xplus[i] += hi
        xminus[i] -= hi
        fplus = np.asarray(f(xplus))
        fminus = np.asarray(f(xminus))
        
        if diff is None:
            diff = np.empty(fplus.shape + x.shape)
        
        diff[(...,) + i] = 0.5*(fplus - fminus)/hi
    
    return diff


def forward_diff(f, x, min_step=1e-7, rel_step=1e-7):
    '''Vectorized forward finite difference.
    
    Parameters
    ----------
    f : callable
        Function to be differentiated.
    x : array_like
        Point at which the derivative will be evaluated.

        
    Examples
    --------
    >>> f = lambda x: [np.sin(x[0]), np.cos(x[1]) + np.exp(x[0])]
    >>> x = [1, 2]
    >>> numerical_diff = forward_diff(f, x)
    >>> analytical_diff = [[np.cos(x[0]), 0], [np.exp(x[0]), -np.sin(x[1])]]
    >>> np.allclose(numerical_diff, analytical_diff)
    True
    
    '''
    x = np.asarray(x, float)
    h = np.maximum(x*rel_step, min_step)
    f0 = np.asarray(f(x))
    
    diff = np.empty(f0.shape + x.shape)
    for i, hi in np.ndenumerate(h):
        xi = x.copy()
        xi[i] += hi
        diff[(...,) + i] = (f(xi) - f0)/hi
    
    return diff

