"""General utility and convenience functions."""


import functools
import re

import numpy as np


def central_diff(f, x, min_step=1e-7, rel_step=1e-7):
    """Vectorized central finite difference.
    
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
    >>> analytical_diff = [[np.cos(x[0]), np.exp(x[0])], [0, -np.sin(x[1])]]
    >>> np.allclose(numerical_diff, analytical_diff)
    True
    
    """
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
            diff = np.empty(x.shape + fplus.shape)
        
        diff[i] = 0.5*(fplus - fminus)/hi
    
    return diff


def forward_diff(f, x, min_step=1e-7, rel_step=1e-7):
    """Vectorized forward finite difference.
    
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
    >>> analytical_diff = [[np.cos(x[0]), np.exp(x[0])], [0, -np.sin(x[1])]]
    >>> np.allclose(numerical_diff, analytical_diff)
    True
    
    """
    x = np.asarray(x, float)
    h = np.maximum(x*rel_step, min_step)
    f0 = np.asarray(f(x))
    
    diff = np.empty(x.shape + f0.shape)
    for i, hi in np.ndenumerate(h):
        xi = x.copy()
        xi[i] += hi
        diff[i] = (f(xi) - f0)/hi
    
    return diff


def extract_subkeys(d, base):
    """Extract items from a dictionary with keys starting with a given string.

    >>> d = {'xxx_yyy': 1, 'xyz': 'abc'}
    >>> extract_subkeys(d, 'xxx_')
    {'yyy': 1}
    
    """
    subkeys = {}
    for key, val in d.items():
        match = re.match('%s(?P<subkey>\w+)' % base, key)
        if match:
            subkeys[match.group('subkey')] = val
    return subkeys


def flat_cat(*args, order='C'):
    """Flattens and then concatenates an array_like sequence.

    >>> flat_cat([[1,2,3]], [4, 5], 6)
    array([1, 2, 3, 4, 5, 6])
    
    """
    return np.concatenate([np.ravel(arg, order=order) for arg in args])


try:
    from cached_property import cached_property
except ModuleNotFoundError:
    def cached_property(f):
        """On-demand property which is calculated only once and memorized."""
        return property(functools.lru_cache()(f))


def cached(f):
    """Caches a method with no arguments after first call."""
    
    #############
    import warnings
    warnings.warn("deprecated", DeprecationWarning)
    #############
    
    cached_name = '_' + f.__name__
    @functools.wraps(f)
    def wrapper(self):
        if not hasattr(self, cached_name):
            setattr(self, cached_name, f(self))
        return getattr(self, cached_name)
    return wrapper


def double_deriv_name(fun, wrt):
    assert len(wrt) == 2
    if wrt[0] == wrt[1]:
        return f'd2{fun}_d{wrt[0]}2'
    else:
        return f'd2{fun}_d{wrt[0]}_d{wrt[1]}'
