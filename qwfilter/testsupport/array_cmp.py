'''Pytest plugin for numpy array and array_like comparisons.'''


import numpy as np


def broadcastable(a, b):
    try:
        np.broadcast(a, b)
        return True
    except ValueError:
        return False


def is_float_convertible(x):
    try:
        np.require(x, dtype=float)
        return True
    except TypeError:
        return False


class ArrayCmp:
    def __init__(self, array, rtol=1e-8, atol=1e-8):
        '''Class for numerical numpy array comparison.'''
        self.array = array
        self.atol = atol
        self.rtol = rtol
    
    def __eq__(self, other):
        '''Test that the arrays are within the tolerance.'''
        return (broadcastable(self.array, other) and 
                is_float_convertible(self.array) and 
                is_float_convertible(other) and 
                np.allclose(self.array, other, rtol=self.rtol, atol=self.atol))

