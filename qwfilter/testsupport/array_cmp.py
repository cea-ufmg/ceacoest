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
        return (is_float_convertible(self.array) and 
                is_float_convertible(other) and 
                broadcastable(self.array, other) and
                np.allclose(self.array, other, rtol=self.rtol, atol=self.atol))
    
    def eq_report(self, config, op, left, right):
        try:
            left = np.require(left, dtype=float)
        except TypeError:
            return (["left argument not convertible to numerical array:"] +
                    repr(left).splitlines())
        
        try:
            right = np.require(right, dtype=float)
        except TypeError:
            return (["right argument not convertible to numerical array:"] +
                    repr(right).splitlines())
        
        if not broadcastable(left, right):
            return ["error: arrays not broadcastable with shapes " +
                    str(left.shape) + " and " + str(right.shape) + "."]
        
        if not np.allclose(left, right, rtol=self.rtol, atol=self.atol):
            summary = "error: arrays not within tolerances rtol={} atol={}:"
            return ([summary.format(self.rtol, self.atol), "", "left="] +
                    np.array_str(left).splitlines() + ["", "right="] +
                    np.array_str(right).splitlines())


def pytest_assertrepr_compare(config, op, left, right):
    '''Return explanation for comparisons in failing ArrayCmp assertions.'''
    if isinstance(left, ArrayCmp) and op == "==":
        return left.eq_report(config, op, left.array, right)

