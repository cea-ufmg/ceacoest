'''Pytest plugin for numpy array and array_like comparisons.'''


import numpy as np


def broadcastable(a, b):
    try:
        np.broadcast(a, b)
        return True
    except ValueError:
        return False


class ArrayCmp:
    def __init__(self, array, rtol=1e-8, atol=1e-8, tol=None):
        '''Class for numerical numpy array comparison.'''
        self.array = array
        self.atol = tol or atol
        self.rtol = tol or rtol
    
    def __eq__(self, other):
        '''Test that the arrays are within the tolerance.'''
        return (broadcastable(self.array, other) and
                np.allclose(self.array, other, rtol=self.rtol, atol=self.atol))
    
    def eq_report(self, config, op, left, right):
        if not broadcastable(left, right):
            return ["error: arrays not broadcastable with shapes " +
                    str(np.shape(left)) + " and " + str(np.shape(right)) + "."]
        
        if not np.allclose(left, right, rtol=self.rtol, atol=self.atol):
            summary = "error: arrays not within tolerances rtol={} atol={}:"
            return ([summary.format(self.rtol, self.atol), "", "left="] +
                    np.array_str(left).splitlines() + ["", "right="] +
                    np.array_str(right).splitlines())


def pytest_assertrepr_compare(config, op, left, right):
    '''Return explanation for comparisons in failing ArrayCmp assertions.'''
    if isinstance(left, ArrayCmp) and op == "==":
        return left.eq_report(config, op, left.array, right)

