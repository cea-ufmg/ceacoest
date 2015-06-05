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
            return ["arrays not broadcastable with shapes " +
                    str(np.shape(left)) + " and " + str(np.shape(right)) + "."]
        
        if not np.allclose(left, right, rtol=self.rtol, atol=self.atol):
            summary = "arrays not within tolerances rtol={} atol={}:"
            return ([summary.format(self.rtol, self.atol), "", "left="] +
                    np.array_str(left).splitlines() + ["", "right="] +
                    np.array_str(right).splitlines())


class ArrayDiff:
    def __init__(self, a, b):        
        self.a = a
        self.b = b
    
    def within_tol(self, atol=0, rtol=0, tol=None):
        # Process input arguments
        if tol is not None:
            atol = rtol = tol
    
        # Convert matrices to ndarray
        try:
            self.a = np.asarray(self.a)
            self.b = np.asarray(self.b)
        except (TypeError, ValueError):
            pass
        
        self.a_array_convertible = isinstance(self.a, np.ndarray)
        self.b_array_convertible = isinstance(self.b, np.ndarray)
        self.broadcastable = broadcastable(self.a, self.b)
        if not (self.broadcastable and self.a_array_convertible and
                self.b_array_convertible):
            return False
        
        self.atol = atol
        self.rtol = rtol
        self.err = np.subtract(self.a, self.b)        
        self.ok = np.abs(self.err) < atol + rtol * np.abs(self.b)
        return np.all(self.ok)
    
    def __lt__(self, other):
        if isinstance(other, dict):
            return self.within_tol(**other)
        elif isinstance(other, (int, float)):
            return self.within_tol(tol=other)
        else:
            raise ValueError("Invalid class for comparison.")

    def report(self, config):
        if not self.a_array_convertible:
            return ["a not convertible to ndarray."]
        if not self.b_array_convertible:
            return ["b not convertible to ndarray."]
        if not self.broadcastable:
            a_shape = np.shape(self.a)
            b_shape = np.shape(self.b)
            return ["arrays not broadcastable with shapes " +
                    str(a_shape) + " and " + str(b_shape) + "."]
        
        summary = "arrays not within tolerances rtol={} and atol={}"
        violations = [str(tup) for tup in zip(*np.nonzero(~self.ok))]
        return ([summary.format(self.rtol, self.atol)] + 
                ["", "a="] + np.array_str(self.a).splitlines() +
                ["", "b="] + np.array_str(self.b).splitlines() +
                ["", "a-b="] + np.array_str(self.err).splitlines() +
                ["", "violations: ", " ".join(violations)])


def pytest_assertrepr_compare(config, op, left, right):
    '''Return explanation for comparisons in failing ArrayCmp assertions.'''
    if isinstance(left, ArrayCmp) and op == "==":
        return left.eq_report(config, op, left.array, right)
    elif isinstance(left, ArrayDiff) and op == "<":
        return left.report(config)

