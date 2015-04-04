'''Test various modules using a Duffing oscillator SDE model.'''


import numpy as np
import sympy
import sym2num

from qwfilter import sde


class SymbolicDuffing(sde.EulerDiscretizedModel):
    '''Symbolic Duffing oscillator model.'''

    var_names = ['t', 'x', 'y', 'q', 'c', 'td', 'k']
    '''Name of model variables.'''
    
    function_names = ['f', 'g', 'h']
    '''Name of the model functions.'''
    
    t = 't'
    '''Time variable.'''

    k = 'k'
    '''Discretized time index.'''
    
    td = 'Td'
    '''Discretization period.'''
    
    x = ['x', 'v']
    '''State vector.'''
    
    y = ['x_meas']
    '''Measurement vector.'''
    
    q = ['alpha', 'beta', 'delta', 'g1']
    '''Unknown parameter vector.'''
    
    c = ['gamma', 'omega']
    '''Constants vector.'''
    
    def f(self, t, x, q, c):
        '''Drift function.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        f1 = s.v
        f2 = (-s.delta * s.v - s.beta * s.x - s.alpha * s.x ** 3  +
              s.gamma * sympy.cos(s.t * s.omega))
        return [f1, f2]

    def g(self, t, x, q, c):
        '''Diffusion matrix.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        return [[0], [s.g1]]
    
    def h(self, t, x, q, c):
        '''Measurement function.'''
        s = self.symbols(t, x, q, c)
        return [s.x]


GeneratedDuffing = sym2num.class_obj(
    SymbolicDuffing(), sym2num.ScipyPrinter(),
    name='GeneratedDuffing', 
    meta=sym2num.ParametrizedModel.meta
)


