'''Test various modules using a Duffing oscillator SDE model.'''


import numpy as np
import sympy
import sym2num.model

from qwfilter import sde


class SymbolicDuffing(sym2num.model.SymbolicModel):
    '''Symbolic Duffing oscillator model.'''
    
    t = 't'
    '''Time variable.'''
    
    x = ['x', 'v']
    '''State vector.'''
    
    y = ['x_meas']
    '''Measurement vector.'''
    
    q = ['alpha', 'beta', 'delta']
    '''Unknown parameter vector.'''
    
    c = ['gamma', 'omega']
    '''Constants vector.'''
    
    def f(self, t, x, q, c):
        '''Drift function.'''
        s = self.symbols(t, x, q, c)
        f1 = s.xd
        f2 = (-s.delta * s.xd - s.beta * s.x - s.alpha * s.x ** 3  +
              s.gamma * sympy.cos(s.t * s.omega))
        return [f1, f2]
    
    def h(self, t, x, q, c):
        '''Measurement function.'''
        s = self.symbols(t, x, q, c)
        return [s.x]
