"""Symbolic model building for optimal control."""


import collections
import functools

import numpy as np
import sym2num.model
import sym2num.var
import sympy

from . import symcol
from .. import utils


class Model(symcol.Model):
    def __init__(self, variables, decision=set()):
        variables.setdefault('p', [])
        variables.setdefault('u', [])
        decision = {'p', 'u', 'up', *decision}
        super().__init__(variables, decision)
        
        # Define endpoint variables
        x = self.variables['x']
        xe = [[f'{n}_initial' for n in x], [f'{n}_final' for n in x]]
        self.variables['xe'] = xe
    
        # Add objectives and constraints
        self.add_objective('IL')
        self.add_objective('M')
        self.add_constraint('g')
        self.add_constraint('h')
    
    def IL(self, xp, up, p, piece_len):
        """Integral of the Lagrangian (total running cost)."""
        ncol = self.collocation.n
        Lp = np.array([self.L(xp[i,:], up[i,:], p) for i in range(ncol)])
        K = self.collocation.K
        dt = piece_len
        IL = Lp @ K * piece_len
        return IL
    
    def g(self, x, u, p):
        """Default (empty) path constraint."""
        return np.array([])
    
    def h(self, xe, p):
        """Default (empty) endpoint constraint."""
        return np.array([])
    
    def L(self, x, u, p):
        """Default (null) Lagrange (running) cost."""
        return np.array(0)
    
    def M(self, xe, p):
        """Default (null) Mayer (endpoint) cost."""
        return np.array(0)

    @property
    def generate_assignments(self):
        gen = {'ng': len(self.default_function_output('g')),
               'nh': len(self.default_function_output('h')),
               **getattr(super(), 'generate_assignments', {})}
        return gen
