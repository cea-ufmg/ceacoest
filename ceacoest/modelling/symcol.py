"""Symbolic collocation model common code."""


import collections
import functools

import numpy as np
import sym2num.model
import sym2num.var
import sympy

from . import symoptim
from .. import utils, rk


class Model(symoptim.Model):
    """Symbolic LGL-collocation model base."""
    
    Variables = sym2num.model.Variables

    def __init__(self, variables, decision=set()):
        # Initialize base class
        super().__init__()

        # Register decision variables
        self.decision.update(decision)
        self.decision.update({'x', 'xp'})
        
        # Register variables given in constructor
        v = self.variables
        for varname, varspec in variables.items():
            v[varname] = varspec

        # Verify the variables and define defaults
        v.setdefault('piece_len', 'piece_len')
        v.setdefault('p', [])
        u = v.setdefault('u', [])
        x = v.setdefault('x')
        if x is None:
            raise ValueError('state vector "x" missing from model variables')
        
        # Create and register derived variables
        ncol = self.collocation.n
        v['xp'] = [[f'{n}_piece_{k}' for n in x] for k in range(ncol)]
        v['up'] = [[f'{n}_piece_{k}' for n in u] for k in range(ncol)]
        
        # Register collocation constraint
        self.add_constraint('e')
        
        # Mark `f` function for code generation
        self.generate_functions.add('f')
    
    @property
    def collocation_order(self):
        """Order of collocation method."""
        return getattr(super(), 'collocation_order', 2)
    
    @utils.cached_property
    def collocation(self):
        """Collocation method."""
        return rk.LGLCollocation(self.collocation_order)
    
    @property
    def generate_assignments(self):
        """Dictionary of assignments in generated class code."""

        gen = {'nx': len(self.variables['x']),
               'nu': len(self.variables['u']),
               'np': len(self.variables['p']),
               'collocation_order': self.collocation_order,
               **getattr(super(), 'generate_assignments', {})}
        return gen
    
    def e(self, xp, up, p, piece_len):
        """Collocation defects (error)."""
        ncol = self.collocation.n
        fp = np.array([self.f(xp[i], up[i], p) for i in range(ncol)])
        J = self.collocation.J
        
        defects = xp[1:] - xp[:-1] - J @ fp * piece_len
        return defects

