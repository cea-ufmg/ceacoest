"""Symbolic model building for output error method (OEM) estimation."""


import collections
import functools

import numpy as np
import sym2num.model
import sympy

from . import symcol


class Model(symcol.Model):
    def __init__(self, variables):
        variables.setdefault('u', [])
        decision = {'p'}
        super().__init__(variables, decision)
        
        # Add objectives and constraints
        self.add_objective('L')
    
    @property
    def generate_assignments(self):
        gen = {'ny': len(self.variables['y']),
               **getattr(super(), 'generate_assignments', {})}
        return gen
