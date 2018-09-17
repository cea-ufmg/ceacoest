"""Optimal control."""


import itertools

import numpy as np

from . import collocation, optim, utils


class Problem(collocation.CollocatedProblem):
    """Optimal control problem with LGL direct collocation."""
    
    def __init__(self, model, t):
        super().__init__(model, t)
        npoints = self.npoints
        npieces = self.npieces
        
        u = self.register_decision('u', model.nu, npoints)
        self.register_derived('up', collocation.PieceRavelledVariable(self,'u'))
        self.register_derived('xe', XEVariable(self))
        
        self.register_constraint('g', model.g, ('x','u','p'), model.ng, npoints)
        self.register_constraint('h', model.h, ('xe', 'p'), model.nh)
        self._register_model_constraint_derivatives('g', ('x', 'u', 'p'))
        self._register_model_constraint_derivatives('h', ('xe', 'p'))
        self._register_model_constraint_derivatives('e', ('xp', 'up', 'p'))
        
        self.register_merit('M', model.M, ('xe', 'p'))
        self.register_merit(
            'IL', model.IL, ('xp', 'up', 'p', 'piece_len'), npieces
        )
        self._register_model_merit_derivatives('M', ('xe', 'p'))
        self._register_model_merit_derivatives('IL', ('xp', 'up', 'p'))


class XEVariable:
    def __init__(self, problem):
        self.p = problem

    @property
    def shape(self):
        return (2, self.p.model.nx)
    
    def build(self, variables):
        x = variables['x']
        return x[::self.p.npoints-1]
    
    def add_to(self, destination, value):
        assert np.shape(value) == self.shape
        x = self.p.decision['x'].unpack_from(destination)
        x[::self.p.npoints-1] += value
    
    def expand_indices(self, ind):
        nx = self.p.model.nx
        x_offset = self.p.decision['x'].offset
        ind = np.asarray(ind, dtype=int)
        return ind + x_offset + (ind >= nx) * (self.p.npoints - 2) * nx
