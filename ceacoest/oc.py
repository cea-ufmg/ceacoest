"""Optimal control."""


import itertools

import numpy as np

from . import col, optim, utils


class Problem(col.Problem):
    """Optimal control problem with LGL direct collocation."""

    def __init__(self, model, t):
        super().__init__(model, t)

        ncol = self.collocation.n
        npoints = self.npoints
        npieces = self.npieces
        
        # Register decision variables
        x = self.decision['x']
        u = self.add_decision('u', (npoints, model.nu))
        self.add_decision('p', model.np)
        
        # Define and add dependent variables
        up = col.PieceRavelledVariable(u, npieces, ncol)
        self.add_dependent_variable('up', up)
        self.add_dependent_variable('xe', XEVariable(x))
        
        # Register problem functions
        self.add_objective(model.IL, npieces)
        self.add_objective(model.M, ())
        self.add_constraint(model.g, (npoints, model.ng))
        self.add_constraint(model.h, model.nh)


class XEVariable:
    """Endpoint states."""
    
    def __init__(self, x):
        self.x = x
        """Specification of the state (x) variable."""
    
    @property
    def shape(self):
        nx = self.x.shape[-1]
        return (2, nx)
    
    @property
    def size(self):
        """Total number of elements."""
        return np.prod(self.shape, dtype=np.intc)
    
    def unpack_from(self, vec):
        """Extract component from parent vector."""
        x = self.x.unpack_from(vec)
        return x[[0, -1]]
    
    def add_to(self, destination, value):
        value = np.asarray(value)
        assert value.shape == self.shape
        
        xval = np.zeros(self.x.shape)
        xval[0] = value[0]
        xval[-1] = value[-1]
        self.x.add_to(destination, xval)
    
    def convert_ind(self, xe_ind):
        xe_ind = np.asarray(xe_ind, int)
        npoints, nx = self.x.shape
        x_final = xe_ind >= nx
        x_ind = xe_ind + x_final * (npoints - 2) * nx
        return self.x.convert_ind(x_ind)
