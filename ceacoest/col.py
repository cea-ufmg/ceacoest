"""Common code for collocated optimization problems."""


import itertools

import numpy as np

from . import optim, rk, utils


class Problem(optim.Problem):
    """Collocated optimization problem base."""
    
    def __init__(self, model, t):
        # Initialize base class
        super().__init__()
        
        self.model = model
        """Underlying model."""
        
        col = rk.LGLCollocation(model.collocation_order)
        self.collocation = col
        """Collocation method."""
        
        assert np.ndim(t) == 1
        self.piece_len = np.diff(t)
        """Normalized length of each collocation piece."""
        
        assert np.all(self.piece_len > 0)
        npieces = len(self.piece_len)
        self.npieces = npieces
        """Number of collocation pieces."""
        
        self.tc = self.collocation.grid(t)
        """Normalized collocation time grid."""
        
        npoints = self.tc.size
        self.npoints = npoints
        """Total number of collocation points."""
        
        x = self.add_decision('x', (npoints, model.nx))
        self.remapped['xp'] = PieceRavelledVariable(x, npieces, col.n)
        
        self.add_constraint(model.e, (npieces, col.ninterv, model.nx))
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'piece_len': self.piece_len, **super().variables(dvec)}


class PieceRavelledVariable:
    def __init__(self, unravelled, npieces, ncol):
        self.unravelled = unravelled
        """Specification of the underlying unravelled variable."""
        
        self.npieces = npieces
        """Number of collocation pieces."""
        
        self.ncol = ncol
        """Number of collocation points per piece."""
    
    @property
    def shape(self):
        """The variable's ndarray shape."""
        return (self.npieces, self.ncol) + self.unravelled.shape[1:]
    
    @property
    def size(self):
        """Total number of elements."""
        return np.prod(self.shape, dtype=np.intc)
    
    def unpack_from(self, vec):
        """Extract component from parent vector."""
        ur = self.unravelled.unpack_from(vec)
        out = np.zeros(self.shape)
        out[:, :-1].flat = ur[:-1].flat
        out[:-1, -1] = out[1:, 0]
        out[-1, -1] = ur[-1]
        return out
    
    def add_to(self, destination, value):
        value = np.asarray(value)
        assert value.shape == self.shape
        
        ninterv = self.ncol - 1
        dec = np.zeros(self.unravelled.shape)
        dec[:-1].flat = value[:, :-1].flat
        dec[ninterv::ninterv] += value[:,-1]
        self.unravelled.add_to(destination, dec)
    
    def convert_ind(self, rav_ind):
        """Convert component indices to parent vector indices."""
        rav_ind = np.asarray(rav_ind, dtype=int)
        piece = rav_ind // np.prod(self.shape[1:], dtype=int)
        ur_ind = rav_ind - piece * np.prod(self.shape[2:], dtype=int)
        return self.unravelled.convert_ind(ur_ind)
