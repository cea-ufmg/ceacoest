"""Output Error Method estimator."""


import itertools

import numpy as np

from . import collocation, optim, utils


class Problem(collocation.CollocatedProblem):
    """Output error method optimization problem with LGL direct collocation."""
    
    def __init__(self, model, t, y, u):
        super().__init__(model, t)
        npoints = self.npoints
        npieces = self.npieces
        ncoarse = len(t)
        
        assert isinstance(y, np.ndarray)
        assert y.shape == (ncoarse, self.model.ny)
        
        ymask = np.ma.getmaskarray(y)
        kmeas_coarse, = np.nonzero(np.any(~ymask, axis=1))
        self.kmeas = kmeas_coarse * self.collocation.ninterv
        """Collocation time indices with active measurements."""
        
        self.y = y[kmeas_coarse]
        """Measurements at the time indices with active measurements."""
        
        self.nmeas = np.size(self.kmeas)
        """Number of measurement indices."""

        if callable(u):
            u = u(self.tc)
        assert isinstance(u, np.ndarray)
        assert u.shape == (self.npoints, model.nu)
        self.u = u
        """The inputs at the fine grid points."""
        
        self.um = self.u[self.kmeas]
        """The inputs at the measurement points."""

        up = np.zeros((npieces, self.collocation.n, model.nu))
        up[:, :-1].flat = u[:-1, :].flat
        up[:-1, -1] = up[1:, 0]
        up[-1, -1] = u[-1]
        self.up = up
        """Piece-ravelled inputs."""
        
        self.register_derived('xm', XMVariable(self))
        self._register_model_constraint_derivatives('e', ('xp', 'p'))
        
        self.register_merit('L', model.L, ('y', 'xm', 'um', 'p'), self.nmeas)
        self._register_model_merit_derivatives('L', ('xm', 'p'))
        
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'y': self.y, 'um': self.um, 'u': self.u, 'up': self.up,
                **super().variables(dvec)}


class XMVariable:
    def __init__(self, problem):
        self.p = problem

    @property
    def tiling(self):
        return self.p.nmeas
    
    @property
    def shape(self):
        return (self.p.nmeas, self.p.model.nx)
    
    def build(self, variables):
        x = variables['x']
        return x[self.p.kmeas]
    
    def add_to(self, destination, value):
        assert np.shape(value) == self.shape
        x = self.p.decision['x'].unpack_from(destination)
        x[self.p.kmeas] += value
    
    def expand_indices(self, ind):
        nx = self.p.model.nx
        x_offset = self.p.decision['x'].offset
        ind = np.asarray(ind, dtype=int)        
        return ind + x_offset + self.p.kmeas[:, None]
