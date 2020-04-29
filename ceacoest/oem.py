"""Output Error Method estimator."""


import itertools

import numpy as np

from . import col, optim


class Problem(col.Problem):
    """Output error method optimization problem with LGL direct collocation."""
    
    def __init__(self, model, t, y, u):
        super().__init__(model, t)
        npoints = self.npoints
        npieces = self.npieces
        ncoarse = len(t)
        
        y = np.asanyarray(y)
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

        # Register problem variables
        self.add_decision('p', model.np)
        self.remapped['xm'] = XMVariable(self.decision['x'], self.kmeas)
        
        # Add objective function
        self.add_objective(model.L, self.nmeas, ['y', 'xm', 'um', 'p'])
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'y': self.y, 'um': self.um, 'u': self.u, 'up': self.up,
                **super().variables(dvec)}


class XMVariable:
    def __init__(self, x, kmeas):
        self.x = x
        """Description of x variable."""
        
        self.kmeas = kmeas
        """Collocation time indices with active measurements."""        
        
        self.nmeas = len(kmeas)
        """Number of active measurement instants."""

    @property
    def shape(self):
        """The variable's ndarray shape."""
        return (self.nmeas, self.x.shape[1])
        
    @property
    def size(self):
        """Total number of elements."""
        return np.prod(self.shape, dtype=np.int)
    
    def unpack_from(self, vec):
        """Extract component from parent vector."""
        x = self.x.unpack_from(vec)
        return x[self.kmeas]

    def add_to(self, destination, value):
        value = np.asarray(value)
        assert value.shape == self.shape
        
        xval = np.zeros(self.x.shape)
        xval[self.kmeas] = value
        self.x.add_to(destination, xval)

    def convert_ind(self, xm_ind):
        """Convert component indices to parent vector indices."""
        nx = self.x.shape[1]
        xm_coord = np.unravel_index(xm_ind, self.shape)
        x_coord = (self.kmeas[xm_coord[0]], xm_coord[1])
        x_ind = np.ravel_multi_index(x_coord, self.x.shape)
        return self.x.convert_ind(x_ind)

