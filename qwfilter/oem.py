"""Output Error Method estimation."""


import numpy as np
from numpy import ma

from . import rk


class CTEstimator:
    """Continuous-time output error method estimator using LGL collocation."""
    
    def __init__(self, model, y, t, **options):
        self.model = model
        """Underlying dynamical system model."""
        
        self.y = ma.asarray(y)
        """Measurements."""
        assert y.ndim == 2 and y.shape[1] == model.ny
        
        self.t_piece = np.asarray(t)
        """Piece boundary times."""
        assert self.t_piece.shape == y.shape[:1]
        
        order = options.get('order', 2)
        self.collocation = rk.LGLCollocation(order)
        """Collocation method."""
        
        self.t_col = self.collocation.grid(t_piece)
        """Collocation time grid."""
        
        yactive = ma.getmaskarray(y)
        k_meas_pieces, = np.nonzero(np.any(yactive, axis=1))
        self.k_meas = k_meas_pieces * self.collocation.ninterv
        """Collocation time indices with active measurements."""
        
        self.yv = ma.getdata(y[k_meas_pieces])
        """Measurement values (unmasked)."""
        
        self.ya = ~ma.getmaskarray(y[k_meas_pieces])
        """Active outputs at the measument indices."""

        self.ncol = len(self.t_col)
        """Number of collocation points."""
        
        self.nd = self.ncol * model.nx + model.nq
        """Length of the decision vector."""
    
    def unpack_decision(self, d):
        """Unpack the decision vector into the states and parameters."""
        assert d.shape == (self.nd,)
        
        q = d[:self.model.nq]
        x = np.reshape(d[self.model.nq:], (-1, self.model.nx))
        return x, q
        
    def pack_decision(self, x, q):
        """Pack the states and parameters into the decision vector."""
        assert x.shape == (self.ncol, self.model.nx)
        assert q.shape == (self.model.nq,)
        
        d = np.empty(self.nd)
        d[:self.model.nq] = q
        d[self.model.nq:] = np.flatten(x)
        return d

    def merit(self, d):
        x, q = self.unpack_decision(d)
        
