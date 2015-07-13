"""Output Error Method estimation."""


import numpy as np
from numpy import ma
from scipy import interpolate

from . import rk


class CTEstimator:
    """Continuous-time output error method estimator using LGL collocation."""
    
    def __init__(self, model, y, t, u, **options):
        order = options.get('order', 2)
        
        self.model = model
        """Underlying dynamical system model."""
        
        self.y = np.asanyarray(y)
        """Measurements."""
        assert y.ndim == 2 and y.shape[1] == model.ny
        
        self.tp = np.asarray(t)
        """Piece boundary times."""
        assert self.tp.shape == y.shape[:1]
        
        self.collocation = rk.LGLCollocation(order)
        """Collocation method."""
        
        self.tc = self.collocation.grid(self.tp)
        """Collocation time grid."""
        
        ymask = ma.getmaskarray(y)
        kmp, = np.nonzero(np.any(~ymask, axis=1))
        self.km = kmp * self.collocation.ninterv
        """Collocation time indices with active measurements."""
        
        self.tm = self.tp[kmp]
        """Times of active measurements."""        
        
        self.ym = y[kmp]
        """Measurements at the time indices with active measurements."""
        
        self.uc = u(self.tc)
        """Inputs at the colocation grid."""

        self.um = self.uc[self.km]
        """Inputs at the times of active measuments."""
        
        self.npiece = len(t) - 1
        """Number of collocation pieces."""
        
        self.ncol = len(self.tc)
        """Number of collocation points."""
        
        self.nd = self.ncol * model.nx + model.nq
        """Length of the decision vector."""

        tr = self.ravel_pieces(self.tc)
        self.dt = tr[:, 1:] - tr[:, :-1]
        """Length of each collocation interval."""
    
    def unpack_decision(self, d):
        """Unpack the decision vector into the states and parameters."""
        assert d.shape == (self.nd,)
        
        q = d[:self.model.nq]
        x = np.reshape(d[self.model.nq:], (-1, self.model.nx))
        return x, q
        
    def pack_decision(self, x, q):
        """Pack the states and parameters into the decision vector."""
        assert np.shape(x) == (self.ncol, self.model.nx)
        assert np.shape(q) == (self.model.nq,)
        
        d = np.empty(self.nd)
        d[:self.model.nq] = q
        d[self.model.nq:] = np.ravel(x)
        return d
    
    def ravel_pieces(self, v):
        assert len(v) == self.ncol
        vr = np.zeros((self.npiece, self.collocation.n) + v.shape[1:])
        vr[:, :-1].flat = v[::-1].flat
        vr[:-1, -1] = vr[1:, 0]
        vr[-1, -1] = v[-1]
        return vr
    
    def merit(self, d):
        """Merit function."""
        x, q = self.unpack_decision(d)
        xm = x[self.km]
        L = self.model.L(self.ym, self.tm, xm, q, self.um)
        return L.sum()
    
    def defects(self, d):
        """ODE equality constraints."""
        x, q = self.unpack_decision(d)
        f = self.model.f(self.tc, x, q, self.uc)
        J = self.collocation.J
        dt = self.dt
        
        xr = self.ravel_pieces(x)
        fr = self.ravel_pieces(f)
        delta = xr[:, 1:] - xr[:, :-1]
        finc = np.einsum('ijk,lj,il->ilk', fr, J, dt)
        defects = delta - finc
        return np.ravel(defects)
