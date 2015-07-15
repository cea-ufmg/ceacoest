"""Output Error Method estimation."""


import numpy as np
from numpy import ma
from scipy import interpolate

from . import rk, utils


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
        
        self.npieces = len(t) - 1
        """Number of collocation pieces."""
        
        self.ncol = self.collocation.n
        """Number of collocation points per piece."""
        
        self.nd = len(self.tc) * model.nx + model.nq
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
        assert np.shape(x) == (self.tc.size, self.model.nx)
        assert np.shape(q) == (self.model.nq,)
        
        d = np.empty(self.nd)
        d[:self.model.nq] = q
        d[self.model.nq:] = np.ravel(x)
        return d

    def pack_x_ind(self, xi, k):
        xi = np.asarray(xi, dtype=int)
        k = np.asarray(k, dtype=int)
        di = self.model.nq + k[:, None] * self.model.nx + xi
        return di

    def pack_q_ind(self, qi, k):
        qi = np.asarray(qi, dtype=int)
        return np.tile(qi, (len(k), 1))
    
    def ravel_pieces(self, v):
        assert len(v) == self.tc.size
        vr = np.zeros((self.npieces, self.collocation.n) + v.shape[1:])
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

    def merit_gradient(self, d):
        """Gradient of merit function."""
        x, q = self.unpack_decision(d)
        xm = x[self.km]
        dM_dx = np.zeros_like(x)
        dM_dx[self.km] = self.model.dL_dx(self.ym, self.tm, xm, q, self.um)
        dM_dq = self.model.dL_dq(self.ym, self.tm, xm, q, self.um).sum(axis=0)
        return self.pack_decision(dM_dx, dM_dq)
    
    def merit_hessian_val(self, d):
        """Values of nonzero elements of merit function Hessian."""
        x, q = self.unpack_decision(d)
        xm = x[self.km]
        d2L_dx2 = self.model.d2L_dx2_val(self.ym, self.tm, xm, q, self.um)
        d2L_dq2 = self.model.d2L_dq2_val(self.ym, self.tm, xm, q, self.um)
        d2L_dx_dq = self.model.d2L_dx_dq_val(self.ym, self.tm, xm, q, self.um)
        return np.concatenate(
            (d2L_dx2.ravel(), d2L_dq2.ravel(), d2L_dx_dq.ravel())
        )
    
    @property
    def merit_hessian_ind(self):
        """Indices of nonzero elements of merit function Hessian."""
        if not hasattr(self, '_merit_hessian_ind'):
            km = self.km
            d2L_dx2 = self.model.d2L_dx2_ind
            d2L_dq2 = self.model.d2L_dq2_ind
            d2L_dx_dq = self.model.d2L_dx_dq_ind
            i = utils.flat_cat(self.pack_x_ind(d2L_dx2[0], km),
                               self.pack_q_ind(d2L_dq2[0], km),
                               self.pack_q_ind(d2L_dx_dq[0], km))
            j = utils.flat_cat(self.pack_x_ind(d2L_dx2[1], km),
                               self.pack_q_ind(d2L_dq2[1], km),
                               self.pack_x_ind(d2L_dx_dq[1], km))
            self._merit_hessian_ind = (i, j)
        return self._merit_hessian_ind
    
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

    def defects_jacobian_val(self, d):
        x, q = self.unpack_decision(d)
        df_dx = self.model.df_dx_val(self.tc, x, q, self.uc)
        df_dq = self.model.df_dq_val(self.tc, x, q, self.uc)
        J = self.collocation.J
        dt = self.dt
        
        dfr_dx = self.ravel_pieces(df_dx)
        dfr_dq = self.ravel_pieces(df_dq)
        return utils.flat_cat(
            np.ones((self.npieces, self.collocation.ninterv, self.model.nx)),
            -np.ones((self.npieces, self.collocation.ninterv, self.model.nx)),
            np.einsum('ijk,lj,il->ijlk', dfr_dx, J, dt),
            np.einsum('ijk,lj,il->ijlk', dfr_dq, J, dt)
        )

        
