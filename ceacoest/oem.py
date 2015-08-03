"""Output Error Method estimation."""


import numpy as np
from numpy import ma
from scipy import sparse

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
        
        t = np.asarray(t) #Piece boundary times
        assert t.shape == y.shape[:1]
        
        self.collocation = rk.LGLCollocation(order)
        """Collocation method."""
        
        self.tc = self.collocation.grid(t)
        """Collocation time grid."""
        
        ymask = ma.getmaskarray(y)
        kmp, = np.nonzero(np.any(~ymask, axis=1))
        self.km = kmp * self.collocation.ninterv
        """Collocation time indices with active measurements."""
        
        self.tm = t[kmp]
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
        
        self.nd = self.tc.size * model.nx + model.nq
        """Length of the decision vector."""

        self.ndefects = self.npieces * self.collocation.ninterv * model.nx

        tr = self.unravel_pieces(self.tc)
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

    def pack_bounds(self, q_lb={}, q_ub={}, q_fix={}):
        """Pack the decision variable bounds."""
        x_lb = np.tile(-np.inf, (self.tc.size, self.model.nx))
        x_ub = np.tile(np.inf, (self.tc.size, self.model.nx))
        q_lb_vec = self.model.pack('q', dict(q_lb, **q_fix), fill=-np.inf)
        q_ub_vec = self.model.pack('q', dict(q_ub, **q_fix), fill=np.inf)
        return (self.pack_decision(x_lb, q_lb_vec),
                self.pack_decision(x_ub, q_ub_vec))
        
    def expand_x_ind(self, xi, k=None):
        xi = np.asarray(xi, dtype=int)
        k = np.arange(self.tc.size) if k is None else np.asarray(k, dtype=int)
        di = self.model.nq + k[:, None] * self.model.nx + xi
        return di
    
    def expand_q_ind(self, qi, k=None):
        nk = self.tc.size if k is None else len(k)
        qi = np.asarray(qi, dtype=int)
        return np.tile(qi, (nk, 1))
    
    def unravel_pieces(self, v):
        v = np.asarray(v)
        assert len(v) == self.tc.size
        ncolinterv = self.collocation.ninterv
        vr_shape = (self.npieces, self.collocation.n) + v.shape[1:]
        vr = np.zeros(vr_shape, dtype=v.dtype)
        vr[:, :-1] = np.reshape(v[:-1], (self.npieces,ncolinterv) + v.shape[1:])
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
        return utils.flat_cat(d2L_dx2, d2L_dq2, d2L_dx_dq)
    
    @property
    @utils.cached
    def merit_hessian_ind(self):
        """Indices of nonzero elements of merit function Hessian."""
        km = self.km
        d2L_dx2 = self.model.d2L_dx2_ind
        d2L_dq2 = self.model.d2L_dq2_ind
        d2L_dx_dq = self.model.d2L_dx_dq_ind
        i = utils.flat_cat(self.expand_x_ind(d2L_dx2[0], km),
                           self.expand_q_ind(d2L_dq2[0], km),
                           self.expand_q_ind(d2L_dx_dq[0], km))
        j = utils.flat_cat(self.expand_x_ind(d2L_dx2[1], km),
                           self.expand_q_ind(d2L_dq2[1], km),
                           self.expand_x_ind(d2L_dx_dq[1], km))
        return (i, j)

    def merit_hessian(self, d):
        """Merit function Hessian as a dense matrix."""
        i, j = self.merit_hessian_ind
        val = self.merit_hessian_val(d)
        offdiag = i != j
        hessian = np.zeros((self.nd, self.nd))
        hessian[i, j] = val
        hessian[j, i] += offdiag * val[offdiag]
        return hessian
    
    def defects(self, d):
        """ODE equality constraints."""
        x, q = self.unpack_decision(d)
        f = self.model.f(self.tc, x, q, self.uc)
        J = self.collocation.J
        dt = self.dt
        
        xr = self.unravel_pieces(x)
        fr = self.unravel_pieces(f)
        delta = xr[:, 1:] - xr[:, :-1]
        finc = np.einsum('ijk,lj,il->ilk', fr, J, dt)
        defects = delta - finc
        return np.ravel(defects)

    def defects_jacobian_val(self, d):
        """Values of nonzero elements of ODE equality constraints Jacobian."""
        x, q = self.unpack_decision(d)
        df_dx = self.model.df_dx_val(self.tc, x, q, self.uc)
        df_dq = self.model.df_dq_val(self.tc, x, q, self.uc)
        J = self.collocation.J
        dt = self.dt
        
        dfr_dx = self.unravel_pieces(df_dx)
        dfr_dq = self.unravel_pieces(df_dq)
        return utils.flat_cat(
            np.ones((self.npieces, self.collocation.ninterv, self.model.nx)),
            -np.ones((self.npieces, self.collocation.ninterv, self.model.nx)),
            -np.einsum('ijk,lj,il->ijkl', dfr_dx, J, dt),
            -np.einsum('ijk,lj,il->ijkl', dfr_dq, J, dt)
        )
    
    @property
    @utils.cached
    def defects_jacobian_ind(self):
        """Indices of nonzero elements of ODE equality constraints Jacobian."""
        x_ind = np.arange(self.model.nx)
        df_dx_i, df_dx_j = self.model.df_dx_ind
        df_dq_i, df_dq_j = self.model.df_dq_ind

        offsets = np.arange(self.npieces * self.collocation.ninterv)
        offsets = offsets.reshape((self.npieces, self.collocation.ninterv, 1))
        offsets *= self.model.nx
        dxr_dx_i = self.unravel_pieces(self.expand_x_ind(x_ind))
        dfr_dx_i = self.unravel_pieces(self.expand_x_ind(df_dx_i))
        dfr_dq_i = self.unravel_pieces(self.expand_q_ind(df_dq_i))
        dxr_dx_j = offsets + x_ind
        dfr_dx_j = np.swapaxes(offsets + df_dx_j, 1, 2)
        dfr_dq_j = np.swapaxes(offsets + df_dq_j, 1, 2)
        i = utils.flat_cat(dxr_dx_i[:, 1:], dxr_dx_i[:, :-1],
                           np.repeat(dfr_dx_i, self.collocation.ninterv),
                           np.repeat(dfr_dq_i, self.collocation.ninterv))
        j = utils.flat_cat(dxr_dx_j, dxr_dx_j,
                           np.repeat(dfr_dx_j, self.collocation.n, axis=0),
                           np.repeat(dfr_dq_j, self.collocation.n, axis=0))
        return (i, j)
        
    def defects_jacobian(self, d):
        """ODE equality constraints Jacobian."""
        val = self.defects_jacobian_val(d)
        ind = self.defects_jacobian_ind
        shape = (self.nd, self.ndefects)
        return sparse.coo_matrix((val, ind), shape=shape).todense()
    
    def defects_hessian_val(self, d):
        """Values of nonzero elements of ODE equality constraints Hessian."""
        x, q = self.unpack_decision(d)
        d2f_dx2 = self.model.d2f_dx2_val(self.tc, x, q, self.uc)
        d2f_dq2 = self.model.d2f_dq2_val(self.tc, x, q, self.uc)
        d2f_dx_dq = self.model.d2f_dx_dq_val(self.tc, x, q, self.uc)
        J = self.collocation.J
        dt = self.dt
        
        d2fr_dx2 = self.unravel_pieces(d2f_dx2)
        d2fr_dq2 = self.unravel_pieces(d2f_dq2)
        d2fr_dx_dq = self.unravel_pieces(d2f_dx_dq)
        return utils.flat_cat(
            -np.einsum('ijk,lj,il->ijkl', d2fr_dx2, J, dt),
            -np.einsum('ijk,lj,il->ijkl', d2fr_dq2, J, dt),
            -np.einsum('ijk,lj,il->ijkl', d2fr_dx_dq, J, dt)
        )
    
    @property
    @utils.cached
    def defects_hessian_ind(self):
        """Indices of nonzero elements of ODE equality constraints Hessian."""
        ncolpt = self.collocation.n
        ncolinterv = self.collocation.ninterv
        d2f_dx2_i, d2f_dx2_j, d2f_dx2_k = self.model.d2f_dx2_ind
        d2f_dq2_i, d2f_dq2_j, d2f_dq2_k = self.model.d2f_dq2_ind
        d2f_dx_dq_i, d2f_dx_dq_j, d2f_dx_dq_k = self.model.d2f_dx_dq_ind
        d2f_dx2_k = np.asarray(d2f_dx2_k, dtype=int)
        d2f_dq2_k = np.asarray(d2f_dq2_k, dtype=int)
        d2f_dx_dq_k = np.asarray(d2f_dx_dq_k, dtype=int)
        
        offsets = np.arange(self.npieces * self.collocation.ninterv)
        offsets = offsets.reshape((self.npieces, self.collocation.ninterv, 1))
        offsets *= self.model.nx
        d2fr_dx2_i = self.unravel_pieces(self.expand_x_ind(d2f_dx2_i))
        d2fr_dq2_i = self.unravel_pieces(self.expand_q_ind(d2f_dq2_i))
        d2fr_dx_dq_i = self.unravel_pieces(self.expand_q_ind(d2f_dx_dq_i))
        d2fr_dx2_j = self.unravel_pieces(self.expand_x_ind(d2f_dx2_j))
        d2fr_dq2_j = self.unravel_pieces(self.expand_q_ind(d2f_dq2_j))
        d2fr_dx_dq_j = self.unravel_pieces(self.expand_x_ind(d2f_dx_dq_j))
        d2fr_dx2_k = np.swapaxes(offsets + d2f_dx2_k, 1, 2)
        d2fr_dq2_k = np.swapaxes(offsets + d2f_dq2_k, 1, 2)
        d2fr_dx_dq_k = np.swapaxes(offsets + d2f_dx_dq_k, 1, 2)
        i = utils.flat_cat(np.repeat(d2fr_dx2_i, ncolinterv),
                           np.repeat(d2fr_dq2_i, ncolinterv),
                           np.repeat(d2fr_dx_dq_i, ncolinterv))
        j = utils.flat_cat(np.repeat(d2fr_dx2_j, ncolinterv),
                           np.repeat(d2fr_dq2_j, ncolinterv),
                           np.repeat(d2fr_dx_dq_j, ncolinterv))
        k = utils.flat_cat(np.repeat(d2fr_dx2_k, ncolpt, 0),
                           np.repeat(d2fr_dq2_k, ncolpt, 0),
                           np.repeat(d2fr_dx_dq_k, ncolpt, 0))
        return (i, j, k)

    def defects_hessian(self, d):
        """ODE equality constraints Hessian as a dense matrix."""
        val = self.defects_hessian_val(d)
        hessian = np.zeros((self.nd, self.nd, self.ndefects))
        for v, i, j, k in zip(val, *self.defects_hessian_ind):
            hessian[i, j, k] += v
            if i != j:
                hessian[j, i, k] += v
        return hessian
    
    def nlp_yaipopt(self, d_bounds=None):
        import yaipopt
        
        if d_bounds is None:
            d_bounds = np.tile([[-np.inf], [np.inf]], self.nd)
        
        constr_bounds = np.zeros((2, self.ndefects))
        constr_jac_ind = self.defects_jacobian_ind[::-1]
        merit_hess_ind = self.merit_hessian_ind
        def_hess_ind = self.defects_hessian_ind
        hess_inds = np.hstack((merit_hess_ind[::-1], def_hess_ind[1::-1]))
        
        def merit(d, new_d=True):
            return self.merit(d)
        def grad(d, new_d=True):
            return self.merit_gradient(d)
        def constr(d, new_d=True):
            return self.defects(d)
        def constr_jac(d, new_d=True):
            return self.defects_jacobian_val(d)
        def lag_hess(d, new_d, obj_factor, lmult, new_lmult):
            def_hess = self.defects_hessian_val(d) * lmult[def_hess_ind[2]]
            merit_hess = self.merit_hessian_val(d) * obj_factor
            return utils.flat_cat(merit_hess, def_hess)        
        
        nlp = yaipopt.Problem(d_bounds, merit, grad,
                              constr_bounds=constr_bounds,
                              constr=constr, constr_jac=constr_jac,
                              constr_jac_inds=constr_jac_ind,
                              hess=lag_hess, hess_inds=hess_inds)
        nlp.num_option('obj_scaling_factor', -1)
        return nlp
