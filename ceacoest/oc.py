"""Optimal control."""


import numpy as np
from numpy import ma
from scipy import sparse

from . import rk, utils


class Problem:
    """Optimal control problem with LGL direct collocation."""
    
    def __init__(self, model, t, **options):
        order = options.get('order', 2)
        
        self.model = model
        """Underlying cost, constraint and dynamical system model."""
        
        self.collocation = rk.LGLCollocation(order)
        """Collocation method."""
        
        self.tc = self.collocation.grid(t)
        """Normalized collocation time grid."""
        assert self.tc[0] == 0 and self.tc[-1] == 1
        
        self.npieces = len(t) - 1
        """Number of collocation pieces."""
        
        self.ncol = self.collocation.n
        """Number of collocation points per piece."""
        
        self.nd = 1 + self.tc.size * (model.nx + model.nu)
        """Length of the decision vector."""
        
        self.nconstr = (self.npieces * self.collocation.ninterv * model.nx +
                        self.tc.size * model.ng + model.nh)
        """Number of NLP constraints."""
        
        self.u_offset = self.tc.size * model.nx
        """Offset of the controls `u` in the decision variable vector."""
        
        self.tf_offset = self.nd - 1
        """Offset of the final time `tf` in the decision variable vector."""
 
        self.g_offset = self.npieces * self.collocation.ninterv * model.nx
        """Offset of the path constraints `g` in the constraint vector."""

        self.h_offset = self.g_offset + self.tc.size * model.ng
        """Offset of the endpoint constraints `h` in the constraint vector."""
        
        tr = self.unravel_pieces(self.tc)
        self.dt = tr[:, 1:] - tr[:, :-1]
        """Normalized length of each collocation interval."""
        
    def unpack_decision(self, d):
        """Unpack the decision vector into the states and parameters."""
        assert d.shape == (self.nd,)
        
        x = d[:self.u_offset].reshape((-1, self.model.nx))
        u = d[self.u_offset:-1].reshape((-1, self.model.nu))
        tf = d[-1]
        return x, u, tf
    
    def pack_decision(self, x, u, tf):
        """Pack the states and parameters into the decision vector."""
        assert np.shape(x) == (self.tc.size, self.model.nx)
        assert np.shape(u) == (self.tc.size, self.model.nu)
        assert np.shape(tf) == ()
        d = np.empty(self.nd)
        d[:self.u_offset] = x.flatten()
        d[self.u_offset:-1] = u.flatten()
        d[-1] = tf
        return d
    
    def pack_bounds(self, x,u, tf):
        """Pack the decision variable bounds."""
        pass
        
    def expand_x_ind(self, xi, k=None):
        xi = np.asarray(xi, dtype=int)
        k = np.arange(self.tc.size) if k is None else np.asarray(k, dtype=int)
        di = k[:, None] * self.model.nx + xi
        return di
    
    def expand_u_ind(self, ui, k=None):
        ui = np.asarray(ui, dtype=int)
        k = np.arange(self.tc.size) if k is None else np.asarray(k, dtype=int)
        di = k[:, None] * self.model.nu + ui + self.u_offset
        return di

    def expand_tf_ind(self, inds):
        return np.repeat(self.tf_offset, len(inds[0]))

    def expand_xe_ind(self, xi):
        xi = np.asarray(xi, dtype=int)
        offset = (self.tc.size - 2) * self.model.nx
        return xi + (xi >= self.model.nx) * offset

    def expand_defects_ind(self, fi):
        """Expand ODE equality constraint indices."""
        deltas = np.arange(self.npieces * self.collocation.ninterv)
        deltas = deltas.reshape((self.npieces, self.collocation.ninterv, 1))
        deltas *= self.model.nx
        return deltas + np.asarray(fi, dtype=int)
    
    def expand_g_ind(self, gi):
        """Expand path constraint indices."""
        gi = np.asarray(gi, dtype=int)
        deltas = self.model.ng * np.arange(self.tc.size)[:, None]
        return deltas + gi + self.g_offset

    def expand_h_ind(self, hi):
        """Expand endpoint constraint indices."""
        hi = np.asarray(hi, dtype=int)
        return hi + self.h_offset

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
        x, u, tf = self.unpack_decision(d)
        xe = x[[0, -1]].flatten()
        return self.model.M(xe, tf)
        
    def merit_gradient(self, d):
        """Gradient of merit function."""
        x, u, tf = self.unpack_decision(d)
        xe = x[[0, -1]].flatten()
        dM_dx = np.zeros_like(x)
        dM_du = np.zeros_like(u)
        dM_dx[[0, -1]] = self.model.dM_dx(xe, tf).reshape((2, -1))
        dM_dt = self.model.dM_dt(xe, tf)
        return self.pack_decision(dM_dx, dM_du, dM_dt)
    
    def merit_hessian_val(self, d):
        """Values of nonzero elements of merit function Hessian."""
        x, u, tf = self.unpack_decision(d)
        xe = x[[0, -1]].flatten()
        d2M_dx2 = self.model.d2M_dx2_val(xe, tf)
        d2M_dx_dt = self.model.d2M_dx_dt_val(xe, tf)
        d2M_dt2 = self.model.d2M_dt2(xe, tf)
        return utils.flat_cat(d2M_dx2, d2M_dx_dt, d2M_dt2)
    
    @property
    @utils.cached
    def merit_hessian_ind(self):
        """Indices of nonzero elements of merit function Hessian."""
        d2M_dx2 = self.model.d2M_dx2_ind
        d2M_dx_dt = self.model.d2M_dx_dt_ind
        i = utils.flat_cat(self.expand_xe_ind(d2M_dx2[0]),
                           self.expand_tf_ind(d2M_dx_dt), self.tf_offset)
        j = utils.flat_cat(self.expand_xe_ind(d2M_dx2[1]),
                           self.expand_xe_ind(d2M_dx_dt[0]), self.tf_offset)
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
    
    def constr(self, d):
        """NLP constraints."""
        x, u, tf = self.unpack_decision(d)
        xe = x[[0, -1]].flatten()
        f = self.model.f(x, u)
        g = self.model.g(x, u)
        h = self.model.h(xe, tf)
        J = self.collocation.J
        dt = self.dt
        
        xr = self.unravel_pieces(x)
        fr = self.unravel_pieces(f)
        delta = xr[:, 1:] - xr[:, :-1]
        finc = np.einsum('ijk,lj,il->ilk', fr, J, dt*tf)
        defects = delta - finc
        return utils.flat_cat(defects, g, h)
    
    def constr_jacobian_val(self, d):
        """Values of nonzero elements of NLP constraints Jacobian."""
        x, u, tf = self.unpack_decision(d)
        xe = x[[0, -1]].flatten()
        f = self.model.f(x, u)
        df_dx = self.model.df_dx_val(x, u)
        df_du = self.model.df_du_val(x, u)
        dg_dx = self.model.dg_dx_val(x, u)
        dg_du = self.model.dg_du_val(x, u)
        dh_dx = self.model.dh_dx_val(xe, tf)
        dh_dt = self.model.dh_dt_val(xe, tf)
        J = self.collocation.J
        dt = self.dt
        
        fr = self.unravel_pieces(f)
        dfr_dx = self.unravel_pieces(df_dx)
        dfr_du = self.unravel_pieces(df_du)
        return utils.flat_cat(
            np.ones((self.npieces, self.collocation.ninterv, self.model.nx)),
            -np.ones((self.npieces, self.collocation.ninterv, self.model.nx)),
            -np.einsum('ijk,lj,il->ijkl', dfr_dx, J, dt*tf),
            -np.einsum('ijk,lj,il->ijkl', dfr_du, J, dt*tf),
            -np.einsum('ijk,lj,il->ilk', fr, J, dt),
            dg_dx, dg_du, dh_dx, dh_dt
        )
    
    @property
    @utils.cached
    def constr_jacobian_ind(self):
        """Indices of nonzero elements of NLP constraints Jacobian."""
        x_ind = np.arange(self.model.nx)
        df_dx_i, df_dx_j = self.model.df_dx_ind
        df_du_i, df_du_j = self.model.df_du_ind
        dg_dx_i, dg_dx_j = self.model.dg_dx_ind
        dg_du_i, dg_du_j = self.model.dg_du_ind
        dh_dx_i, dh_dx_j = self.model.dh_dx_ind
        dh_dt_j = self.model.dh_dt_ind
        
        nfinc = self.npieces * self.collocation.ninterv * self.model.nx
        dxr_dx_i = self.unravel_pieces(self.expand_x_ind(x_ind))
        dfr_dx_i = self.unravel_pieces(self.expand_x_ind(df_dx_i))
        dfr_du_i = self.unravel_pieces(self.expand_u_ind(df_du_i))
        dfinc_dt_i = np.repeat(self.tf_offset, nfinc)
        dxr_dx_j = self.expand_defects_ind(x_ind)
        dfr_dx_j = np.swapaxes(self.expand_defects_ind(df_dx_j), 1, 2)
        dfr_du_j = np.swapaxes(self.expand_defects_ind(df_du_j), 1, 2)
        dfinc_dt_j = dxr_dx_j
        i = utils.flat_cat(dxr_dx_i[:, 1:], dxr_dx_i[:, :-1],
                           np.repeat(dfr_dx_i, self.collocation.ninterv),
                           np.repeat(dfr_du_i, self.collocation.ninterv),
                           dfinc_dt_i, 
                           self.expand_x_ind(dg_dx_i), 
                           self.expand_u_ind(dg_du_i),
                           self.expand_xe_ind(dh_dx_i),
                           np.repeat(self.tf_offset, len(dh_dt_j)))
        j = utils.flat_cat(dxr_dx_j, dxr_dx_j,
                           np.repeat(dfr_dx_j, self.collocation.n, axis=0),
                           np.repeat(dfr_du_j, self.collocation.n, axis=0),
                           dfinc_dt_j, 
                           self.expand_g_ind(dg_dx_j),
                           self.expand_g_ind(dg_du_j),
                           self.expand_h_ind(dh_dx_j),
                           self.expand_h_ind(dh_dt_j))
        assert i.size == j.size
        return (i, j)
        
    def constr_jacobian(self, d):
        """NLP constraints Jacobian."""
        val = self.constr_jacobian_val(d)
        ind = self.constr_jacobian_ind
        shape = (self.nd, self.nconstr)
        assert ind[0].size == ind[1].size
        return sparse.coo_matrix((val, ind), shape=shape).todense()
    
    def constr_hessian_val(self, d):
        """Values of nonzero elements of NLP constraints Hessian."""
        x, u, tf = self.unpack_decision(d)
        xe = x[[0, -1]].flatten()
        df_dx = self.model.df_dx_val(x, u)
        df_du = self.model.df_du_val(x, u)
        d2f_dx2 = self.model.d2f_dx2_val(x, u)
        d2f_du2 = self.model.d2f_du2_val(x, u)
        d2f_dx_du = self.model.d2f_dx_du_val(x, u)
        d2g_dx2 = self.model.d2g_dx2_val(x, u)
        d2g_du2 = self.model.d2g_du2_val(x, u)
        d2g_dx_du = self.model.d2g_dx_du_val(x, u)
        d2h_dx2 = self.model.d2h_dx2_val(xe, tf)
        d2h_dt2 = self.model.d2h_dt2_val(xe, tf)
        d2h_dx_dt = self.model.d2h_dx_dt_val(xe, tf)
        J = self.collocation.J
        dt = self.dt
        
        dfr_dx = self.unravel_pieces(df_dx)
        dfr_du = self.unravel_pieces(df_du)
        d2fr_dx2 = self.unravel_pieces(d2f_dx2)
        d2fr_du2 = self.unravel_pieces(d2f_du2)
        d2fr_dx_du = self.unravel_pieces(d2f_dx_du)
        return utils.flat_cat(
            -np.einsum('ijk,lj,il->ijkl', d2fr_dx2, J, dt*tf),
            -np.einsum('ijk,lj,il->ijkl', d2fr_du2, J, dt*tf),
            -np.einsum('ijk,lj,il->ijkl', d2fr_dx_du, J, dt*tf),
            -np.einsum('ijk,lj,il->ijkl', dfr_dx, J, dt),
            -np.einsum('ijk,lj,il->ijkl', dfr_du, J, dt),
            d2g_dx2, d2g_du2, d2g_dx_du, d2h_dx2, d2h_dt2, d2h_dx_dt,
        )
    
    @property
    @utils.cached
    def constr_hessian_ind(self):
        """Indices of nonzero elements of NLP constraints Hessian."""
        ncolpt = self.collocation.n
        ncolinterv = self.collocation.ninterv
        df_dx_i, df_dx_j = self.model.df_dx_ind
        df_du_i, df_du_j = self.model.df_du_ind
        d2f_dx2_i, d2f_dx2_j, d2f_dx2_k = self.model.d2f_dx2_ind
        d2f_du2_i, d2f_du2_j, d2f_du2_k = self.model.d2f_du2_ind
        d2f_dx_du_i, d2f_dx_du_j, d2f_dx_du_k = self.model.d2f_dx_du_ind
        d2g_dx2_i, d2g_dx2_j, d2g_dx2_k = self.model.d2g_dx2_ind
        d2g_du2_i, d2g_du2_j, d2g_du2_k = self.model.d2g_du2_ind
        d2g_dx_du_i, d2g_dx_du_j, d2g_dx_du_k = self.model.d2g_dx_du_ind
        d2h_dx2_i, d2h_dx2_j, d2h_dx2_k = self.model.d2h_dx2_ind
        d2h_dt2_k = self.model.d2h_dt2_ind
        d2h_dx_dt_j, d2h_dx_dt_k = self.model.d2h_dx_dt_ind
        
        dfr_dx_i = self.unravel_pieces(self.expand_x_ind(df_dx_i))
        dfr_du_i = self.unravel_pieces(self.expand_u_ind(df_du_i))
        d2fr_dx2_i = self.unravel_pieces(self.expand_x_ind(d2f_dx2_i))
        d2fr_du2_i = self.unravel_pieces(self.expand_u_ind(d2f_du2_i))
        d2fr_dx_du_i = self.unravel_pieces(self.expand_u_ind(d2f_dx_du_i))
        dfr_dx_j = np.swapaxes(self.expand_defects_ind(df_dx_j), 1, 2)
        dfr_du_j = np.swapaxes(self.expand_defects_ind(df_du_j), 1, 2)
        d2fr_dx2_j = self.unravel_pieces(self.expand_x_ind(d2f_dx2_j))
        d2fr_du2_j = self.unravel_pieces(self.expand_u_ind(d2f_du2_j))
        d2fr_dx_du_j = self.unravel_pieces(self.expand_x_ind(d2f_dx_du_j))
        d2fr_dx2_k = np.swapaxes(self.expand_defects_ind(d2f_dx2_k), 1, 2)
        d2fr_du2_k = np.swapaxes(self.expand_defects_ind(d2f_du2_k), 1, 2)
        d2fr_dx_du_k = np.swapaxes(self.expand_defects_ind(d2f_dx_du_k), 1, 2)
        i = utils.flat_cat(np.repeat(d2fr_dx2_i, ncolinterv),
                           np.repeat(d2fr_du2_i, ncolinterv),
                           np.repeat(d2fr_dx_du_i, ncolinterv),
                           np.repeat(self.tf_offset, dfr_dx_j.size * ncolpt),
                           np.repeat(self.tf_offset, dfr_du_j.size * ncolpt),
                           self.expand_x_ind(d2g_dx2_i),
                           self.expand_u_ind(d2g_du2_i),
                           self.expand_u_ind(d2g_dx_du_i),
                           self.expand_xe_ind(d2h_dx2_i),
                           np.repeat(self.tf_offset, len(d2h_dt2_k)),
                           np.repeat(self.tf_offset, len(d2h_dx_dt_k)))
        j = utils.flat_cat(np.repeat(d2fr_dx2_j, ncolinterv),
                           np.repeat(d2fr_du2_j, ncolinterv),
                           np.repeat(d2fr_dx_du_j, ncolinterv),
                           np.repeat(dfr_dx_i, ncolinterv),
                           np.repeat(dfr_du_i, ncolinterv),
                           self.expand_x_ind(d2g_dx2_j),
                           self.expand_u_ind(d2g_du2_j),
                           self.expand_x_ind(d2g_dx_du_j),
                           self.expand_xe_ind(d2h_dx2_j),
                           np.repeat(self.tf_offset, len(d2h_dt2_k)),
                           self.expand_xe_ind(d2h_dx_dt_j))
        k = utils.flat_cat(np.repeat(d2fr_dx2_k, ncolpt, 0),
                           np.repeat(d2fr_du2_k, ncolpt, 0),
                           np.repeat(d2fr_dx_du_k, ncolpt, 0),
                           np.repeat(dfr_dx_j, ncolpt, axis=0),
                           np.repeat(dfr_du_j, ncolpt, axis=0),
                           self.expand_g_ind(d2g_dx2_k),
                           self.expand_g_ind(d2g_du2_k),
                           self.expand_g_ind(d2g_dx_du_k),
                           self.expand_h_ind(d2h_dx2_k),
                           self.expand_h_ind(d2h_dt2_k),
                           self.expand_h_ind(d2h_dx_dt_k))
        assert i.size == j.size == k.size
        return (i, j, k)
    
    def constr_hessian(self, d):
        """NLP constraints Hessian."""
        val = self.constr_hessian_val(d)
        hessian = np.zeros((self.nd, self.nd, self.nconstr))
        ind = self.constr_hessian_ind
        for v, i, j, k in zip(val, *ind):
            hessian[i, j, k] += v
            if i != j:
                hessian[j, i, k] += v
        return hessian
    
    def nlp_yaipopt(self, d_bounds=None):
        import yaipopt
        
        if d_bounds is None:
            d_bounds = np.tile([[-np.inf], [np.inf]], self.nd)
        
        constr_bounds = np.zeros((2, self.nconstr))
        constr_jac_ind = self.constr_jacobian_ind[::-1]
        merit_hess_ind = self.merit_hessian_ind
        def_hess_ind = self.constr_hessian_ind
        hess_inds = np.hstack((merit_hess_ind[::-1], def_hess_ind[1::-1]))
        
        def merit(d, new_d=True):
            return self.merit(d)
        def grad(d, new_d=True):
            return self.merit_gradient(d)
        def constr(d, new_d=True):
            return self.constr(d)
        def constr_jac(d, new_d=True):
            return self.constr_jacobian_val(d)
        def lag_hess(d, new_d, obj_factor, lmult, new_lmult):
            def_hess = self.constr_hessian_val(d) * lmult[def_hess_ind[2]]
            merit_hess = self.merit_hessian_val(d) * obj_factor
            return utils.flat_cat(merit_hess, def_hess)        
        
        nlp = yaipopt.Problem(d_bounds, merit, grad,
                              constr_bounds=constr_bounds,
                              constr=constr, constr_jac=constr_jac,
                              constr_jac_inds=constr_jac_ind,
                              hess=lag_hess, hess_inds=hess_inds)
        nlp.num_option('obj_scaling_factor', -1)
        return nlp
