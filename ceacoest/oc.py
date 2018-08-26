"""Optimal control."""

import numpy as np

from . import optim, rk


class Problem(optim.Problem):
    """Optimal control problem with LGL direct collocation."""
    
    def __init__(self, model, t, **options):
        self.model = model
        """Underlying cost, constraint and dynamical system model."""

        self.collocation = col = rk.LGLCollocation(model.collocation_order)
        """Collocation method."""
        
        self.piece_len = np.diff(t)
        """Normalized length of each collocation piece."""

        assert np.all(self.piece_len > 0)
        self.npieces = npieces = len(self.piece_len)
        """Number of collocation pieces."""
        
        self.tc = self.collocation.grid(t)
        """Normalized collocation time grid."""
        
        super().__init__()
        npoints = self.tc.size #Total number of collocation points
        x = self.register_decision('x', model.nx, npoints)
        u = self.register_decision('u', model.nu, npoints)
        p = self.register_decision('p', model.np)
        
        self.set_index_offsets('x', np.arange(npoints) * model.nx + x.offset)
        self.set_index_offsets('u', np.arange(npoints) * model.nu + u.offset)
        self.set_index_offsets(
            'xp', np.arange(npieces) * col.ninterv * model.nx + x.offset
        )
        self.set_index_offsets(
            'up', np.arange(npieces)*col.ninterv*model.nu + u.offset
        )
        self.set_index_offsets('xe', self.xe_offsets)
        
        self.register_constraint('g', model.g, ('x','u','p'), model.ng, npoints)
        self.register_constraint('h', model.h, ('xe', 'p'), model.nh)
        self.register_constraint(
            'e', model.e, ('xp','up','p', 'piece_len'), model.ne, npieces
        )
        
        self.register_constraint_jacobian(
            'g', 'x', model.dg_dx_val, model.dg_dx_ind
        )
        self.register_constraint_jacobian(
            'g', 'u', model.dg_du_val, model.dg_du_ind
        )
        self.register_constraint_jacobian(
            'g', 'p', model.dg_du_val, model.dg_dp_ind
        )
        self.register_constraint_jacobian(
            'e', 'xp', model.de_dxp_val, model.de_dxp_ind
        )
        self.register_constraint_jacobian(
            'e', 'up', model.de_dup_val, model.de_dup_ind
        )
        self.register_constraint_jacobian(
            'e', 'p', model.de_dp_val, model.de_dp_ind
        )
        self.register_constraint_jacobian(
            'h', 'xe', model.dh_dxe_val, model.dh_dxe_ind
        )
        self.register_constraint_jacobian(
            'h', 'p', model.dh_dp_val, model.dh_dp_ind
        )
    
    def xe_offsets(self, xe_ind):
        """Calculate offsets of indices of the endpoint states (xe)."""
        x_off = self.decision['x'].offset
        npoints = self.tc.size
        return x_off + (xe_ind >= self.model.nx)*self.model.nx*(npoints - 1)
            
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'piece_len': self.piece_len, **super().variables(dvec)}


class PieceRavelledVariable:
    def __init__(self, problem, var_name):
        self.p = problem
        self.var_name = var_name
    
    @property
    def var(self):
        self.p.variables[var_name]
    
    @property
    def nvar(self):
        return self.var.shape[1]
    
    def build(self, variables):
        v = variables[self.var_name]
        assert v.shape == self.var.shape
        vp = np.zeros((self.p.npieces, self.p.collocation.n, self.nvar))
        vp[:, :-1].flat = v[:-1, :].flat
        vp[:-1, -1] = vp[1:, 0]
        vp[-1, -1] = v[-1]
        return vp
    
    def pack_into(self, vec, value):
        vp = np.asarray(value)
        assert vp.shape == (self.p.npieces, self.p.collocation.n, self.nvar)
        v = np.zeros(self.var.shape)
        v[:-1, :].flat = vp[:, :-1].flat
        v[-1] = vp[-1, -1]
        v[self.p.collocation.n::self.p.collocation.n] += vp[:, -1]
        self.var.pack_into(vec, self.repack_pieces(v))
    
    def expand_indices(self, ind):
        npieces = self.p.npieces
        increments = self.p.collocation.ninterv * self.nvar
        return ind + np.arange(npieces)[:, None] * increments + self.var.offset
