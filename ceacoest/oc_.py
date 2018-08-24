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
        
        assert np.ndim(t) == 1
        assert t[0] == 0 and t[-1] == 1
        self.tc = self.collocation.grid(t)
        """Normalized collocation time grid."""
        
        self.piece_len = np.diff(t)
        """Normalized length of each collocation piece."""
        
        self.npieces = npieces = len(t) - 1
        """Number of collocation pieces."""
        
        super().__init__()
        npoints = self.tc.size #Total number of collocation points
        x = self.register_decision('x', model.nx, npoints)
        u = self.register_decision('u', model.nu, npoints)
        p = self.register_decision('p', model.np)
        T = self.register_decision('T', ())
        
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
        self.register_constraint('h', model.h, ('xe', 'p', 'T'), model.nh)
        self.register_constraint(
            'e', model.e, ('xp','up','p','T','piece_len'), model.ne, npieces
        )
        
        self.register_constraint_jacobian(
            'g', 'x', model.dg_dx_val, model.dg_dx_ind
        )
        self.register_constraint_jacobian(
            'g', 'u', model.dg_du_val, model.dg_du_ind
        )
        self.register_constraint_jacobian(
            'e', 'xp', model.de_dxp_val, model.de_dxp_ind
        )
        self.register_constraint_jacobian(
            'e', 'up', model.de_dup_val, model.de_dup_ind
        )
        self.register_constraint_jacobian(
            'h', 'xe', model.dh_dxe_val, model.dh_dxe_ind
        )
        self.register_constraint_jacobian(
            'h', 'T', model.dh_dT_val, model.dh_dT_ind
        )
    
    def xe_offsets(self, xe_ind):
        """Calculate offsets of indices of the endpoint states (xe)."""
        x_off = self.decision['x'].offset
        npoints = self.tc.size
        return x_off + (xe_ind >= self.model.nx)*self.model.nx*(npoints - 1)
        
    def unravel_pieces(self, v):
        v = np.asarray(v)
        assert v.ndim > 0 and v.shape[0] == self.tc.size
        vp = np.zeros((self.npieces, self.collocation.n) + v.shape[1:])
        vp[:, :-1].flat = v[:-1, :].flat
        vp[:-1, -1] = vp[1:, 0]
        vp[-1, -1] = v[-1]
        return vp
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        var = super().variables(dvec)
        var.update(xp=self.unravel_pieces(var['x']),
                   up=self.unravel_pieces(var['u']),
                   xe=var['x'][[0, -1]],
                   piece_len=self.piece_len)
        return var

