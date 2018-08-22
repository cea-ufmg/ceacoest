"""Optimal control."""

import numpy as np

from . import optim, rk


class Problem(optim.Problem):
    """Optimal control problem with LGL direct collocation."""
    
    def __init__(self, model, t, **options):
        self.model = model
        """Underlying cost, constraint and dynamical system model."""
        
        self.collocation = rk.LGLCollocation(model.collocation_order)
        """Collocation method."""
        
        self.tc = self.collocation.grid(t)
        """Normalized collocation time grid."""
        
        assert t[0] == 0 and t[-1] == 1
        assert self.tc[0] == 0 and self.tc[-1] == 1
        npoints = self.tc.size #Total number of collocation points
        
        self.piece_len = np.diff(t)
        """Normalized length of each collocation piece."""
        
        self.npieces = len(t) - 1
        """Number of collocation pieces."""
        
        super().__init__()
        x = self.register_decision('x', (npoints, model.nx))
        u = self.register_decision('u', (npoints, model.nx))
        p = self.register_decision('p', model.np) 
        T = self.register_decision('T', ())
        
        self.register_constraint(
            'g', (npoints, model.ng), model.g, ('x', 'u', 'p')
        )
        self.register_constraint(
            'h', model.nh, model.h, ('xe', 'p', 'T')
        )

        self.register_constraint_jacobian(
            'g', 'x', model.dg_dx_val, model.dg_dx_ind
        )
        self.register_constraint_jacobian(
            'g', 'u', model.dg_du_val, model.dg_du_ind
        )
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        var = super().variables(dvec)
        x = var['x']
        u = var['u']
        xp = np.zeros((self.npieces, self.collocation.n, self.model.nx))
        up = np.zeros((self.npieces, self.collocation.n, self.model.nu))
        xp[:, :-1].flat = x[:-1, :].flat
        up[:, :-1].flat = u[:-1, :].flat
        xp[:-1, -1] = xp[1:, 0]
        up[:-1, -1] = up[1:, 0]
        xp[-1, -1] = x[-1]
        up[-1, -1] = u[-1]
        
        var.update(xp=xp, up=up, piece_len=self.piece_len)
        return var
    
