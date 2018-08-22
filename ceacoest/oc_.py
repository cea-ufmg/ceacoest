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
        
        self.piece_len = np.diff(t)
        """Length of each collocation piece."""
        
        self.npieces = len(t) - 1
        """Number of collocation pieces."""
        
        super().__init__()
        self.register_decision('T', ())
        self.register_decision('x', (self.tc.size, model.nx))
        self.register_decision('u', (self.tc.size, model.nx))
        self.register_decision('p', (self.tc.size, model.np))
