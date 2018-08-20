"""Optimal control."""

import numpy as np

from . import optim, rk


class Problem(optim.Problem):
    """Optimal control problem with LGL direct collocation."""
    
    def __init__(self, model, t, **options):
        self.model = model
        """Underlying cost, constraint and dynamical system model."""
        
        collocation_order = options.get('collocation_order', 3)
        self.collocation = rk.LGLCollocation(collocation_order)
        """Collocation method."""
        
        self.tc = self.collocation.grid(t)
        """Normalized collocation time grid."""
        assert self.tc[0] == 0 and self.tc[-1] == 1
        
        self.npieces = len(t) - 1
        """Number of collocation pieces."""
        
