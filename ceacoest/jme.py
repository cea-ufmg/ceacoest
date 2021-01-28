"""Joint maximum a posteriori state-path and parameter estimator for SDEs."""


import numpy as np

from . import oem


class Problem(oem.Problem):
    """Joint MAP state-path and parameter estimation problem."""
    
    def __init__(self, model, t, y, u):
        super().__init__(model, t, y, u)
    
    def _init_collocation_variables(self):
        """Register problem collocation variables."""
        super()._init_collocation_variables()
        col = self.collocation
        model = self.model

        # Coordinates over the basis for the process noise intensity w
        self.add_decision('wc', (self.npieces, col.ninterv, model.nw))

        # Add fictitious log-density of tubes to objective function
        self.add_objective(model.tube_L, self.npieces)
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'G': self.model.G, **super().variables(dvec)}
