"""Output Error Method estimation."""


import numpy as np

from . import rk


class CTEstimator:
    """Continuous-time output error method estimator."""
    def __init__(self, model, y, t, **options):
        self.model = model
        """The underlying dynamical system model."""

        self.y = y
        """The measurements."""
    
        self.t_piece = t
        """The piece boundary times."""

        order = options.get('order', 3)
        self.collocation = rk.LGLCollocation(order)
