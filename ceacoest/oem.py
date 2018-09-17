"""Output Error Method estimator."""


import itertools

import numpy as np

from . import optim, rk, utils


class Problem(optim.Problem):
    """Output error method optimization problem with LGL direct collocation."""
    

