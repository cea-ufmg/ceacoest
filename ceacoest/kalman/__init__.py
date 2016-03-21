"""Kalman filtering / smoothing package."""

from . import base
from . import unscented
from . import extended


DTUnscentedKalmanFilter = unscented.DTKalmanFilter
