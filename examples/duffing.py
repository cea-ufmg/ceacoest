#!/usr/bin/env python

"""Test various modules using a Duffing oscillator SDE model."""


import importlib

import numpy as np
import sympy
import sym2num.model
from numpy import ma
from scipy import stats

from ceacoest import kalman
from ceacoest.modelling import symsde


# Reload modules for testing
for m in (symsde, kalman):
    importlib.reload(m)


class SymbolicDuffing(sym2num.model.Base):
    """Symbolic continuous-time Duffing oscillator model."""
    
    generate_functions = ['f']
    
    def __init__(self):
        super().__init__()
    
        v = self.variables
        v['self']['gamma'] = 'gamma'
        v['self']['omega'] = 'omega'
        v['self']['alpha'] = 'alpha'
        v['self']['beta'] = 'beta'
        v['self']['delta'] = 'delta'
        v['self']['g2'] = 'g2'
        v['self']['X_meas_std'] = 'X_meas_std'
        
        v['x'] = ['x', 'v']
        v['y'] = ['X_meas']
        v['t'] = 't'
        
        self.set_default_members()
    
    @sym2num.model.collect_symbols
    def f(self, t, x, *, s):
        """Drift function."""
        u = s.gamma * sympy.sin(s.t * s.omega)
        f1 = s.v
        f2 = -s.delta * s.v - s.beta * s.x - s.alpha * s.x ** 3 + u
        return np.array([f1, f2])
    
    @sym2num.model.collect_symbols
    def g(self, t, x, *, s):
        """Diffusion matrix."""
        return np.array([[0], [s.g2]])
    
    @sym2num.model.collect_symbols
    def h(self, t, x, *, s):
        """Measurement function."""
        return np.array([s.x])
    
    @sym2num.model.collect_symbols
    def R(self, *, s):
        """Measurement covariance."""
        return np.array([[s.X_meas_std ** 2]])


def sim(model, seed=1):
    np.random.seed(seed)
    
    # Generate the time vector
    dt = model.dt
    N = int(30 // dt)
    k = np.arange(N)
    t = k * dt
    
    x0 = np.array([-1, 1])
    Px0 = np.diag([0.1, 0.1]) ** 2

    # Simulate the system
    w = np.random.randn(N - 1, model.nw)
    x = np.zeros((N, model.nx))
    x[0] = stats.multivariate_normal.rvs(x0, Px0)
    for k in range(N - 1):
        x[k + 1] = model.f(k, x[k])  + model.g(k, x[k]).dot(w[k])
    
    # Sample the outputs
    R = model.R()
    v = np.random.multivariate_normal(np.zeros(model.ny), R, N)
    y = ma.asarray(model.h(k, x) + v)
    return t, x, y


if __name__ == '__main__':
    sym_cont_mdl = SymbolicDuffing()
    sym_disc_mdl = symsde.ItoTaylorAS15DiscretizedModel(sym_cont_mdl)
    model = sym_disc_mdl.compile_class()()

    params = dict(
        alpha=1, beta=-1, delta=0.2, gamma=0.3, omega=1,
        g2=0.1, X_meas_std=0.1,
        dt=0.05,
    )
    for k,v in params.items():
        setattr(model, k, v)
    
    t, x, y = sim(model)
    
    #kf = kalman.DTExtendedFilter(model)
    #[xs, Pxs] = kf.smooth(y)
