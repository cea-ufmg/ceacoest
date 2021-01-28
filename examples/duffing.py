#!/usr/bin/env python

"""Test various modules using a Duffing oscillator SDE model."""


import importlib

import numpy as np
import sympy
import sym2num.model
from numpy import ma
from scipy import stats

from ceacoest import kalman
from ceacoest.kalman import base, extended, unscented
from ceacoest.modelling import symsde, symstats


# Reload modules for testing
for m in ():
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
        return [f1, f2]
    
    @sym2num.model.collect_symbols
    def g(self, t, x, *, s):
        """Diffusion matrix."""
        return [[0], [s.g2]]
    
    @sym2num.model.collect_symbols
    def h(self, t, x, *, s):
        """Measurement function."""
        return [s.x]
    
    @sym2num.model.collect_symbols
    def R(self, *, s):
        """Measurement covariance."""
        return [[s.X_meas_std ** 2]]


class SymbolicDiscretizedDuffing(symsde.ItoTaylorAS15DiscretizedModel):
    
    ContinuousTimeModel = SymbolicDuffing

    def __init__(self):
        super().__init__()
        
        self.add_derivative('h', 'x', 'dh_dx')
        self.add_derivative('f', 'x', 'df_dx')
    
    @property
    def generate_functions(self):
        return ['dh_dx', 'df_dx', *super().generate_functions]


def sim(model, seed=1):
    np.random.seed(seed)
    
    # Problem data
    tf = 30
    x0 = np.array([-1, 1])
    Px0 = np.diag([0.1, 0.1]) ** 2
    
    # Generate the time vector
    dt_sim = model.dt_sim
    Nsim = int(tf / dt_sim) + 1
    ksim = np.arange(Nsim)
    tsim = ksim * dt_sim
    
    # Simulate the system
    model.dt = model.dt_sim
    w = np.random.randn(Nsim - 1, model.nw)
    xsim = np.zeros((Nsim, model.nx))
    xsim[0] = stats.multivariate_normal.rvs(x0, Px0)
    for k in ksim[:-1]:
        xsim[k + 1] = model.f(k, xsim[k])  + model.g(k, xsim[k]) @ w[k]
    
    # Sample the outputs
    dt_est = model.dt_est
    Nest = int(tf / dt_est) + 1
    est_inc = int(dt_est / dt_sim)
    R = model.R()
    v = np.random.multivariate_normal(np.zeros(model.ny), R, Nest)
    y = ma.asarray(model.h(k, xsim[::est_inc]) + v)
    test = np.arange(Nest) * dt_est
    return tsim, xsim, test, y, x0, Px0


if __name__ == '__main__':
    sym_disc_mdl = SymbolicDiscretizedDuffing()
    model = sym_disc_mdl.compile_class()()
    
    params = dict(
        alpha=1, beta=-1, delta=0.2, gamma=0.3, omega=1,
        g2=0.1, X_meas_std=0.1,
        dt_sim=0.005, dt_est=0.1,
    )
    for k,v in params.items():
        setattr(model, k, v)
    
    tsim, xsim, test, y, x0, Px0 = sim(model)
    model.dt = model.dt_est
    
    ekf = kalman.DTExtendedFilter(model, x0, Px0)
    [xef, Pxef] = ekf.filter(y)
    
    ekf = kalman.DTExtendedFilter(model, x0, Px0)
    [xes, Pxes] = ekf.smooth(y)
    
    ukf = kalman.DTUnscentedFilter(model, x0, Px0)
    [xuf, Pxuf] = ukf.filter(y)
    
    ukf = kalman.DTUnscentedFilter(model, x0, Px0)
    [xus, Pxus] = ukf.smooth(y)
