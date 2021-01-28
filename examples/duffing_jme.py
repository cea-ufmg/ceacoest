#!/usr/bin/env python

"""Duffing oscillator SDE MAP state-path and parameter estimation."""


import importlib

import numpy as np
import sympy
import sym2num.model
from numpy import ma
from scipy import interpolate, stats, signal

from ceacoest import jme
from ceacoest.modelling import symjme, symsde, symstats


# Reload modules for testing
for m in (symjme, jme):
    importlib.reload(m)


class SymbolicDuffingSim(sym2num.model.Base):
    """Symbolic continuous-time Duffing oscillator model for simulation."""
    
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
        v['self']['x1_meas_std'] = 'x1_meas_std'
        
        v['x'] = ['x', 'v']
        v['y'] = ['x1_meas']
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
        return [[s.x1_meas_std ** 2]]


class SymbolicDuffingJME(symjme.Model):
    """Symbolic Duffing oscilator JME estimation model."""
    
    collocation_order = 3
    
    def __init__(self):
        v = self.Variables(
            x=['x1', 'x2'],
            y=['x1_meas'],
            u=['u1'],
            p=['gamma', 'alpha', 'beta', 'delta', 'x1_meas_std'],
            G=[['g1'], ['g2']],
        )
        super().__init__(v)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        f1 = s.x2
        f2 = -s.delta*s.x2 - s.beta*s.x1 - s.alpha*s.x1**3 + s.gamma*s.u1
        return [f1, f2]
    
    @sym2num.model.collect_symbols
    def L(self, y, x, u, p, *, s):
        """Measurement log likelihood."""
        return symstats.normal_logpdf1(s.x1_meas, s.x1, s.x1_meas_std)


class SymbolicDiscretizedDuffing(symsde.ItoTaylorAS15DiscretizedModel):
    ContinuousTimeModel = SymbolicDuffingSim


def sim(sim_model, seed=1):
    np.random.seed(seed)
    
    # Problem data
    tf = 900
    x0 = np.array([-1, 1])
    Px0 = np.diag([0.1, 0.1]) ** 2
    
    # Generate the time vector
    dt_sim = sim_model.dt_sim
    Nsim = int(tf / dt_sim) + 1
    ksim = np.arange(Nsim)
    tsim = ksim * dt_sim
    
    # Simulate the system
    sim_model.dt = sim_model.dt_sim
    w = np.random.randn(Nsim - 1, sim_model.nw)
    xsim = np.zeros((Nsim, sim_model.nx))
    xsim[0] = stats.multivariate_normal.rvs(x0, Px0)
    for k in ksim[:-1]:
        xsim[k + 1] = sim_model.f(k, xsim[k])  + sim_model.g(k, xsim[k]) @ w[k]
    
    # Sample the outputs
    R = sim_model.R()
    v = np.random.multivariate_normal(np.zeros(sim_model.ny), R, Nsim)
    ysim = ma.asarray(sim_model.h(k, xsim) + v)
    return tsim, xsim, ysim, x0, Px0


def cdiff(x, T=1):
    xd = np.empty_like(x)
    xd[1:-1] = 0.5 / T * (x[2:] - x[:-2])
    xd[0] = xd[1]
    xd[-1] = xd[-2]
    return xd


if __name__ == '__main__':
    sym_disc_mdl = SymbolicDiscretizedDuffing()
    sim_model = sym_disc_mdl.compile_class()()
    
    params = dict(
        alpha=1, beta=-1, delta=0.2, gamma=0.3, omega=1,
        g2=0.1, x1_meas_std=0.1,
        dt_sim=0.005,
    )
    for k,v in params.items():
        setattr(sim_model, k, v)
    
    tsim, xsim, ysim, x0, Px0 = sim(sim_model)
    keep = 5
    y = ysim[::keep].copy()
    t = tsim[::keep]
    mask = np.ones_like(t, bool)
    mask[::3] = False
    y[mask] = ma.masked

    # Make guess
    tf = t[~y.mask.ravel()]
    lpf = signal.butter(3, 0.3, output='sos')
    yf0 = signal.sosfiltfilt(lpf, y.compressed())
    yf1 = cdiff(yf0, np.diff(tf[:2]))
    
    sym_jme_mdl = SymbolicDuffingJME()
    jme_model = sym_jme_mdl.compile_class()()
    jme_model.G = np.array([[0], [params['g2']]])
    
    ufun = lambda t: np.sin(np.asarray(t)[..., None]*params['omega'])
    problem = jme.Problem(jme_model, t, y, ufun)
    tc = problem.tc
    
    # Define initial guess
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['x'][:, 0] = interpolate.interp1d(tf, yf0)(tc)
    var0['x'][:, 1] = interpolate.interp1d(tf, yf1)(tc)
    var0['p'][-1] = np.std(yf0 - y.compressed())  # x1_meas_std
    
    # Define bounds on variables
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    var_L['p'][-1] = 1e-5 # x1_meas_std

    # Define bounds on constraints
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    
    # Define problem scaling
    dec_scale = np.ones(problem.ndec)
    constr_scale = np.ones(problem.ncons)
    obj_scale = -1.0
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 100.0)
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)

    opt = problem.variables(decopt)
    xopt = opt['x']
    popt = opt['p']
