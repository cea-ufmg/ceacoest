"""Test various modules using a Duffing oscillator SDE model."""


import numpy as np
import sympy
import sym2num.model
from numpy import ma
from scipy import stats

from ceacoest import kalman
from ceacoest.modelling import symsde


class SymbolicDuffing(sym2num.model.Base):
    """Symbolic Duffing oscillator model."""
    
    generate_functions = ['f']
    derivatives = [('df_dx', 'f', 'x'),
                   ('df_dt', 'f', 't'),
                   ('d2f_dx2', 'df_dx', 'x')]
    
    @property
    def variables(self):
        """Model variables definition."""
        v = super().variables
        v['self'] = {'consts': 'gamma omega X0_std V0_std'}
        v['x'] = 'X V'
        v['p'] = 'alpha beta delta g2 X_meas_std X0 V0'
        v['y'] = 'X_meas',
        v['t'] = 't'
        return v
    
    @sym2num.model.collect_symbols
    def f(self, t, x, p, *, s):
        """Drift function."""
        u = s.gamma * sympy.cos(s.t * s.omega)
        f1 = s.V
        f2 = -s.delta * s.V - s.beta * s.X - s.alpha * s.X ** 3 + u
        return [f1, f2]
    
    @sym2num.model.collect_symbols
    def g(self, t, x, p, *, s):
        """Diffusion matrix."""
        return [[0], [s.g2]]
    
    @sym2num.model.collect_symbols
    def h(self, t, x, p, *, s):
        """Measurement function."""
        return [s.X]
    
    @sym2num.model.collect_symbols
    def R(self, p, *, s):
        """Measurement covariance."""
        return [[s.X_meas_std ** 2]]
    
    @sym2num.model.collect_symbols
    def x0(self, p, *, s):
        """Initial state prior mean."""
        return [s.X0, s.V0]
    
    @sym2num.model.collect_symbols
    def Px0(self, p, *, s):
        """Initial state prior covariance."""
        return sympy.diag(s.X0_std, s.V0_std)**2


class SymbolicDiscretizedDuffing(symsde.EulerDiscretizedSDEModel):
    ContinuousTimeModel = SymbolicDuffing
    
    derivatives = [('df_dx', 'f', 'x'),
                   ('dh_dx', 'h', 'x'),]

    @property
    def generate_functions(self):
        gen = ['df_dx', 'dh_dx']
        return super().generate_functions + gen


symb_mdl = SymbolicDuffing()
disc_mdl = SymbolicDiscretizedDuffing()

GeneratedDuffing = sym2num.model.compile_class(disc_mdl)

class DiscretizedDuffing(GeneratedDuffing):
    def __init__(self, dt, p, c):
        self.dt = np.asarray(dt)
        """Discretization time step"""
        
        self.ct_model_p = np.asarray(p)
        """Model parameters"""

        self.consts = np.asarray(c)
        """Model constants"""

    def parametrize(self, p):
        self.ct_model_p = np.asarray(p)
    
    def t(self, k):
        return self.dt * k


def sim():
    np.random.seed(1)
    
    # Generate the time vector
    dt = 0.05
    N = int(30 // dt)
    k = np.arange(N)
    t = k * dt

    # Instantiate the model
    given = dict(
        alpha=1, beta=-1, delta=0.2, gamma=0.3, omega=1,
        g1=0, g2=0.1, X_meas_std=0.1,
        X0=-1, V0=1, X0_std=0.1, V0_std=0.1
    )
    p = np.r_[[given[v.name] for v in symb_mdl.variables['p']]]
    c = np.r_[[given[v.name] for v in symb_mdl.variables['self'].consts]]
    model = DiscretizedDuffing(dt, p, c)
    
    # Simulate the system
    w = np.random.randn(N - 1, model.nw)
    x = np.zeros((N, model.nx))
    x[0] = stats.multivariate_normal.rvs(model.x0(), model.Px0())
    for k in range(N - 1):
        x[k + 1] = model.f(k, x[k])  + model.g(k, x[k]).dot(w[k])
    
    # Sample the outputs
    R = model.R()
    v = np.random.multivariate_normal(np.zeros(model.ny), R, N)
    y = ma.asarray(model.h(k, x) + v)
    return model, t, x, y, p


if __name__ == '__main__':
    model, t, x, y, p = sim()
    kf = kalman.DTExtendedFilter(model)
    [xs, Pxs] = kf.smooth(y)
