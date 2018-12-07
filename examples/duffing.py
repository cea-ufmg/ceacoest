"""Test various modules using a Duffing oscillator SDE model."""


import numpy as np
import sympy
import sym2num.model
from numpy import ma
from scipy import stats

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


symb_mdl = SymbolicDuffing()
disc_mdl = SymbolicDiscretizedDuffing()

GeneratedDuffing = sym2num.model.compile_class(symb_mdl)
