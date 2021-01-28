"""Symbolic model building for joint MAP state-path and parameter estimation."""


import collections
import functools

import numpy as np
import sym2num.model
import sympy

from . import symoem


class Model(symoem.Model):

    use_om = True
    """Whether to use the Onsager--Machlup drift divergence."""

    def __init__(self, variables, decision=set()):
        nw = variables['G'].shape[1]
        ninterv = self.collocation.ninterv
        wc = [[f'wc{j}_interv_{i}' for j in range(nw)] for i in range(ninterv)]
        variables['wc'] = wc
        decision = decision | {'wc'}
        super().__init__(variables, decision)
        
        self.add_derivative('f', 'x', 'df_dx')
        self.add_objective('tube_L')
    
    @property
    def generate_assignments(self):
        gen = {'nw': self.variables['G'].shape[1],
               **getattr(super(), 'generate_assignments', {})}
        return gen

    def e(self, xp, up, p, piece_len, wc, G):
        """Collocation defects (error)."""
        ncol = self.collocation.n
        J = self.collocation.J
        JT_range = self.collocation.JT_range
        fp = np.array([self.f(xp[i], up[i], p) for i in range(ncol)])
        wp = JT_range @ wc
        
        defects = xp[1:] - xp[:-1] - J @ (fp + wp @ G.T) * piece_len
        return defects
    
    def tube_L(self, xp, up, p, piece_len, wc):
        """Fictitious log-density of state-path tube."""
        JT_range = self.collocation.JT_range
        wp = JT_range @ wc

        ncol = self.collocation.n        
        drift_div_p = np.array(
            [np.trace(self.df_dx(xp[i,:], up[i,:], p)) for i in range(ncol)]
        )

        K = self.collocation.K
        tube_L = -0.5 * (drift_div_p + np.sum(wp ** 2, 1)) @ K * piece_len
        return tube_L
        
