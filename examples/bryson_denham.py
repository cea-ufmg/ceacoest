#!/usr/bin/env python

"""Bryson--Denham optimal control problem."""


import importlib

import numpy as np
import sympy
import sym2num.model

from ceacoest import oc, col, optim
from ceacoest.modelling import genoptim, symoc, symcol, symoptim


# Reload modules for testing
for m in (optim, col, oc, genoptim, symoptim, symcol, symoc):
    importlib.reload(m)


class SymbolicBrysonDenham(symoc.Model):
    """Symbolic Bryson--Denham optimal control model."""
    
    collocation_order = 3

    def __init__(self):
        v = self.Variables(
            x=['x1', 'x2'],
            u=['u1'],
            p=[],
        )
        super().__init__(v)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        return [s.x2, s.u1]
    
    @sym2num.model.collect_symbols
    def L(self, x, u, p, *, s):
        """Lagrange (running) cost."""
        return 0.5 * s.u1**2


if __name__ == '__main__':
    symmodel = SymbolicBrysonDenham()
    GeneratedBrysonDenham = symmodel.compile_class()    
    model = GeneratedBrysonDenham()
    
    tcoarse = np.linspace(0, 1, 101)
    problem = oc.Problem(model, tcoarse)
    tc = problem.tc
    
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    
    var_L['x'][:, 0] = 0
    var_U['x'][:, 0] = 1/9
    var_L['x'][0] = [0, 1]
    var_U['x'][0] = [0, 1]
    var_L['x'][-1] = [0, -1]
    var_U['x'][-1] = [0, -1]
    
    constr_bounds = np.zeros((2, problem.ncons))
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    uopt = opt['u']
