#!/usr/bin/env python

"""Brachistochrone optimal control example."""


import importlib

import numpy as np
import sympy
import sym2num.model

from ceacoest import oc, col, optim
from ceacoest.modelling import genoptim, symoc, symcol, symoptim


# Reload modules for testing
for m in (optim, col, oc, genoptim, symoptim, symcol, symoc):
    importlib.reload(m)


class SymbolicBrachistochroneModel(symoc.Model):
    """Symbolic Brachistochrone optimal control model."""
    
    collocation_order = 3
    
    def __init__(self):
        v = self.Variables(
            x=['x', 'y', 'v'],
            u=['theta'],
            p=['tf']
        )
        super().__init__(v)

    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""        
        g = 9.80665
        xdot = [s.v * sympy.sin(s.theta) * s.tf,
                -s.v * sympy.cos(s.theta) * s.tf,
                g * sympy.cos(s.theta) * s.tf]
        return xdot
    
    @sym2num.model.collect_symbols
    def M(self, xe, p, *, s):
        """Mayer (endpoint) cost."""
        return s.tf


if __name__ == '__main__':
    symmodel = SymbolicBrachistochroneModel()
    GeneratedBrachistochrone = symmodel.compile_class()    
    model = GeneratedBrachistochrone()

    t = np.linspace(0, 1, 200)
    problem = oc.Problem(model, t)
    tc = problem.tc
    dec0 = np.zeros(problem.ndec)
    var0 = problem.variables(dec0)
    var0['p'][:] = 1
    
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    var_L = problem.variables(dec_L)
    var_U = problem.variables(dec_U)
    
    var_L['p'][:] = 1e-5
    var_L['x'][0, :3] = 0
    var_U['x'][0, :3] = 0
    var_L['x'][-1, :2] = [10, -5]
    var_U['x'][-1, :2] = [10, -5]
    var_L['u'][:] = -np.pi
    var_U['u'][:] = np.pi
    
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    uopt = opt['u']
    popt = opt['p']

