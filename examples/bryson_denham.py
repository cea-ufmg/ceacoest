"""Bryson--Denham optimal control problem."""


import functools

import numpy as np
import sympy
import sym2num.model
import sym2num.utils
import sym2num.var

from ceacoest import oc, optim
from ceacoest.modelling import symoc


@symoc.collocate(order=3)
class BrysonDenham:
    """Symbolic Bryson--Denham optimal control model."""
        
    @sym2num.utils.classproperty
    @functools.lru_cache()
    def variables(cls):
        """Model variables definition."""
        vars = [sym2num.var.SymbolArray('x', ['x1', 'x2']),
                sym2num.var.SymbolArray('u', ['u1']),
                sym2num.var.SymbolArray('p', ['T'])]
        return sym2num.var.make_dict(vars)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        return sympy.Array([s.x2, s.u1]) * s.T
    
    @sym2num.model.collect_symbols
    def g(self, x, u, p, *, s):
        """Path constraints."""
        return sympy.Array([], 0)
    
    @sym2num.model.collect_symbols
    def h(self, xe, p, *, s):
        """Endpoint constraints."""
        return sympy.Array([], 0)
    
    @sym2num.model.collect_symbols
    def L(self, x, u, p, *, s):
        """Lagrange (running) cost."""
        return sympy.Array(0.5*s.u1**2*s.T) 
    
    @sym2num.model.collect_symbols
    def M(self, xe, p, *, s):
        """Mayer (endpoint) cost."""
        return sympy.Array(0)


if __name__ == '__main__':
    symb_mdl = BrysonDenham()
    GeneratedBrysonDenham = sym2num.model.compile_class(symb_mdl)
    mdl = GeneratedBrysonDenham()
    
    tcoarse = np.linspace(0, 1, 101)
    problem = oc.Problem(mdl, tcoarse)
    
    dec_L, dec_U = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    problem.set_decision_item('T', 0, dec_L)
    problem.set_decision_item('T', 4, dec_U)
    problem.set_decision_item('x1', 0, dec_L)
    problem.set_decision_item('x1', 1/9, dec_U)
    problem.set_decision_item('x1_initial', 0, dec_L)
    problem.set_decision_item('x1_initial', 0, dec_U)
    problem.set_decision_item('x1_final', 0, dec_L)
    problem.set_decision_item('x1_final', 0, dec_U)
    problem.set_decision_item('x2_initial', 1, dec_L)
    problem.set_decision_item('x2_initial', 1, dec_U)
    problem.set_decision_item('x2_final', -1, dec_L)
    problem.set_decision_item('x2_final', -1, dec_U)
    constr_bounds = np.zeros((2, problem.ncons))
    
    with problem.ipopt((dec_L, dec_U), constr_bounds) as nlp:
        dec0 = np.zeros(problem.ndec)
        problem.set_decision('p', 1, dec0)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    uopt = opt['u']
    Topt = opt['p'][0]
    topt = problem.tc * Topt
