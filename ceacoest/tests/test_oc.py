"""Optimal control test module."""


import numpy as np
import pytest
import sympy
import sym2num.model
import sym2num.var

from ceacoest import oc, symb_oc
from .test_optim import (test_merit_gradient, test_merit_hessian, 
                         test_constraint_jacobian, test_constraint_hessian,
                         seed, dec)


from ceacoest.testsupport.array_cmp import ArrayDiff


@symb_oc.collocate(order=3)
class SymbolicModel:
    """Symbolic optimal control test model."""
        
    @sym2num.model.make_variables_dict
    def variables():
        """Model variables definition."""
        return [
            sym2num.var.SymbolArray('x', ['x1', 'x2', 'x3']),
            sym2num.var.SymbolArray('u', ['u1', 'u2']),
            sym2num.var.SymbolArray('p', ['p1', 'p2']),
        ]
    
    @sym2num.model.symbols_from('x, u, p')
    def f(self, s):
        """ODE function."""
        return sympy.Array(
            [s.x1 * sympy.cos(s.x2), 
             s.u1 * s.u2 * s.p1 * s.x2**2, 
             s.u1**2 * s.x1**3 * sympy.exp(s.p1)]
        )
    
    @sym2num.model.symbols_from('x, u, p')
    def g(self, s):
        """Path constraints."""
        return sympy.Array(
            [(s.u1 ** 2 + 2) **  (s.x1** 2 + 1) * s.p1,
             s.p1 ** 3 * s.p2 * s.u1 ** 2 * s.p2])
    
    @sym2num.model.symbols_from('xe, p')
    def h(self, s):
        """Endpoint constraints."""
        return sympy.Array(
            [s.x1_start * s.p1 ** 3, 
             s.x1_start * s.x2_end ** 3 * s.p1])
    
    @sym2num.model.symbols_from('xe, p')
    def M(self, s):
        """Mayer (endpoint) cost."""
        return sympy.Array(
            s.x3_end ** 3 * s.p2 ** 3 + s.p1 * s.x1_start 
            + s.x1_end * s.x2_start ** 2
        )


@pytest.fixture(scope='module')
def model():
    """Optimal control collocation model."""
    symb_mdl = SymbolicModel()
    GeneratedModel = sym2num.model.compile_class('GeneratedModel', symb_mdl)
    return GeneratedModel()


@pytest.fixture(params=[1, 2, 4], ids=lambda i: f'{i}piece')
def npieces(request):
    """Number of collocation pieces."""
    return request.param


@pytest.fixture
def problem(model, npieces):
    """Optimal control problem."""
    t = np.linspace(0, 1, npieces + 1)
    return oc.Problem(model, t)
