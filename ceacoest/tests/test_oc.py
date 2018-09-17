"""Optimal control test module."""


import functools

import numpy as np
import pytest
import sympy
import sym2num.model
import sym2num.var

from ceacoest import oc
from ceacoest.modelling import symoc
from .test_optim import (test_merit_gradient, test_merit_hessian, 
                         test_constraint_jacobian, test_constraint_hessian,
                         seed, dec)


from ceacoest.testsupport.array_cmp import ArrayDiff


@pytest.fixture(scope='module', params=[2, 4], ids=lambda i: f'{i}ord-col')
def collocation_order(request):
    """Order of collocation method."""
    return request.param


@pytest.fixture(scope='module')
def model(collocation_order):
    """Optimal control collocation model."""
    
    @symoc.collocate(order=collocation_order)
    class SymbolicModel:
        """Symbolic optimal control test model."""
        
        @property
        @functools.lru_cache()
        def variables(self):
            """Model variables definition."""
            var_list = [
                sym2num.var.SymbolArray('x', ['x1', 'x2', 'x3']),
                sym2num.var.SymbolArray('u', ['u1', 'u2']),
                sym2num.var.SymbolArray('p', ['p1', 'p2']),
            ]
            return sym2num.var.make_dict(var_list)
        
        @sym2num.model.collect_symbols
        def f(self, x, u, p, *, s):
            """ODE function."""
            return sympy.Array(
                [s.x1 * sympy.cos(s.x2), 
                 s.u1 * s.u2 * s.p1 * s.x2**2, 
                 s.u1**2 * s.x1**3 * sympy.exp(s.p1)]
            )
        
        @sym2num.model.collect_symbols
        def g(self, x, u, p, *, s):
            """Path constraints."""
            return sympy.Array(
                [(s.u1 ** 2 + 2) **  (s.x1** 2 + 1) * s.p1,
                 s.p1 ** 3 * s.p2 * s.u1 ** 2 * s.p2])
        
        @sym2num.model.collect_symbols
        def h(self, xe, p, *, s):
            """Endpoint constraints."""
            return sympy.Array(
                [s.x1_initial * s.p1 ** 3, 
                 s.x1_initial * s.x2_final ** 3 * s.p1])
        
        @sym2num.model.collect_symbols
        def M(self, xe, p, *, s):
            """Mayer (endpoint) cost."""
            return sympy.Array(
                s.x3_final ** 3 * s.p2 ** 3 + s.p1 * s.x1_initial 
                + s.x1_final * s.x2_initial ** 2
            )
        
        @sym2num.model.collect_symbols
        def L(self, x, u, p, *, s):
            """Lagrange (running) cost."""
            L = (s.x1 ** 2 + s.p2**4 + s.p1 * s.u2 * (s.x1 + 2)
                 + (s.p2 + s.u1 + s.u2 + s.x1) ** 2)
            return sympy.Array(L)
        
        
    symb_mdl = SymbolicModel()
    GeneratedModel = sym2num.model.compile_class(symb_mdl)
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
