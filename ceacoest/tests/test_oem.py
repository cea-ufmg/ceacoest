"""Output error method test module."""


import functools

import numpy as np
import pytest
import sympy
import sym2num.model
import sym2num.var

from ceacoest import oem
from ceacoest.modelling import symoem
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
    """Output error method collocation model."""
    
    @symoem.collocate(order=collocation_order)
    class SymbolicModel:
        """Symbolic output error method test model."""
        
        @property
        @functools.lru_cache()
        def variables(self):
            """Model variables definition."""
            var_list = [
                sym2num.var.SymbolArray('x', ['x1', 'x2', 'x3']),
                sym2num.var.SymbolArray('u', ['u1']),
                sym2num.var.SymbolArray('y', ['x1_meas', 'x2_meas']),
                sym2num.var.SymbolArray('p', ['p1', 'p2']),
            ]
            return sym2num.var.make_dict(var_list)
        
        @sym2num.model.collect_symbols
        def f(self, x, u, p, *, s):
            """ODE function."""
            return sympy.Array(
                [s.x1 * sympy.cos(s.x2), 
                 s.u1 * s.p1 * s.x2**2, 
                 s.u1**2 * s.x1**3 * sympy.exp(s.p1)]
            )
                
        @sym2num.model.collect_symbols
        def L(self, y, x, u, p, *, s):
            """Lagrange (running) cost."""
            L = (s.x1 ** 2 + s.p2**4 + s.p1 * (s.x1 + 2)
                 + (s.p2 + s.u1 + s.u1 + s.x1) ** 2 ) + s.x1_meas
            return sympy.Array(L)
    
    symb_mdl = SymbolicModel()
    GeneratedModel = sym2num.model.compile_class(symb_mdl)
    return GeneratedModel()


@pytest.fixture(params=[1, 2, 4], ids=lambda i: f'{i}piece')
def npieces(request):
    """Number of collocation pieces."""
    return request.param


@pytest.fixture
def y(model, seed, npieces):
    """OEM measurements."""
    return np.random.randn(npieces + 1, model.ny)


@pytest.fixture
def u():
    """OEM input function."""
    return lambda t: np.cos(t)[:, None]


@pytest.fixture
def problem(model, npieces, y, u):
    """Output error method problem."""
    t = np.linspace(0, 1, npieces + 1)
    return oem.Problem(model, t, y, u)
