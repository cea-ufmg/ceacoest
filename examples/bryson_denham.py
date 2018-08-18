"""Bryson--Denham optimal control problem."""


import numpy as np
import sympy

from sym2num import model, utils, var


class BrysonDenham(model.Base):
    """Symbolic Bryson--Denham optimal control model."""
    
    derivatives = [('df_dx', 'f', 'x')]
    generate_functions = ['f', 'g', 'h', 'M', 'df_dx']
    generate_sparse = ['df_dx']
    
    @model.make_variables_dict
    def variables():
        """Model variables definition."""
        x = ['x1', 'x2', 'x3']
        xe = [xi + '_start' for xi in x] + [xi + '_end' for xi in x]
        return [
            var.SymbolArray('x', x),
            var.SymbolArray('xe', xe),
            var.SymbolArray('u', ['u1']),
            var.SymbolArray('T'),
        ]
    
    @property
    def generate_assignments(self):
        return dict(nx=len(self.variables['x']),
                    nu=len(self.variables['u']))
    
    @model.symbols_from('x, u')
    def f(self, s):
        """ODE function."""
        return sympy.Array([s.x2, s.u1, 0.5*s.u1**2])

    @model.symbols_from('x, u')
    def g(self, s):
        """Path constraints."""
        return sympy.Array([], 0)
    
    @model.symbols_from('xe, T')
    def h(self, s):
        """Endpoint constraints."""
        return sympy.Array([], 0)
    
    @model.symbols_from('xe, T')
    def M(self, s):
        """Mayer (endpoint) cost."""
        return sympy.Array(s.x3_end)


if __name__ == '__main__':
    symb_mdl = BrysonDenham()
    GeneratedBrysonDenham = model.compile_class(
        'GeneratedBrysonDenham', symb_mdl
    )
    mdl = GeneratedBrysonDenham()
