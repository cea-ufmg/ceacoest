"""Bryson--Denham optimal control problem."""


import numpy as np
import sympy

from sym2num import model, utils, var


class BrysonDenham(model.Base):
    """Symbolic Bryson--Denham optimal control model."""
    
    derivatives = [('df_dx', 'f', 'x'), ('df_du', 'f', 'u'),
                   ('d2f_dx2', 'df_dx',  'x'), 
                   ('d2f_dx_du', 'df_dx', 'u'),
                   ('d2f_du2', 'df_du',  'u'),
                   ('dg_dx', 'g', 'x'), ('dg_du', 'g', 'u'),
                   ('d2g_dx2', 'dg_dx',  'x'), 
                   ('d2g_dx_du', 'dg_dx', 'u'),
                   ('d2g_du2', 'dg_du',  'u'),
                   ('dh_dx', 'h', 'xe'), ('dh_dT', 'h', 'T'),
                   ('d2h_dx2', 'dh_dx',  'xe'), 
                   ('d2h_dx_dT', 'dh_dx', 'T'),
                   ('d2h_dT2', 'dh_dT',  'T'),
                   ('dM_dx', 'M', 'xe'), ('dM_dT', 'M', 'T'),
                   ('d2M_dx2', 'dM_dx',  'xe'), 
                   ('d2M_dx_dT', 'dM_dx', 'T'),
                   ('d2M_dT2', 'dM_dT',  'T')]
    """List of the model function derivatives to calculate."""
    
    generate_functions = ['f', 'g', 'h', 'M', 'd2M_dT2', 'dM_dT']
    """List of the model functions to generate."""

    generate_sparse = ['df_dx', 'df_du', 'd2f_dx_du',
              ('d2f_dx2', lambda i,j,k: i<=j), ('d2f_du2', lambda i,j,k: i<=j),
              'dg_dx', 'dg_du', 'd2g_dx_du',
              ('d2g_dx2', lambda i,j,k: i<=j), ('d2g_du2', lambda i,j,k: i<=j),
              'd2M_dx_dT', ('d2M_dx2', lambda i,j: i<=j),
              'dh_dx', 'dh_dT', 'd2h_dT2', 'd2h_dx_dT', 
              ('d2h_dx2', lambda i,j,k: i<=j)]
    """List of the model functions to generate in a sparse format."""
    
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
                    nu=len(self.variables['u']),
                    ng=len(self.default_function_output('g')),
                    nh=len(self.default_function_output('h')))
    
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
