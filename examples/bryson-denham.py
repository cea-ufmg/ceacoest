"""Bryson--Denham optimal control problem."""


import numpy as np
import sym2num

from ceacoest import oc


class SymbolicModel(sym2num.SymbolicModel):
    """Symbolic Bryson--Denham optimal control model."""
    
    var_names = {'x', 'u', 'tf', 'xe'}
    """Name of model variables."""
    
    function_names = {'f', 'g', 'h', 'M'}
    """Name of the model functions."""
    
    derivatives = [('df_dx', 'f', 'x'), ('df_du', 'f', 'u'),
                   ('d2f_dx2', 'df_dx',  'x'), 
                   ('d2f_dx_du', 'df_dx', 'u'),
                   ('d2f_du2', 'df_du',  'u'),
                   ('dg_dx', 'g', 'x'), ('dg_du', 'g', 'u'),
                   ('d2g_dx2', 'dg_dx',  'x'), 
                   ('d2g_dx_du', 'dg_dx', 'u'),
                   ('d2g_du2', 'dg_du',  'u'),
                   ('dh_dx', 'h', 'xe'), ('dh_dt', 'h', 'tf'),
                   ('d2h_dx2', 'dh_dx',  'xe'), 
                   ('d2h_dx_dt', 'dh_dx', 'tf'),
                   ('d2h_dt2', 'dh_dt',  'tf'),
                   ('dM_dx', 'M', 'xe'), ('dM_dt', 'M', 'tf'),
                   ('d2M_dx2', 'dM_dx',  'xe'), 
                   ('d2M_dx_dt', 'dM_dx', 'tf'),
                   ('d2M_dt2', 'dM_dt',  'tf')]
    """List of the model function derivatives to calculate / generate."""

    sparse = ['df_dx', 'df_du', 'd2f_dx_du',
              ('d2f_dx2', lambda i,j,k: i<=j), ('d2f_du2', lambda i,j,k: i<=j),
              'd2M_dx_dt', ('d2M_dx2', lambda i,j: i<=j)]
    """List of the model functions to generate in a sparse format."""

    tf = 'tf'
    """Final time."""
    
    x = ['x1', 'x2', 'x3']
    """State vector."""
    
    u = ['u']
    """Control vector."""
    
    xe = [xi + '_start' for xi in x] + [xi + '_end' for xi in x]
    """State vector at endpoints."""
    
    def f(self, x, u):
        """ODE function."""
        s = self.symbols(x=x, u=u)
        return [s.x2, s.u, 0.5*s.u**2]
    
    def g(self, x, u):
        """Path constraints."""
        s = self.symbols(x=x, u=u)
        return []
    
    def h(self, xe, tf):
        """Endpoint constraints."""
        s = self.symbols(xe=xe, tf=tf)
        return [s.x1_start**3 + s.x1_start, s.x1_end**3 + s.x1_end]
    
    def M(self, xe, tf):
        """Mayer (endpoint) cost."""
        s = self.symbols(xe=xe, tf=tf)
        return s.x3_end


sym_model = SymbolicModel()
printer = sym2num.ScipyPrinter()
GeneratedModel = sym2num.class_obj(sym_model, printer)
GeneratedModel.nx = len(sym_model.x)
GeneratedModel.nu = len(sym_model.u)
GeneratedModel.ng = len(sym_model.functions['g'].out)
GeneratedModel.nh = len(sym_model.functions['h'].out)


if __name__ == '__main__':
    t = np.linspace(0, 1, 3)
    model = GeneratedModel()
    problem = oc.Problem(model, t)
    
