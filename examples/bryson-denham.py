"""Bryson--Denham optimal control problem."""


import numpy as np
import sym2num

from ceacoest import oc


class SymbolicModel(sym2num.SymbolicModel):
    """Symbolic Bryson--Denham optimal control model."""
    
    generated_name = 'GeneratedModel'
    
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
              'dg_dx', 'dg_du', 'd2g_dx_du',
              ('d2g_dx2', lambda i,j,k: i<=j), ('d2g_du2', lambda i,j,k: i<=j),
              'd2M_dx_dt', ('d2M_dx2', lambda i,j: i<=j),
              'dh_dx', 'dh_dt', 'd2h_dt2', 'd2h_dx_dt', 
              ('d2h_dx2', lambda i,j,k: i<=j)]
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
        return []
    
    def M(self, xe, tf):
        """Mayer (endpoint) cost."""
        s = self.symbols(xe=xe, tf=tf)
        return s.x3_end

    meta = 'sym2num.model.ParametrizedModel.meta'
    
    @property
    def imports(self):
        return super().imports + ('import sym2num.model',)


sym_model = SymbolicModel()
printer = sym2num.ScipyPrinter()
GeneratedModel = sym2num.class_obj(sym_model, printer)
GeneratedModel.nx = len(sym_model.x)
GeneratedModel.nu = len(sym_model.u)
GeneratedModel.ng = len(sym_model.functions['g'].out)
GeneratedModel.nh = len(sym_model.functions['h'].out)


if __name__ == '__main__':
    t = np.linspace(0, 1, 50)
    model = GeneratedModel()
    xe_fix = dict(x1_start=0, x2_start=1, x3_start=0, x1_end=0, x2_end=-1)
    problem = oc.Problem(model, t)
    problem.set_x_bounds({'x1': 0}, {'x1': 1/9})
    problem.set_xe_bounds(xe_fix)
    problem.d_bounds[:, -1] = 1
    
    d0 = np.zeros(problem.nd)
    d0[-1] = 1
    nlp = problem.nlp_yaipopt()
    dopt, solinfo = nlp.solve(d0)
    xopt, uopt, tfopt = problem.unpack_decision(dopt)
    topt = problem.tc * tfopt
