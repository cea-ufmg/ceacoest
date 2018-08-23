"""Symbolic model building for optimal control."""


import functools

import numpy as np
import sym2num.model
import sym2num.var
import sympy

from . import utils, rk


class CollocatedModel(sym2num.model.Base):
    """Symbolic LGL-collocation optimal control model subclass."""
    
    @property
    def derivatives(self):
        """List of the model function derivatives to calculate."""
        derivatives =  [('de_dxp', 'e', 'xp_flat'),
                        ('d2e_dxp2', 'de_dxp', 'xp_flat'),
                        ('de_dup', 'e', 'up_flat'),
                        ('d2e_dup2', 'de_dup', 'up_flat'),
                        ('d2e_dxp_dup', 'de_dxp', 'up_flat'),
                        ('dg_dx', 'g', 'x'), 
                        ('dg_du', 'g', 'u'),
                        ('d2g_dx2', 'dg_dx',  'x'), 
                        ('d2g_dx_du', 'dg_dx', 'u'),
                        ('d2g_du2', 'dg_du',  'u'),
                        ('dh_dxe', 'h', 'xe_flat'),
                        ('dh_dT', 'h', 'T_flat'),
                        ('d2h_dxe2', 'dh_dxe',  'xe_flat'), 
                        ('d2h_dxe_dT', 'dh_dxe', 'T'),
                        ('d2h_dT2', 'dh_dT',  'T'),
                        ('dM_dxe', 'M', 'xe'), 
                        ('dM_dT', 'M', 'T'),
                        ('d2M_dxe2', 'M',  ('xe_flat', 'xe_flat')), 
                        ('d2M_dxe_dT', 'M', ('xe_flat', 'T')),
                        ('d2M_dT2', 'dM_dT',  'T')]
        return getattr(super(), 'derivatives', []) + derivatives
    
    @property
    def generate_functions(self):
        """List of the model functions to generate."""
        gen = ['e', 'f', 'g', 'h', 'M', 'd2M_dT2', 'dM_dT', 'dM_dxe']
        return getattr(super(), 'generate_functions', []) + gen
    
    @property
    def generate_sparse(self):
        """List of the model functions to generate in a sparse format."""
        gen = ['de_dxp', 'de_dup', 'd2e_dxp_dup',
               ('d2e_dxp2', lambda i,j,k: i<=j), 
               ('d2e_dup2', lambda i,j,k: i<=j),
               'dg_dx', 'dg_du', 'd2g_dx_du',
               ('d2g_dx2', lambda i,j,k: i<=j), 
               ('d2g_du2', lambda i,j,k: i<=j),
               'd2M_dxe_dT', 
               ('d2M_dxe2', lambda i,j: i<=j),
               'dh_dxe', 'dh_dT', 'd2h_dT2', 'd2h_dxe_dT', 
               ('d2h_dxe2', lambda i,j,k: i<=j)]
        return getattr(super(), 'generate_sparse', []) + gen
    
    @property
    def generate_assignments(self):
        gen = dict(nx=len(self.variables['x']),
                   nu=len(self.variables['u']),
                   np=len(self.variables['p']),
                   ne=len(self.default_function_output('e')),
                   ng=len(self.default_function_output('g')),
                   nh=len(self.default_function_output('h')),
                   collocation_order=self.collocation.n,
                   **getattr(super(), 'generate_assignments', {}))
        return gen
    
    @utils.cached_property
    def collocation(self):
        """Collocation method."""
        collocation_order = getattr(self, 'collocation_order', 2)
        return rk.LGLCollocation(collocation_order)
    
    @utils.cached_property
    def variables(self):
        """Model variables definition."""
        v = super().variables
        ncol = self.collocation.n

        x = [xi.name for xi in v['x']]
        u = [ui.name for ui in v['u']]
        
        # Piece states and controls
        xp = [[f'{n}_piece_{k}' for n in x] for k in range(ncol)]
        up = [[f'{n}_piece_{k}' for n in u] for k in range(ncol)]

        # Endpoint states
        xe = [[f'{n}_start' for n in x], [f'{n}_end' for n in x]]
        
        additional_variables = sym2num.model.make_variables_dict(
            [sym2num.var.SymbolArray('T'),
             sym2num.var.SymbolArray('piece_len'),
             sym2num.var.SymbolArray('xp', xp),
             sym2num.var.SymbolArray('up', up),
             sym2num.var.SymbolArray('xe', xe),
             sym2num.var.SymbolArray('xp_flat', sympy.flatten(xp)),
             sym2num.var.SymbolArray('up_flat', sympy.flatten(up)),
             sym2num.var.SymbolArray('xe_flat', sympy.flatten(xe)),
             sym2num.var.SymbolArray('T_flat', ['T'])]
        )
        return dict(**v, **additional_variables)
    
    def e(self, xp, up, p, T, piece_len):
        """Collocation defects (error)."""
        fp = sympy.Matrix([self.f(xp[i, :], up[i, :], p)
                           for i in range(self.collocation.n)])
        J = sympy.Matrix(self.collocation.J)
        dt = sympy.Array(np.diff(self.collocation.points))*piece_len[()]*T[()]
        
        xp = xp.tomatrix()
        defects = xp[1:, :] - xp[:-1, :] - sympy.diag(*dt)*J*fp
        return sympy.Array(defects, len(defects))


def collocate(order=2):
    def decorator(BaseModel):
        @functools.wraps(BaseModel, updated=())
        class OptimalControlModel(CollocatedModel, BaseModel):
            collocation_order = order
        return OptimalControlModel
    return decorator
