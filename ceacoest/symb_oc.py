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
                        ('de_dp', 'e', 'p'),
                        ('d2e_dp2', 'de_dp', 'p'),
                        ('d2e_dxp_dup', 'de_dxp', 'up_flat'),
                        ('d2e_dxp_dp', 'de_dxp', 'p'),
                        ('d2e_dup_dp', 'de_dup', 'p'),
                        ('dg_dx', 'g', 'x'), 
                        ('dg_du', 'g', 'u'),
                        ('dg_dp', 'g', 'p'),
                        ('d2g_dx2', 'dg_dx',  'x'), 
                        ('d2g_du2', 'dg_du',  'u'),
                        ('d2g_dp2', 'dg_dp',  'p'),
                        ('d2g_dx_du', 'dg_dx', 'u'),
                        ('d2g_dx_dp', 'dg_dx', 'p'),
                        ('d2g_du_dp', 'dg_du', 'p'),
                        ('dh_dxe', 'h', 'xe_flat'),
                        ('dh_dp', 'h', 'p'),
                        ('d2h_dxe2', 'dh_dxe',  'xe_flat'),
                        ('d2h_dp2', 'dh_dp',  'p'),
                        ('d2h_dxe_dp', 'dh_dxe', 'p'),
                        ('dM_dxe', 'M', 'xe'), 
                        ('dM_dp', 'M', 'p'),
                        ('d2M_dxe2', 'M',  ('xe_flat', 'xe_flat')), 
                        ('d2M_dxe_dp', 'M', ('xe_flat', 'p')),
                        ('d2M_dp2', 'dM_dp',  'p')]
        return getattr(super(), 'derivatives', []) + derivatives
    
    @property
    def generate_functions(self):
        """List of the model functions to generate."""
        gen = ['e', 'f', 'g', 'h', 'M', 'dM_dp', 'dM_dxe']
        return getattr(super(), 'generate_functions', []) + gen
    
    @property
    def generate_sparse(self):
        """List of the model functions to generate in a sparse format."""
        gen = ['de_dxp', 'de_dup', 'de_dp',
               'd2e_dxp_dup', 'd2e_dxp_dp', 'd2e_dup_dp',
               ('d2e_dxp2', lambda i,j,k: i<=j),
               ('d2e_dup2', lambda i,j,k: i<=j),
               ('d2e_dp2', lambda i,j,k: i<=j),
               'dg_dx', 'dg_du', 'dg_dp',
               'd2g_dx_du', 'd2g_dx_dp', 'd2g_du_dp',
               ('d2g_dx2', lambda i,j,k: i<=j),
               ('d2g_du2', lambda i,j,k: i<=j),
               ('d2g_dp2', lambda i,j,k: i<=j),
               'd2M_dxe_dp',
               ('d2M_dxe2', lambda i,j: i<=j),
               ('d2M_dp2', lambda i,j: i<=j),
               'dh_dxe', 'dh_dp', 'd2h_dp2', 'd2h_dxe_dp', 
               ('d2h_dxe2', lambda i,j,k: i<=j),
               ('d2h_dp2', lambda i,j,k: i<=j)]
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
        
        additional_variables = sym2num.var.make_dict(
            [sym2num.var.SymbolArray('piece_len'),
             sym2num.var.SymbolArray('xp', xp),
             sym2num.var.SymbolArray('up', up),
             sym2num.var.SymbolArray('xe', xe),
             sym2num.var.SymbolArray('xp_flat', sympy.flatten(xp)),
             sym2num.var.SymbolArray('up_flat', sympy.flatten(up)),
             sym2num.var.SymbolArray('xe_flat', sympy.flatten(xe))]
        )
        return dict(**v, **additional_variables)
    
    def e(self, xp, up, p, piece_len):
        """Collocation defects (error)."""
        fp = sympy.Matrix([self.f(xp[i, :], up[i, :], p)
                           for i in range(self.collocation.n)])
        J = sympy.Matrix(self.collocation.J)
        dt = piece_len[()]
        
        xp = xp.tomatrix()
        defects = xp[1:, :] - xp[:-1, :] - dt * J * fp
        return sympy.Array(defects, len(defects))


def collocate(order=2):
    def decorator(BaseModel):
        @functools.wraps(BaseModel, updated=())
        class OptimalControlModel(CollocatedModel, BaseModel):
            collocation_order = order
        return OptimalControlModel
    return decorator
