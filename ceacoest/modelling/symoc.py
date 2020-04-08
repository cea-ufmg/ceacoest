"""Symbolic model building for optimal control."""


import collections
import functools

import numpy as np
import sym2num.model
import sym2num.var
import sympy

from . import symcol
from .. import utils


class OCModel(symcol.CollocatedModel):
    def __init__(self, variables, decision=set()):
        variables.setdefault('p', [])
        variables.setdefault('u', [])
        decision = {'p', 'u', 'up', *decision}
        super().__init__(variables, decision)
        
        # Define endpoint variables
        x = self.variables['x']
        xe = [[f'{n}_initial' for n in x], [f'{n}_final' for n in x]]
        self.variables['xe'] = xe
    
        # Add objectives and constraints
        self.add_objective('IL')
        self.add_objective('M')
        self.add_constraint('g')
        self.add_constraint('h')
    
    def IL(self, xp, up, p, piece_len):
        """Integral of the Lagrangian (total running cost)."""
        ncol = self.collocation.n
        Lp = np.array([self.L(xp[i,:], up[i,:], p) for i in range(ncol)])
        K = self.collocation.K
        dt = piece_len
        IL = Lp @ K * piece_len
        return IL
    
    def g(self, x, u, p):
        """Default (empty) path constraint."""
        return np.array([])
    
    def h(self, xe, p):
        """Default (empty) endpoint constraint."""
        return np.array([])
    
    def L(self, x, u, p):
        """Default (null) Lagrange (running) cost."""
        return np.array(0)
    
    def M(self, xe, p):
        """Default (null) Mayer (endpoint) cost."""
        return np.array(0)


class OldModelSubclass(symcol.OldCollocatedModel):
    """Symbolic LGL-collocation optimal control model subclass."""
    
    @property
    def derivatives(self):
        """List of the model function derivatives to calculate."""
        derivatives =  [('de_dup', 'e', 'up_flat'),
                        ('d2e_dup2', 'de_dup', 'up_flat'),
                        ('d2e_dxp_dup', 'de_dxp', 'up_flat'),
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
                        ('d2M_dp2', 'dM_dp',  'p'),
                        ('dIL_dxp', 'IL', 'xp'),
                        ('dIL_dup', 'IL', 'up'),
                        ('dIL_dp', 'IL', 'p'),
                        ('d2IL_dxp2', 'IL', ('xp_flat', 'xp_flat')),
                        ('d2IL_dup2', 'IL', ('up_flat', 'up_flat')),
                        ('d2IL_dp2', 'dIL_dp', 'p'),
                        ('d2IL_dxp_dup', 'IL', ('xp_flat', 'up_flat')),
                        ('d2IL_dxp_dp', 'IL', ('xp_flat', 'p')),
                        ('d2IL_dup_dp', 'IL', ('up_flat', 'p'))]
        return getattr(super(), 'derivatives', []) + derivatives
    
    @property
    def generate_functions(self):
        """Iterable of the model functions to generate."""
        gen = {'g', 'h', 'M', 'dM_dp', 'dM_dxe',
               'IL', 'dIL_dxp', 'dIL_dup', 'dIL_dp'}
        return getattr(super(), 'generate_functions', set()) | gen
    
    @property
    def generate_sparse(self):
        """List of the model functions to generate in a sparse format."""
        gen = ['de_dup', 'd2e_dxp_dup', 'd2e_dup_dp',
               ('d2e_dup2', lambda i,j,k: i<=j),
               'dg_dx', 'dg_du', 'dg_dp',
               'd2g_dx_du', 'd2g_dx_dp', 'd2g_du_dp',
               ('d2g_dx2', lambda i,j,k: i<=j),
               ('d2g_du2', lambda i,j,k: i<=j),
               ('d2g_dp2', lambda i,j,k: i<=j),
               'd2M_dxe_dp',
               ('d2M_dxe2', lambda i,j: i<=j),
               ('d2M_dp2', lambda i,j: i<=j),
               ('d2IL_dxp2', lambda i,j: i<=j),
               ('d2IL_dup2', lambda i,j: i<=j),
               ('d2IL_dp2', lambda i,j: i<=j),
               'd2IL_dxp_dup', 'd2IL_dxp_dp', 'd2IL_dup_dp',
               'dh_dxe', 'dh_dp', 'd2h_dp2', 'd2h_dxe_dp', 
               ('d2h_dxe2', lambda i,j,k: i<=j),
               ('d2h_dp2', lambda i,j,k: i<=j)]
        return getattr(super(), 'generate_sparse', []) + gen
    
    @property
    def generate_assignments(self):
        gen = {'ng': len(self.default_function_output('g')),
               'nh': len(self.default_function_output('h')),
               **getattr(super(), 'generate_assignments', {})}
        return gen
        
    @utils.cached_property
    def variables(self):
        """Model variables definition."""
        v = super().variables
        x = [xi.name for xi in v['x']]
        
        # Endpoint states
        xe = [[f'{n}_initial' for n in x], [f'{n}_final' for n in x]]
        
        additional_vars = sym2num.var.make_dict(
            [sym2num.var.SymbolArray('xe', xe),
             sym2num.var.SymbolArray('xe_flat', sympy.flatten(xe))]
        )
        return collections.OrderedDict([*v.items(), *additional_vars.items()])
    
    def IL(self, xp, up, p, piece_len):
        """Integral of the Lagrangian (total running cost)."""
        ncol = self.collocation.n
        Lp = [self.L(xp[i,:], up[i,:], p)[()] for i in range(ncol)]
        K = self.collocation.K
        dt = piece_len[()]
        IL = sum(Lp[i] * K[i] * dt for i in range(ncol))
        return sympy.Array(IL)


def collocate(order=2):
    def decorator(BaseModel):
        @functools.wraps(BaseModel, updated=())
        class OptimalControlModel(ModelSubclass, BaseModel, sym2num.model.Base):
            collocation_order = order
        return OptimalControlModel
    return decorator
