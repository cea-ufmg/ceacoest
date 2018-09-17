"""Symbolic collocation model common code."""


import collections
import functools

import numpy as np
import sym2num.model
import sym2num.var
import sympy

from .. import utils, rk


class CollocatedModel(sym2num.model.Base):
    """Symbolic LGL-collocation model base."""
    
    @property
    def derivatives(self):
        """List of the model function derivatives to calculate."""
        derivatives =  [('de_dxp', 'e', 'xp_flat'),
                        ('d2e_dxp2', 'de_dxp', 'xp_flat'),
                        ('de_dp', 'e', 'p'),
                        ('d2e_dp2', 'de_dp', 'p'),
                        ('d2e_dxp_dp', 'de_dxp', 'p')]
        return getattr(super(), 'derivatives', []) + derivatives
    
    @property
    def generate_functions(self):
        """Iterable of the model functions to generate."""
        gen = {'e', 'f'}
        return getattr(super(), 'generate_functions', set()) | gen
    
    @property
    def generate_sparse(self):
        """List of the model functions to generate in a sparse format."""
        gen = ['de_dxp', 'de_dp', 'd2e_dxp_dp', 
               ('d2e_dxp2', lambda i,j,k: i<=j),
               ('d2e_dp2', lambda i,j,k: i<=j)]
        return getattr(super(), 'generate_sparse', []) + gen
    
    @property
    def generate_assignments(self):
        gen = {'nx': len(self.variables['x']),
               'nu': len(self.variables['u']),
               'np': len(self.variables['p']),
               'ne': len(self.default_function_output('e')),
               'collocation_order': self.collocation.n,
               'symbol_index_map': self.symbol_index_map,
               'array_shape_map': self.array_shape_map,
               **getattr(super(), 'generate_assignments', {})}
        return gen

    @property
    def generate_imports(self):
        """List of imports to include in the generated class code."""
        return ['sym2num.model'] + getattr(super(), 'generate_imports', [])

    @property
    def generated_bases(self):
        """Base classes of the generated model class."""
        bases = ['sym2num.model.ModelArrayInitializer']
        return bases + getattr(super(), 'generated_bases', [])

    
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
        
        additional_vars = sym2num.var.make_dict(
            [sym2num.var.SymbolArray('piece_len'),
             sym2num.var.SymbolArray('xp', xp),
             sym2num.var.SymbolArray('up', up),
             sym2num.var.SymbolArray('xp_flat', sympy.flatten(xp)),
             sym2num.var.SymbolArray('up_flat', sympy.flatten(up))]
        )
        return collections.OrderedDict([*v.items(), *additional_vars.items()])
    
    def e(self, xp, up, p, piece_len):
        """Collocation defects (error)."""
        ncol = self.collocation.n
        fp = sympy.Matrix([self.f(xp[i, :], up[i, :], p) for i in range(ncol)])
        J = sympy.Matrix(self.collocation.J)
        dt = piece_len[()]
        
        xp = xp.tomatrix()
        defects = xp[1:, :] - xp[:-1, :] - dt * J * fp
        return sympy.Array(defects, len(defects))
