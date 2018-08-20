"""Symbolic model building for optimal control."""


import numpy as np
import sym2num.model
import sym2num.var
import sympy

from . import utils, rk


class Model(sym2num.model.Base):
    """Symbolic LGL-collocation optimal control model subclass."""
    
    @utils.cached_property
    def collocation(self):
        """Collocation method."""
        collocation_order = getattr(self, 'collocation_order', 2)
        return rk.LGLCollocation(collocation_order)
    
    @utils.cached_property
    def variables(self):
        """Model variables definition."""
        v = super().variables()
        ncol = self.collocation.n

        x = [xi.name for xi in v['x']]
        u = [ui.name for ui in v['u']]
        
        # Piece states and controls
        xp = [[f'{n}_piece_{k}' for n in x] for k in range(ncol)]
        up = [[f'{n}_piece_{k}' for n in u] for k in range(ncol)]

        # Endpoint states
        xe = [[f'{n}_start' for n in x], [f'{n}_end' for n in x]]
        
        additional_variables = make_variables_dict(
            [sym2num.var.SymbolArray('T'),
             sym2num.var.SymbolArray('piece_len'),
             sym2num.var.SymbolArray('xp', xp),
             sym2num.var.SymbolArray('up', up),
             sym2num.var.SymbolArray('xe', xe),
             sym2num.var.SymbolArray('xp_flat', sympy.flatten(xp)),
             sym2num.var.SymbolArray('up_flat', sympy.flatten(up)),
             sym2num.var.SymbolArray('xe_flat', sympy.flatten(xe))]
        )
        return dict(**v, **aditional_variables)
    
    def e(self, xp, up, p, T, piece_len):
        fp = sympy.Matrix([self.f(xp[i, :], up[i, :], p) 
                           for i in range(self.collocation.n)])
        J = sympy.Matrix(self.collocation.J)
        
        defects = xp[1:, :] - xp[:-1, :] - J*fp*T[()]*piece_len[()]
        return sympy.Array(defects, len(defects))
