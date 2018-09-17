"""Symbolic model building for output error method (OEM) estimation."""


import collections
import functools

import numpy as np
import sym2num.var
import sympy

from . import symcol
from .. import utils


class ModelSubclass(symcol.CollocatedModel):
    """Symbolic LGL-collocation output error method model subclass."""
    
    @property
    def derivatives(self):
        """List of the model function derivatives to calculate."""
        derivatives =  [('dL_dxm', 'L', 'xm'),
                        ('dL_dp', 'L', 'p'),
                        ('d2L_dxm2', 'dL_dxm', 'xm'),
                        ('d2L_dp2', 'dL_dp', 'p'),
                        ('d2L_dxm_dp', 'dL_dxm', 'p')]
        return getattr(super(), 'derivatives', []) + derivatives
    
    @property
    def generate_functions(self):
        """Iterable of the model functions to generate."""
        gen = {'L', 'dL_dxm', 'dL_dp'}
        return getattr(super(), 'generate_functions', set()) | gen
    
    @property
    def generate_sparse(self):
        """List of the model functions to generate in a sparse format."""
        gen = [('d2L_dxm2', lambda i,j: i<=j),
               ('d2L_dp2', lambda i,j: i<=j),
               'd2L_dxm_dp']
        return getattr(super(), 'generate_sparse', []) + gen
    
    @property
    def generate_assignments(self):
        return {'ny': len(self.variables['y']),
                **getattr(super(), 'generate_assignments', {})}
    
    @utils.cached_property
    def variables(self):
        """Model variables definition."""
        v = super().variables
        additional_vars = sym2num.var.make_dict(
            [sym2num.var.SymbolArray('xm', v['x']),
             sym2num.var.SymbolArray('um', v['u'])]
        )
        return collections.OrderedDict([*v.items(), *additional_vars.items()])


def collocate(order=2):
    def decorator(BaseModel):
        @functools.wraps(BaseModel, updated=())
        class OptimalControlModel(ModelSubclass, BaseModel):
            collocation_order = order
        return OptimalControlModel
    return decorator
