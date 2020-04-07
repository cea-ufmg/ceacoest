"""Auxiliary code for generated optimization models."""


import collections
import inspect
import types

import numpy as np


def optimization_meta(name, bases, dict):
    cls = type(name, bases, dict)
    base_shapes = cls.base_shapes
    
    for name, desc in cls.constraints.items():
        method = getattr(cls, name)
        constraint_obj = ConstraintFunctionMeta(
            name, method, desc, cls
        )
        setattr(cls, name, constraint_obj)
    
    return cls


class OptimizationFunction:
    def __init__(self, model):
        self.model = model
        """The parent model."""
    
    def __call__(self, *args, **kwargs):
        return self.method(self.model, *args, **kwargs)

    def _sparse_deriv_ind(self, deriv, dec_ext={}, out_ext=()):
        ret = collections.OrderedDict()
        for wrt, dname in deriv.items():
            ind = []
            base_ind = getattr(self.model, f'{dname}_ind')
            for wrt_name, wrt_ind in zip(wrt, base_ind):
                wrt_sz = shape_size(self.model.base_shapes[wrt_name])
                out_sz = self.out_sz
                wrt_ext = dec_ext.get(wrt_name, ())
            
                wrt_offs = ndim_range(wrt_ext) * np.ones(out_ext, int) * wrt_sz
                ind.append(wrt_ind + wrt_offs[..., None])
            
            # Extend the output indices
            out_ind = base_ind[-1]
            out_offs = ndim_range(out_ext) * out_sz
            ind.append(out_ind + out_offs[..., None])
            
            # Save in dictionary
            ret[wrt] = np.array(ind)
        return ret

    
class ConstraintFunction(OptimizationFunction):
    def jac_ind(self, dec_ext={}, out_ext=()):
        ret = collections.OrderedDict()
        for wrt, dname in self._jac.items():
            wrt_ind, out_ind = getattr(self.model, f'{dname}_ind')
            wrt_sz = shape_size(self.ModelClass.base_shapes[wrt])
            out_sz = self.out_sz
            wrt_ext = dec_ext.get(wrt, ())

            wrt_offsets = ndim_range(wrt_ext) * np.ones(out_ext, int) * wrt_sz
            out_offsets = ndim_range(out_ext) * out_sz
            
            key = wrt, type(self).__name__
            ret[key] = np.array([wrt_ind + wrt_offsets[..., None],
                                 out_ind + out_offsets[..., None]])
        return ret
    
    def hess_ind(self, dec_ext={}, out_ext=()):
        ret = collections.OrderedDict()
        for (wrt0, wrt1), dname in self._hess.items():
            wrt0_ind, wrt1_ind, out_ind = getattr(self.model, f'{dname}_ind')
            wrt0_sz = shape_size(self.ModelClass.base_shapes[wrt0])
            wrt1_sz = shape_size(self.ModelClass.base_shapes[wrt1])
            out_sz = self.out_sz
            wrt0_ext = dec_ext.get(wrt0, ())
            wrt1_ext = dec_ext.get(wrt1, ())

            wrt0_off = ndim_range(wrt0_ext) * np.ones(out_ext, int) * wrt0_sz
            wrt1_off = ndim_range(wrt1_ext) * np.ones(out_ext, int) * wrt1_sz
            out_off = ndim_range(out_ext) * out_sz
            
            key = wrt0, wrt1, type(self).__name__
            ret[key] = np.array([wrt0_ind + wrt0_off[..., None],
                                 wrt1_ind + wrt1_off[..., None],
                                 out_ind + out_off[..., None]])
        return ret


class OptimizationFunctionMeta(type):
    bases = OptimizationFunction,
    """Bases of generated class."""
    
    def __new__(cls, name, method, desc, ModelClass):
        return super().__new__(cls, name, cls.bases, {})
    
    def __init__(self, name, method, desc, ModelClass):
        self.method = staticmethod(method)
        """The underlying callable optimization function."""
        
        self.__signature__ = bound_signature(method)
        """The class call signature."""
       
        self.ModelClass = ModelClass
        """The underlying model class."""
    
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        else:
            return self(instance)


class ConstraintFunctionMeta(OptimizationFunctionMeta):

    bases = ConstraintFunction,
    """Bases of generated class."""
    
    def __init__(self, name, method, desc, ModelClass):
        # Initialize base class
        super().__init__(name, method, desc, ModelClass)

        self.out_sz = shape_size(desc['shape'])
        """Constraint function output base size."""
        
        self._jac = collections.OrderedDict(desc['der1'])
        """First derivatives."""

        self._hess = collections.OrderedDict(desc['der2'])
        """Second derivatives."""


def bound_signature(method):
    """Return the signature of a method when bound."""
    sig = inspect.signature(method)
    param = list(sig.parameters.values())[1:]
    return inspect.Signature(param, return_annotation=sig.return_annotation)


def shape_size(shape):
    return np.prod(shape, dtype=int)


def ndim_range(shape):
    assert isinstance(shape, tuple)
    return np.arange(shape_size(shape)).reshape(shape)
