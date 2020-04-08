"""Auxiliary code for generated optimization models."""


import collections
import functools
import inspect
import types

import numpy as np


def optimization_meta(name, bases, dict):
    cls = type(name, bases, dict)
    base_shapes = cls.base_shapes
    
    for name, desc in cls.constraints.items():
        method = getattr(cls, name)
        constraint = ConstraintFunctionMeta(
            name, method, desc, cls
        )
        setattr(cls, name, constraint)

    for name, desc in cls.objectives.items():
        method = getattr(cls, name)
        objective = ObjectiveFunctionMeta(
            name, method, desc, cls
        )
        setattr(cls, name, objective)
    
    return cls


class OptimizationFunction:
    def __init__(self, model):
        self.model = model
        """The parent model."""
        
        self.__signature__ = bound_signature(self.method)
        """The object call signature."""
        
        # Assign descriptive signature to the sparse value functions
        method_sig = inspect.signature(self.method)
        self.hess_val = with_signature(self.hess_val, method_sig)
    
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
    
    def _sparse_deriv_val(self, deriv, *args, **kwargs):
        ret = collections.OrderedDict()
        for wrt, dname in deriv.items():
            ret[wrt] = getattr(self.model, f'{dname}_val')(*args, **kwargs)
        return ret
    
    def hess_ind(self, dec_ext={}, out_ext=()):
        return self._sparse_deriv_ind(self._hess, dec_ext, out_ext)

    def hess_val(self, *args, **kwargs):
        return self._sparse_deriv_val(self._hess, *args, **kwargs)


class OptimizationFunctionMeta(type):
    bases = OptimizationFunction,
    """Bases of generated class."""
    
    def __new__(cls, name, method, desc, ModelClass):
        return super().__new__(cls, name, cls.bases, {})
    
    def __init__(self, name, method, desc, ModelClass):
        self.method = staticmethod(method)
        """The underlying callable optimization function."""
               
        self.ModelClass = ModelClass
        """The underlying model class."""
        
        self._hess = collections.OrderedDict(desc['hess'])
        """Second derivatives."""
    
    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        else:
            return self(instance)


class ObjectiveFunction(OptimizationFunction):
    def grad(self, *args, **kwargs):
        arg_names = self.__signature__.parameters.keys()
        arg_indices = {n: i for i, n in enumerate(arg_names)}
        ret = collections.OrderedDict()
        for wrt, dname in self._grad.items():
            # Calculate the gradient
            grad_fun = getattr(self.model, dname)
            grad_val = grad_fun(*args, **kwargs)
            
            # Get the shape of the wrt argument
            try:
                wrt_shape = np.shape(kwargs[wrt])
            except KeyError:
                wrt_shape = np.shape(args[arg_indices[wrt]])
            
            # Accumulate so the gradient has the same shape as the variable
            ret[wrt] = grad_val.reshape(-1, *wrt_shape).sum(0)
        return ret


class ObjectiveFunctionMeta(OptimizationFunctionMeta):

    bases = ObjectiveFunction,
    """Bases of generated class."""
    
    def __init__(self, name, method, desc, ModelClass):
        # Initialize base class
        super().__init__(name, method, desc, ModelClass)
        
        self.out_sz = 1
        """Function output base size."""
        
        self._grad = collections.OrderedDict(desc['grad'])
        """First derivatives."""
        
        # Assign descriptive signature to the gradient
        method_sig = inspect.signature(self.method)
        self.grad = with_signature(self.grad, method_sig)


class ConstraintFunction(OptimizationFunction):
    def jac_ind(self, dec_ext={}, out_ext=()):
        return self._sparse_deriv_ind(self._jac, dec_ext, out_ext)

    def jac_val(self, *args, **kwargs):
        return self._sparse_deriv_val(self._jac, *args, **kwargs)
    
    def hess_ind(self, dec_ext={}, out_ext=()):
        return self._sparse_deriv_ind(self._hess, dec_ext, out_ext)

    def hess_val(self, *args, **kwargs):
        return self._sparse_deriv_val(self._hess, *args, **kwargs)


class ConstraintFunctionMeta(OptimizationFunctionMeta):

    bases = ConstraintFunction,
    """Bases of generated class."""
    
    def __init__(self, name, method, desc, ModelClass):
        # Initialize base class
        super().__init__(name, method, desc, ModelClass)
        
        self.out_sz = shape_size(desc['shape'])
        """Constraint function output base size."""
        
        self._jac = collections.OrderedDict(desc['jac'])
        """First derivatives."""
        
        self._hess = collections.OrderedDict(desc['hess'])
        """Second derivatives."""
        
        # Assign descriptive signature to the sparse value functions
        method_sig = inspect.signature(self.method)
        self.jac_val = with_signature(self.jac_val, method_sig)
        self.hess_val = with_signature(self.hess_val, method_sig)


def bound_signature(method):
    """Return the signature of a method when bound."""
    sig = inspect.signature(method)
    param = list(sig.parameters.values())[1:]
    return inspect.Signature(param, return_annotation=sig.return_annotation)


def with_signature(f, sig):
    @functools.wraps(f)
    def new_f(*args, **kwargs):
        return f(*args, **kwargs)
    new_f.__signature__ = sig
    return new_f


def shape_size(shape):
    return np.prod(shape, dtype=int)


def ndim_range(shape):
    assert isinstance(shape, tuple)
    return np.arange(shape_size(shape)).reshape(shape)
