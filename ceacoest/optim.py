"""General sparse optimization problem modeling."""


import numbers

import numpy as np


class Problem:
    """Sparse optimization problem."""
    
    def __init__(self):
        self.decision = {}
        """Decision variable component specifications."""

        self.ndec = 0
        """Size of the decision vector."""

        self.constraints = {}
        """Problem constraints."""

        self.ncons = 0
        """Size of the constraint vector."""

        self.known_index_offsets = {}
        """Problem variable index offsets for derivatives."""
        
        self.nnzjac = 0
        """Number of nonzero constraint Jacobian elements."""

        self.jacobian = []
        """Constraint Jacobian components."""

        self.nnzchess = 0
        """Number of nonzero constraint Hessian elements."""

        self.constraint_hessian = []
        """Constraint Jacobian components."""
    
    def register_decision(self, name, shape, tiling=None):
        component = Decision(self.ndec, shape, tiling)
        self.decision[name] = component
        self.ndec += component.size
        return component
    
    def unpack_decision(self, dvec):
        """Unpack the vector of decision variables into its components."""
        dvec = np.asarray(dvec)
        assert dvec.shape == (self.ndec,)

        components = {}
        for name, spec in self.decision.items():
            components[name] = spec.unpack_from(dvec)
        return components
    
    def pack_decision(self, **components):
        """Pack the decision variable components into the vector."""
        dvec = np.zeros(self.ndec)
        for name, value in components:
            self.decision[name].pack_into(dvec, value)
        return dvec
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return self.unpack_decision(dvec)
    
    def set_index_offsets(self, var_name, offsets):
        if isinstance(offsets, np.ndarray) and offsets.ndim == 1:
            offsets = offsets[:, None]
        self.known_index_offsets[var_name] = offsets
    
    def get_index_offsets(self, var_name):
        try:
            return self.known_index_offsets[var_name]
        except KeyError:
            return self.decision[var_name].offset
    
    def register_constraint(self, name, f, argument_names, shape, tiling=None):
        cons = Constraint(self.ncons, f, argument_names, shape, tiling)
        self.constraints[name] = cons
        self.ncons += cons.size
        return cons
    
    def constraint(self, dvec, out=None):
        var = self.variables(dvec)
        if out is None:
            out = np.zeros(self.ncons)
        for name, cons in self.constraints.items():
            cons(var, pack_into=out)
        return out
    
    def register_constraint_jacobian(self, constraint_name, wrt, val, ind):
        cons = self.constraints[constraint_name]
        ##### fix #####
        raise NotImplementedError
        wrt_expansion = self.get_index_offsets(wrt)
        jac = ConstraintJacobian(val, ind, self.nnzjac, wrt_offsets, cons)
        self.nnzjac += jac.size
        self.jacobian.append(jac)

    def constraint_jacobian_val(self, dvec, out=None):
        var = self.variables(dvec)
        if out is None:
            out = np.zeros(self.nnzjac)
        for jac in self.jacobian:
            jac(var, pack_into=out)
        return out
    
    def constraint_jacobian_ind(self, out=None):
        if out is None:
            out = np.zeros((2, self.nnzjac), dtype=int)
        for jac in self.jacobian:
            jac.ind(pack_into=out)
        return out


class Component:
    """Specificiation of a problem's decision or constraint vector component."""
    
    def __init__(self, shape, offset):
        self.shape = (shape,) if isinstance(shape, numbers.Integral) else shape
        """The component's ndarray shape."""
        
        if shape == 4 and self.shape == (0,):
            import ipdb; ipdb.set_trace()

        self.offset = offset
        """Offset into the parent vector."""
    
    @property
    def size(self):
        """Total number of elements."""
        return np.prod(self.shape, dtype=int)
    
    @property
    def slice(self):
        """This component's slice in the parent vector."""
        return slice(self.offset, self.offset + self.size)
    
    def unpack_from(self, vec):
        """Extract component from parent vector."""
        return np.asarray(vec)[self.slice].reshape(self.shape)
    
    def pack_into(self, vec, value):
        """Pack component into parent vector."""
        value = np.asarray(value)
        if value.shape != self.shape:
            try:
                value = np.broadcast_to(value, self.shape)
            except ValueError:
                msg = "value with shape {} could not be broadcast to {}"
                raise ValueError(msg.format(value.shape, self.shape))
        vec[self.slice] = value.flatten()


class IndexedComponent(Component):
    def __init__(self, shape, offset):
        super().__init__(shape, offset)
        
        self.tiling = None
        """Number of repetitions of the template shape."""
    
    def set_tiling(self, tiling):
        """Sets this component's tiling and shape, must be called only once."""
        if self.tiling is not None:
            raise RuntimeError("tiling can only be set once")        
        if not isinstance(tiling, (numbers.Integral, type(None))):
            raise TypeError("tiling must be integer or None")
        if tiling is not None:
            self.tiling = tiling
            self.shape = (tiling,) + self.shape
    
    def expand_indices(self, ind):
        if self.tiling is None:
            return np.asarray(ind, dtype=int) + self.offset
        else:
            increment = np.prod(self.shape[1:], dtype=int)
            return np.arange(self.tiling)[:, None]*increment + ind + self.offset


class Decision(IndexedComponent):
    def __init__(self, offset, shape, tiling=None):
        super().__init__(shape, offset)
        self.set_tiling(tiling)


class CallableComponent(Component):
    def __init__(self, shape, offset, fun, argument_names):
        super().__init__(shape, offset)
        
        self.fun = fun
        """Underlying function."""
        
        self.argument_names = argument_names
        """Underlying function argument names."""
    
    def __call__(self, arg_dict, pack_into=None):
        args = tuple(arg_dict[n] for n in self.argument_names)
        ret = self.fun(*args)
        assert ret.shape == self.shape
        
        if pack_into is not None:
            self.pack_into(pack_into, ret)
        
        return ret


class Constraint(CallableComponent, IndexedComponent):

    def __init__(self, offset, fun, argument_names, shape, tiling=None):
        super().__init__(shape, offset, fun, argument_names)
        self.set_tiling(tiling)
        assert 1 <= len(self.shape) <= 2
    
    @property
    def index_offsets(self):
        """Constraint index offsets for derivatives"""
        if len(self.shape) == 1:
            return self.offset
        else:
            n, m = self.shape
            return m * np.arange(n)[:, None] + self.offset


class ConstraintJacobian(CallableComponent):
    def __init__(self, val, ind, offset, wrt_expansion, constraint):
        assert np.ndim(ind) == 2
        assert np.size(ind, 0) == 2
        self.template_ind = np.asarray(ind, dtype=int)
        """Nonzero Jacobian element indices template."""
        
        self.constraint = constraint
        """Specification of the parent constraint."""
        
        self.wrt_expansion = wrt_expansion
        """Jacobian row index expansion function."""
        
        nnz = np.size(ind, 1)
        broadcast = len(constraint.shape) == 2
        shape = (constraint.shape[0], nnz) if broadcast else (nnz,)
        super().__init__(shape, offset, val, constraint.argument_names)
    
    def ind(self, pack_into=None):
        ind = self.template_ind
        ret = np.zeros((2,) + self.shape, dtype=int)
        ret[1] = self.constraint.expand_indices(ind[1])
        ret[0] = self.wrt_expansion(ind[0])
        
        if pack_into is not None:
            self.pack_into(pack_into[0], ret[0])
            self.pack_into(pack_into[1], ret[1])
        
        return ret
