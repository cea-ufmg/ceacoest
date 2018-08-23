"""General sparse optimization problem modeling."""


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
    
    def register_decision(self, name, shape):
        component = Component(shape, self.ndec)
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
    
    def register_constraint(self, name, shape, f, argument_names):
        cons = Constraint(shape, self.ncons, f, argument_names)
        self.constraints[name] = cons
        self.ncons += cons.size
        return cons
    
    def constraint(self, dvec):
        var = self.variables(dvec)
        cvec = np.zeros(self.ncons)
        for name, cons in self.constraints.items():
            cons(var, pack_into=cvec)
        return cvec

    def register_constraint_jacobian(self, constraint_name, wrt, val, ind):
        cons = self.constraints[constraint_name]
        wrt_offsets = self.get_index_offsets(wrt)
        jac = ConstraintJacobian(val, ind, self.nnzjac, wrt_offsets, cons)
        self.jacobian.append(jac)


class Component:
    """Specificiation of a problem's decision or constraint vector component."""
    
    def __init__(self, shape, offset):
        self.shape = (shape,) if isinstance(shape, int) else shape
        """The component's ndarray shape."""
        
        self.offset = offset
        """Offset into the parent vector."""
        
        self.size = np.prod(shape, dtype=int)
        """Total number of elements."""
        
        self.slice = slice(offset, offset + self.size)
        """This component's slice in the parent vector."""
    
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


class Constraint(CallableComponent):

    def __init__(self, shape, offset, fun, argument_names):
        super().__init__(shape, offset, fun, argument_names)
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
    def __init__(self, val, ind, offset, wrt_offsets, constraint):
        assert np.ndim(ind) == 2
        assert np.size(ind, 0) == 2
        self.ind = np.asarray(ind, dtype=int)
        """Nonzero Jacobian element indices."""
        
        self.constraint = constraint
        """Specification of the parent constraint."""
        
        nnz = np.size(ind, 1)
        broadcast = len(constraint.shape) == 2
        shape = (constraint.shape[0], nnz) if broadcast else (nnz,)
        super().__init__(shape, offset, val, constraint.argument_names)
    
    def ind(self, pack_into=None):
        ret = np.zeros((2,) + self.shape, dtype=int)
        ret[1] = self.ind[1] + self.constraint.index_offsets
        if callable(self.wrt_offsets):
            ret[0] = self.ind[0] + self.wrt_offsets(ind[0])
        else:
            ret[0] = self.ind[0] + self.wrt_offsets
        
        if pack_into is not None:
            self.pack_into(pack_into[0], ret[0])
            self.pack_into(pack_into[1], ret[1])
        
        return ret
