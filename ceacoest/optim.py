"""General sparse optimization problem modeling."""


import collections
import itertools
import numbers

import numpy as np


class Problem:
    """Sparse optimization problem."""
    
    def __init__(self):
        self.decision = {}
        """Decision variable component specifications."""

        self.ndec = 0
        """Size of the decision vector."""

        self.derived = collections.OrderedDict()
        """Specifications of variables derived from the decision variables."""

        self.merits = {}
        """Merit components."""

        self.merit_gradients = []
        """Merit gradient components."""

        self.constraints = {}
        """Problem constraints."""

        self.ncons = 0
        """Size of the constraint vector."""

        self.nnzjac = 0
        """Number of nonzero constraint Jacobian elements."""

        self.jacobian = []
        """Constraint Jacobian components."""

        self.nnzchess = 0
        """Number of nonzero constraint Hessian elements."""

        self.constraint_hessian = []
        """Constraint Hessian components."""

        self.nnzmhess = 0
        """Number of nonzero merit Hessian elements."""

        self.merit_hessian = []
        """Merit Hessian components."""

    def register_decision(self, name, shape, tiling=None):
        component = Decision(self.ndec, shape, tiling)
        self.decision[name] = component
        self.ndec += component.size
        return component
    
    def register_derived(self, name, spec):
        self.derived[name] = spec
    
    def unpack_decision(self, dvec):
        """Unpack the vector of decision variables into its components."""
        dvec = np.asarray(dvec)
        assert dvec.shape == (self.ndec,)

        components = {}
        for name, spec in self.decision.items():
            components[name] = spec.unpack_from(dvec)
        return components
    
    def set_decision(self, name, value, out=None):
        out = np.zeros(self.ndec) if out is None else out
        self.decision[name].assign_to(out, value)
        return out
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        variables = self.unpack_decision(dvec)
        for name, spec in self.derived.items():
            variables[name] = spec.build(variables)
        return variables

    def wrt_spec(self, name):
        return self.decision.get(name, None) or self.derived[name]
    
    def register_merit(self, name, fun, arg_names, tiling=None):
        self.merits[name] = Merit(fun, arg_names, tiling)
        
    def merit(self, dvec):
        var = self.variables(dvec)
        return sum(m(var) for m in self.merits.values())
    
    def register_merit_gradient(self, merit_name, wrt_name, fun):
        merit = self.merits[merit_name]
        wrt = self.decision.get(wrt_name, None) or self.derived[wrt_name]
        grad = MeritGradient(fun, wrt, merit)
        self.merit_gradients.append(grad)
        
    def merit_gradient(self, dvec, out=None):
        var = self.variables(dvec)
        out = np.zeros(self.ndec) if out is None else out
        for grad in self.merit_gradients:
            grad(var, add_to=out)
        return out
    
    def register_merit_hessian(self, merit_name, wrt_names, val, ind):
        wrt = (self.wrt_spec(wrt_names[0]), self.wrt_spec(wrt_names[1]))
        merit = self.merits[merit_name]
        hess = MeritHessian(val, ind, self.nnzmhess, wrt, merit)
        self.nnzmhess += hess.size
        self.merit_hessian.append(hess)
    
    def merit_hessian_val(self, dvec, out=None):
        var = self.variables(dvec)
        if out is None:
            out = np.zeros(self.nnzmhess)
        for hess in self.merit_hessian:
            hess(var, assign_to=out)
        return out
    
    def merit_hessian_ind(self, out=None):
        if out is None:
            out = np.zeros((2, self.nnzmhess), dtype=int)
        for hess in self.merit_hessian:
            hess.ind(assign_to=out)
        return out
    
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
            cons(var, assign_to=out)
        return out
    
    def set_constraint(self, name, val, out=None):
        out = np.zeros(self.ncons) if out is None else out
        cons = self.constraints[name]
        cons.assign_to(out, value)
        return out
    
    def register_constraint_jacobian(self, constraint_name, wrt_name, val, ind):
        cons = self.constraints[constraint_name]
        wrt = self.decision.get(wrt_name, None) or self.derived[wrt_name]
        jac = ConstraintJacobian(val, ind, self.nnzjac, wrt, cons)
        self.nnzjac += jac.size
        self.jacobian.append(jac)

    def constraint_jacobian_val(self, dvec, out=None):
        var = self.variables(dvec)
        if out is None:
            out = np.zeros(self.nnzjac)
        for jac in self.jacobian:
            jac(var, assign_to=out)
        return out
    
    def constraint_jacobian_ind(self, out=None):
        if out is None:
            out = np.zeros((2, self.nnzjac), dtype=int)
        for jac in self.jacobian:
            jac.ind(assign_to=out)
        return out

    def register_constraint_hessian(self, constraint_name, wrt_names, val, ind):
        wrt = (self.wrt_spec(wrt_names[0]), self.wrt_spec(wrt_names[1]))
        cons = self.constraints[constraint_name]
        hess = ConstraintHessian(val, ind, self.nnzchess, wrt, cons)
        self.nnzchess += hess.size
        self.constraint_hessian.append(hess)
    
    def constraint_hessian_val(self, dvec, multipliers=None, out=None):
        var = self.variables(dvec)
        if out is None:
            out = np.zeros(self.nnzchess)
        for hess in self.constraint_hessian:
            hess(var, multipliers, assign_to=out)
        return out
    
    def constraint_hessian_ind(self, out=None):
        if out is None:
            out = np.zeros((2, self.nnzchess), dtype=int)
        for hess in self.constraint_hessian:
            hess.ind(assign_to=out)
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
    
    def add_to(self, destination, value):
        assert destination.ndim == 1
        assert destination.size >= self.offset + self.size
        destination[self.slice] += np.broadcast_to(value, self.shape).flat
    
    def assign_to(self, destination, value):
        assert destination.ndim == 1
        assert destination.size >= self.offset + self.size
        destination[self.slice] = np.broadcast_to(value, self.shape).flat


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
    
    def __call__(self, arg_dict, assign_to=None):
        args = tuple(arg_dict[n] for n in self.argument_names)
        out = self.fun(*args)
        assert out.shape == self.shape
        
        if assign_to is not None:
            self.assign_to(assign_to, out)
        
        return out


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
    def __init__(self, val, ind, offset, wrt, constraint):
        assert np.ndim(ind) == 2 and np.size(ind, 0) == 2
        self.template_ind = np.asarray(ind, dtype=int)
        """Nonzero Jacobian element indices template."""
        
        self.constraint = constraint
        """Specification of the parent constraint."""
        
        self.wrt = wrt
        """Decision or derived variable specification."""
        
        nnz = np.size(ind, 1)
        tiling = constraint.tiling
        shape = (nnz,) if tiling is None else (tiling, nnz)
        super().__init__(shape, offset, val, constraint.argument_names)
    
    def ind(self, assign_to=None):
        ind = self.template_ind
        ret = np.zeros((2,) + self.shape, dtype=int)
        ret[1] = self.constraint.expand_indices(ind[1])
        ret[0] = self.wrt.expand_indices(ind[0])
        
        if assign_to is not None:
            self.assign_to(assign_to[0], ret[0])
            self.assign_to(assign_to[1], ret[1])
        
        return ret


class ConstraintHessian(CallableComponent):
    def __init__(self, val, ind, offset, wrt, constraint):
        assert np.ndim(ind) == 2 and np.size(ind, 0) == 3
        self.template_ind = np.asarray(ind, dtype=int)
        """Nonzero Hessian element indices template."""
        
        self.constraint = constraint
        """Specification of the parent constraint."""
        
        assert len(wrt) == 2
        self.wrt = wrt
        """Decision or derived variable specification."""
        
        nnz = np.size(ind, 1)
        tiling = constraint.tiling
        shape = (nnz,) if tiling is None else (tiling, nnz)
        super().__init__(shape, offset, val, constraint.argument_names)
    
    def ind(self, assign_to=None):
        ind = self.template_ind
        ret = np.zeros((2,) + self.shape, dtype=int)
        ret[1] = self.wrt[0].expand_indices(ind[1])
        ret[0] = self.wrt[1].expand_indices(ind[0])
        
        if assign_to is not None:
            self.assign_to(assign_to[0], ret[0])
            self.assign_to(assign_to[1], ret[1])
        
        return ret

    def __call__(self, arg_dict, multipliers=None, assign_to=None):
        out = super().__call__(arg_dict)
        if multipliers is not None:
            mind = self.template_ind[2]
            out = out * self.constraint.unpack_from(multipliers)[..., mind]
        if assign_to is not None:
            self.assign_to(assign_to, out)
        return out
        

class Merit:
    def __init__(self, fun, argument_names, tiling=None):
        self.fun = fun
        """Merit function."""
        
        self.argument_names = argument_names
        """Merit function argument names."""
        
        assert isinstance(tiling, (numbers.Integral, type(None)))
        self.tiling = tiling
        """Number of repetitions of the template shape."""
    
    @property
    def shape(self):
        return () if self.tiling is None else (self.tiling,)
    
    def __call__(self, arg_dict):
        args = tuple(arg_dict[n] for n in self.argument_names)
        out = self.fun(*args)
        assert out.shape == self.shape
        return out if self.tiling is None else out.sum(0)


class MeritGradient:
    def __init__(self, fun, wrt, merit):
        self.fun = fun
        """Gradient function."""
        
        self.merit = merit
        """Underlying merit function."""
        
        self.wrt = wrt
        """Decision or derived variable specification."""
    
    @property
    def shape(self):
        return self.merit.shape + self.wrt.shape
    
    def __call__(self, arg_dict, add_to=None):
        args = tuple(arg_dict[n] for n in self.merit.argument_names)
        grad = self.fun(*args)
        assert grad.shape == self.shape
        if self.merit.tiling is not None:
            grad = grad.sum(0)
        if add_to is not None:
            self.wrt.add_to(add_to, grad)
        return grad


class MeritHessian(Component):
    def __init__(self, val, ind, offset, wrt, merit):
        self.val = val
        """Hessian nonzero elements function."""
        
        assert np.ndim(ind) == 2 and np.size(ind, 0) == 2
        self.template_ind = np.asarray(ind, dtype=int)
        """Nonzero Hessian element indices template."""
        
        assert len(wrt) == 2
        self.wrt = wrt
        """Decision or derived variable specifications."""

        self.merit = merit
        """Underlying merit function."""
        
        nnz = np.size(ind, 1)
        shape = (nnz,) if merit.tiling is None else (merit.tiling, nnz)
        super().__init__(shape, offset)
    
    def __call__(self, arg_dict, assign_to=None):
        args = tuple(arg_dict[n] for n in self.merit.argument_names)
        out = self.val(*args)
        assert out.shape == self.shape
        if assign_to is not None:
            self.assign_to(assign_to, out)
        return out

    def ind(self, assign_to=None):
        ind = self.template_ind
        ret = np.zeros((2,) + self.shape, dtype=int)
        ret[1] = self.wrt[0].expand_indices(ind[1])
        ret[0] = self.wrt[1].expand_indices(ind[0])
        
        if assign_to is not None:
            self.assign_to(assign_to[0], ret[0])
            self.assign_to(assign_to[1], ret[1])
        
        return ret
