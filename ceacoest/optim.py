"""General sparse optimization problem modelling."""


import collections
import contextlib
import itertools
import numbers

import numpy as np

from . import utils


class Problem:
    """Optimization problem with sparse derivatives."""
    
    def __init__(self):
        self.decision = collections.OrderedDict()
        """Decision variable component specifications."""
        
        self.remapped = {}
        """Specification of remapped variables."""
        
        self.objectives = []
        """Objective function specifications."""
        
        self.constraints = []
        """Constraint function specifications."""
        
        self.ndec = 0
        """Size of the decision vector."""
        
        self.ncons = 0
        """Size of the constraint vector."""
    
    def add_decision(self, name, shape):
        """Add a decision variable to this problem."""
        if isinstance(shape, numbers.Integral):
            shape = shape,
        dec = Decision(shape, self.ndec)
        self.ndec += dec.size
        self.decision[name] = dec
        return dec
    
    def add_objective(self, fun, shape, args=None):
        """Add an objective function to this problem."""
        if isinstance(shape, numbers.Integral):
            shape = shape,
        self.objectives.append(Objective(shape, fun, args))
    
    def add_constraint(self, fun, shape, args=None):
        """Add a constraint function to this problem."""
        if isinstance(shape, numbers.Integral):
            shape = shape,
        cons = Constraint(shape, self.ncons, fun, args)
        self.ncons += cons.size
        self.constraints.append(cons)
    
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        dvec = np.asarray(dvec)
        assert dvec.shape == (self.ndec,)
        items = itertools.chain(self.decision.items(), self.remapped.items())
        return {k: v.unpack_from(dvec) for k,v in items}
    
    @property
    def variable_shapes(self):
        items = itertools.chain(self.decision.items(), self.remapped.items())
        return {k: v.shape for k,v in items}
    
    def variable_spec(self, varname):        
        spec = self.decision.get(varname) or self.remapped.get(varname)
        if spec is None:
            raise ValueError(f"no such variable {varname}")
        return spec

    def _sparse_fun_val(self, dvec, nnz, components, val_fun_name):
        variables = self.variables(dvec)
        val = np.empty(nnz)
        offset = 0
        for comp in components:
            val_fun = getattr(comp, val_fun_name)
            for varnames, comp_val in val_fun(variables).items():
                comp_nnz = comp_val.size
                assert offset + comp_nnz <= nnz                
                s = slice(offset, offset + comp_nnz)
                val[s] = comp_val.ravel()
                offset += comp_nnz
        assert offset == nnz
        return val
    
    def obj(self, dvec):
        """Optimization problem objective function."""
        variables = self.variables(dvec)
        obj_val = 0.0
        for obj in self.objectives:
            obj_val += np.sum(obj(variables))
        return obj_val
    
    def obj_grad(self, dvec):
        """Objective function gradient."""
        variables = self.variables(dvec)
        grad = np.zeros(self.ndec)
        for obj in self.objectives:
            for wrt, val in obj.grad(variables).items():
                wrt_var = self.variable_spec(wrt)
                if wrt_var is None:
                    raise RuntimeError(f"unrecognized variable '{wrt}'")
                wrt_var.add_to(grad, val)
        return grad

    @property
    def obj_hess_nnz(self):
        return sum(obj.hess_nnz for obj in self.objectives)
    
    @property
    def obj_hess_ind(self):
        shapes = self.variable_shapes
        nnz = self.obj_hess_nnz
        ind = np.empty((2, nnz), int)
        offset = 0
        for obj in self.objectives:
            for (var0name, var1name), loc_ind in obj.hess_ind(shapes).items():
                var0 = self.variable_spec(var0name)
                var1 = self.variable_spec(var1name)
                loc_nnz = loc_ind[0].size
                assert offset + loc_nnz <= nnz
                
                s = slice(offset, offset + loc_nnz)
                ind[0, s] = var0.convert_ind(loc_ind[0].ravel())
                ind[1, s] = var1.convert_ind(loc_ind[1].ravel())
                offset += loc_nnz
        assert offset == nnz
        return ind
    
    def obj_hess_val(self, dvec):
        nnz = self.obj_hess_nnz
        components = self.objectives
        return self._sparse_fun_val(dvec, nnz, components, 'hess_val')
    
    def constr(self, dvec):
        cvec = np.zeros(self.ncons)
        variables = self.variables(dvec)
        for constr in self.constraints:
            constr.pack_into(cvec, constr(variables))
        return cvec
    
    @property
    def constr_jac_nnz(self):
        return sum(c.jac_nnz for c in self.constraints)
    
    @property
    def constr_jac_ind(self):
        shapes = self.variable_shapes
        nnz = self.constr_jac_nnz
        ind = np.empty((2, nnz), int)
        offset = 0
        for c in self.constraints:
            for (varname,), loc_ind in c.jac_ind(shapes).items():
                var = self.variable_spec(varname)
                loc_nnz = loc_ind[0].size
                assert offset + loc_nnz <= nnz

                s = slice(offset, offset + loc_nnz)
                ind[0, s] = var.convert_ind(loc_ind[0].ravel())
                ind[1, s] = c.convert_ind(loc_ind[1].ravel())
                offset += loc_nnz
        assert offset == nnz
        return ind
    
    def constr_jac_val(self, dvec):
        nnz = self.constr_jac_nnz
        components = self.constraints
        return self._sparse_fun_val(dvec, nnz, components, 'jac_val')
    
    @property
    def constr_hess_nnz(self):
        return sum(c.hess_nnz for c in self.constraints)
    
    @property
    def constr_hess_ind(self):
        shapes = self.variable_shapes
        nnz = self.constr_hess_nnz
        ind = np.empty((3, nnz), int)
        offset = 0
        for c in self.constraints:
            for (var0name, var1name), loc_ind in c.hess_ind(shapes).items():
                var0 = self.variable_spec(var0name)
                var1 = self.variable_spec(var1name)
                loc_nnz = loc_ind[0].size
                assert offset + loc_nnz <= nnz
                
                s = slice(offset, offset + loc_nnz)
                ind[0, s] = var0.convert_ind(loc_ind[0].ravel())
                ind[1, s] = var1.convert_ind(loc_ind[1].ravel())
                ind[2, s] = c.convert_ind(loc_ind[2].ravel())
                offset += loc_nnz
        assert offset == nnz
        return ind

    def constr_hess_val(self, dvec):
        nnz = self.constr_hess_nnz
        components = self.constraints
        return self._sparse_fun_val(dvec, nnz, components, 'hess_val')
    
    @property
    def lag_hess_nnz(self):
        return self.obj_hess_nnz + self.constr_hess_nnz
    
    @property
    def lag_hess_ind(self):
        obj_nnz = self.obj_hess_nnz
        nnz = self.lag_hess_nnz
        ind = np.zeros((2, nnz), int)
        ind[:, :obj_nnz] = self.obj_hess_ind
        ind[:, obj_nnz:] = self.constr_hess_ind[:2]
        return ind

    def lag_hess_val(self, dvec, obj_mult, constr_mult):
        assert np.shape(constr_mult) == (self.ncons,)
        nnz = self.lag_hess_nnz
        obj_nnz = self.obj_hess_nnz
        mult_ind = self.constr_hess_ind[2]
        
        val = np.zeros(nnz)
        val[:obj_nnz] = self.obj_hess_val(dvec) * obj_mult
        val[obj_nnz:] = self.constr_hess_val(dvec) * constr_mult[mult_ind]
        return val
    
    @contextlib.contextmanager
    def ipopt(self, d_bounds, constr_bounds):
        from mseipopt import ez
        f = self.obj
        g = self.constr
        grad = self.obj_grad
        jac_ind = lambda: self.constr_jac_ind[[1,0]]
        hess_ind = lambda: self.lag_hess_ind
        jac = jac_ind, self.constr_jac_val
        hess = hess_ind, self.lag_hess_val
        nele_jac = self.constr_jac_nnz
        nele_hess = self.lag_hess_nnz
        with ez.Problem(d_bounds, constr_bounds, f, g,
                        grad, jac, nele_jac, hess, nele_hess) as problem:
            yield problem


class Component:
    """Specificiation of a problem's decision or constraint vector component."""
    
    def __init__(self, shape, offset):
        self.shape = (shape,) if isinstance(shape, numbers.Integral) else shape
        """The component's ndarray shape."""
        
        self.offset = offset
        """Offset into the parent vector."""
    
    def __repr__(self):
        clsname = type(self).__name__
        offset = self.offset
        shape = self.shape
        return f'<{clsname} {offset=} {shape=}>'
    
    @property
    def size(self):
        """Total number of elements."""
        return np.prod(self.shape, dtype=np.intc)
    
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
        destination[self.slice] += np.broadcast_to(value, self.shape).ravel()
    
    def pack_into(self, destination, value):
        assert destination.ndim == 1
        assert destination.size >= self.offset + self.size
        destination[self.slice] = np.broadcast_to(value, self.shape).ravel()
    
    def convert_ind(self, comp_ind):
        """Convert component indices to parent vector indices."""
        comp_ind = np.asarray(comp_ind, dtype=int)
        return comp_ind + self.offset


class Decision(Component):
    """A decision variable within an optimization problem."""


class OptimizationFunction:
    def __init__(self, shape, fun, args=None):
        self.fun = fun
        """The underlying constraint object."""

        self.args = utils.sig_arg_names(fun) if args is None else args
        """The underlying function argument names."""
        
        self.shape = shape
        """Function output shape"""
    
    def __call__(self, variables):
        args = (variables[arg] for arg in self.args)
        return self.fun(*args)
    
    @property
    def hess_nnz(self):
        return self.fun.hess_nnz(self.shape)
    
    def hess_ind(self, var_shapes):
        return self.fun.hess_ind(var_shapes, self.shape)
    
    def hess_val(self, variables):
        args = (variables[arg] for arg in self.args)
        return self.fun.hess_val(*args)
    
    @property
    def name(self):
        try:
            return type(self.fun).__name__
        except AttributeError:
            pass


class Constraint(Component, OptimizationFunction):
    """A constraint within an optimization problem."""
    
    def __init__(self, shape, offset, fun, args=None):
        # Initizalize base classes
        Component.__init__(self, shape, offset)
        OptimizationFunction.__init__(self, shape, fun, args)

    @property
    def jac_nnz(self):
        return self.fun.jac_nnz(self.shape)
    
    def jac_ind(self, var_shapes):
        return self.fun.jac_ind(var_shapes, self.shape)

    def jac_val(self, variables):
        args = (variables[arg] for arg in self.args)
        return self.fun.jac_val(*args)


class Objective(OptimizationFunction):
    """An objective within an optimization problem."""
    
    def grad(self, variables):
        args = (variables[arg] for arg in self.args)
        return self.fun.grad(*args)
