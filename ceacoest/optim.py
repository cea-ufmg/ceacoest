"""General sparse optimization problem modeling."""


import numpy as np


class Problem:
    """Sparse optimization problem."""
    
    def __init__(self):
        self.decision_components = {}
        """Decision variable component specifications."""

        self.ndec = 0
        """Size of the decision vector."""

        self.constraints = {}
        """Problem constraints."""

        self.ncons = 0
        """Size of the constraint vector."""

    def register_decision(self, name, shape):
        component = Component(shape, self.ndec)
        self.decision_components[name] = component
        self.ndec += component.size
    
    def register_derived(self, name, component):
        """Register a problem variable derived from the decision variables."""
        self.decision_components[name] = component
    
    def unpack_decision(self, dvec):
        """Unpack the vector of decision variables into its components."""
        dvec = np.asarray(dvec)
        assert d.shape == (self.ndec,)

        components = {}
        for name, spec in self.decision_components.items():
            components[name] = spec.extract_from(dvec)
        return components
    
    def pack_decision(self, **components):
        """Pack the decision variable components into the vector."""
        dvec = np.zeros(self.ndec)
        for name, value in components:
            self.decision_components[name].pack_into(dvec, value)
        return dvec
    
    def register_constraint(self, name, shape, f, argument_names):
        cons = Constraint(shape, self.ncons, f, argument_names)
        self.constraints[name] = cons
        self.ncons += cons.size
        return cons
    
    def constraint(self, dvec):
        dcomp = self.unpack_decision(dvec)
        cvec = np.zeros(self.ncons)
        for name, cons in self.constraints:
            cons(dcomp, pack_into=cvec)
        return cvec


class Component:
    """Specificiation of a problem's decision or constraint vector component."""
    
    def __init__(self, shape, offset=None, index=None):
        self.shape = shape
        """The component's ndarray shape."""
        
        self.offset = offset
        """Offset into the parent vector."""
        
        self.size = np.prod(shape, dtype=int)
        """Total number of elements."""
        
        if offset is not None and index is None:
            index = slice(offset, offset + self.size)
        if isinstance(index, np.ndarray):
            index = np.ravel(index)
        self.index = index
        """This component's index in the parent vector."""
    
    def unpack_from(self, vec):
        """Extract component from parent vector."""
        return np.asarray(vec)[self.index].reshape(self.shape)
    
    def pack_into(self, vec, value):
        """Pack component into parent vector."""
        value = np.asarray(value)
        if value.shape != self.shape:
            try:
                value = np.broadcast_to(value, self.shape)
            except ValueError:
                msg = "value with shape {} could not be broadcast to {}"
                raise ValueError(msg.format(value.shape, self.shape))
        vec[self.index] += value.flatten()


class Constraint(Component):
    def __init__(self, shape, offset, f, argument_names, bshape=()):
        super.__init__(shape, offset)
        
        self.f = f
        """Constraint function."""
        
        self.argument_names = argument_names
        """Constraint function argument names."""
        
        self.bshape = bshape
        """Broadcasted part of the output shape."""
        
        self.index_expansion = {}
        """Argument derivatives index expansion matrices."""
        
        self.nnzjac = 0
        """Number of nonzero jacobian matrix elements."""
        
        self.jac_valfuns = {}
        """Component specifications of the jacobian value functions."""
        
        self.nnzhess = 0
        """Number of nonzero hessian tensor elements."""
        
        self.hess_valfuns = {}
        """Component specifications of the hessian value functions."""
        
    def __call__(self, arg_dict, pack_into=None):
        args = tuple(arg_dict[n] for n in self.argument_names)
        ret = self.f(*args)
        assert ret.shape == self.shape
        
        if pack_into is not None:
            self.pack_into(pack_into, ret)
        
        return ret

    def set_index_expansion(self, name, expansion):
        self.index_expansion[name] = expansion
    
    def register_jacobian(self, valfun, ind, wrt):
        shape = self.bshape + (len(ind),)
        comp = self.jac_valfuns[valfun] = Component(shape, self.nnzjac)
        comp.ind = ind
        comp.deriv_wrt = wrt
        self.nnzjac += comp.size

    def register_jacobian(self, valfun, ind, wrt):
        shape = self.bshape + (len(ind),)
        comp = self.hess_valfuns[valfun] = Component(shape, self.nnzhess)
        comp.ind = ind
        comp.deriv_wrt = wrt
        self.nnzhess += comp.size
        
    def jac_val(self, arg_dict, out=None):
        if out is not None:
            assert out.shape == (self.nnzjac,)
        else:
            out = np.zeros(self.nnzjac)
        
        args = tuple(arg_dict[n] for n in self.argument_names)
        for valfun, comp in self.jac_valfuns.items():
            value = valfun(*args)
            assert np.shape(value) == comp.shape
            comp.pack_into(value, out)
        
        return out
    
    def hess_val(self, arg_dict, out=None):
        if out is not None:
            assert out.shape == (self.nnzhess,)
        else:
            out = np.zeros(self.nnzhess)
        
        args = tuple(arg_dict[n] for n in self.argument_names)
        for valfun, comp in self.hess_valfuns.items():
            value = valfun(*args)
            assert np.shape(value) == comp.shape
            comp.pack_into(value, out)
        
        return out

