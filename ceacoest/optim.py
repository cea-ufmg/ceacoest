"""General sparse optimization problem modeling."""


import collections

import attrdict
import numpy as np


class Problem:
    """Sparse optimization problem."""
    
    def __init__(self):
        self.decision_components = collections.OrderedDict()
        """Decision variable component specifications."""
        
        self.nd = 0
        """Total number of problem decision variable elements."""
    
    def register_decision_variable(self, name, shape):
        component = DecisionComponent(shape, self.nd)
        self.decision_components[name] = component
        self.nd += component.size

    def unpack_decision(self, dvec):
        """Unpack the vector of decision variables into its components."""
        dvec = np.asarray(dvec)
        assert d.shape == (self.nd,)

        components = {}
        for name, spec in self.decision_components.items():
            components[name] = spec.extract_from(dvec)
        return components
    
    def pack_decision(self, **components):
        """Pack the decision variable components into the vector."""
        dvec = np.zeros(self.nd)
        for name, spec in self.decision_components.items():
            spec.pack_into(dvec, components[name])
        return dvec


class DecisionComponent:
    """Specificiation of a problem's decision variable component."""
    
    def __init__(self, shape, offset):
        self.shape = shape
        """The component's ndarray shape."""
        
        self.offset = offset
        """Offset into the decision vector."""
        
        self.size = np.prod(shape, dtype=int)
        """Total number of elements."""
        
        self.slice = slice(offset, offset + self.size)
        """This component's slice in the decision variables vector."""
    
    def unpack_from(self, dvec):
        """Extract component from decicion variable vector."""
        return np.asarray(dvec)[self.slice].reshape(self.shape)

    def pack_into(self, dvec, value):
        """Pack component into decicion variable vector."""
        assert np.shape(value) == self.shape
        dvec[self.slice] = np.ravel(value)


Decision = collections.namedtuple('Decision', ['shape', 'offset', 'size'])
