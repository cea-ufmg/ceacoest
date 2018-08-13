"""General sparse optimization problem modeling."""


import collections

import attrdict
import numpy as np

#### 
#### Perhaps change the name of dvar to components and rethink terminology.
#### The vector d has many X. Also intead of problem.decision it could be
#### and the class Decision we could have something more akin to Spec, X-Spec.
####

class Problem:
    """Sparse optimization problem."""
    
    def __init__(self):
        self.decision = collections.OrderedDict()
        """Problem decision variable specifications."""

        self.nd = 0
        """Total number of problem decision variables."""
    
    def register_decision_variable(self, name, shape):
        size = np.prod(shape, dtype=int)
        self.decision[name] = Decision(shape, self.nd)
        self.nd += size

    def unpack_decision(self, dvec):
        """Unpack the vector of decision variables into its components."""
        dvec = np.asarray(dvec)
        assert d.shape == (self.nd,)

        dvar = {}
        for name, spec in self.decision.items():
            dvar[name] = dvec[spec.slice].reshape(spec.shape)
        return dvar
    
    def pack_decision(self, dvar):
        """Pack the decision variable components into the vector."""
        dvec = np.empty(self.nd)
        for name, spec in self.decision.items():
            assert np.shape(dvar[name]) == spec.shape
            dvec[spec.slice] = np.ravel(dvar[name])
        return dvec


class Decision:
    def __init__(self, shape, offset):
        self.shape = shape
        self.offset = offset
        self.size = size = np.prod(shape, dtype=int)
        self.slice = slice(offset, offset + size)


Decision = collections.namedtuple('Decision', ['shape', 'offset', 'size'])
