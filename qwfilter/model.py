'''Utilities for model building.'''


import inspect

import numpy as np


class ParametrizedModel:
    def __init__(self, params):
        # Save a copy of the given params
        self._params = params.copy()
        
        # Add default parameters for the empty variables
        for name, spec in self.var_spec:
            if np.size(spec) == 0 and name not in params:
                self._params[name] = np.array([])
    
    def parametrize(self, params):
        new_params = self._params.copy()
        new_params.update(params)
        return type(self)(params)
    
    def parametrized_call(self, f, *args, **kwargs):
        spec = inspect.getfullargspec(f)
        call_args = self._params.copy()
        call_args.update((k, v) in kwargs if v is not None)
        call_args.update(zip(spec.args, args))
        return f(**call_args)

