'''Stochastic differential equation module.'''


import abc
import inspect

import numpy as np
import sym2num


class DiscretizedModel(sym2num.ParametrizedModel):
    def __init__(self, params={}, *, t=None, td=None):
        '''Discretized model constructor.
        
        Parameters
        ----------
        params : mapping or key/val sequence, optional
            The fixed variables in the model.
        t : array_like, optional
            The discretization time grid, with at least two elements. Must be
            given if `td` is None. If `td` is scalar and `t` is None then `t`
            is calculated as `t = k * td`.
        td : array_like, optional
            The discretization period or periods. If it is None than it is
            calculated as the finite differences of `t`.
        
        '''
        # Initialize superclass and check if all required arguments were given
        super().__init__(params)
        if t is None and td is None:
            raise TypeError("At least one of `t` or `td` arguments required.")
        
        # Process t and td arguments
        if t is None:
            if np.shape(td) != ():
                raise ValueError('`td` must be scalar if `t` is None.')
        elif td is None:
            t = np.asarray(t)
            if t.size < 2:
                msg = "At least 2 points required for discretization grid."
                raise ValueError(msg)
            td = np.diff(t.flatten(), axis=None)
            if np.all(td == td.flat[0]):
                td = td[0]
        
        # Save arguments
        self._t = t
        '''Discretization time grid.'''
        
        self._td = td
        '''Discretization period.'''

    def _discretized_time_arguments(self, call_args):
        try:
            k = call_args['k']
        except KeyError:
            raise TypeError("Discretized index argument `k` required.")
        
        try:
            td = call_args['td']
        except KeyError:
            td = self._td if np.shape(self._td) == () else self._td[k]
        
        try:
            t = call_args['t']
        except KeyError:
            t = td * k if self._t is None else self._t[k]
        
        return dict(t=t, td=td, k=k)
    
    def call_args(self, f, *args, **kwargs):
        call_args = super().call_args(f, *args, **kwargs)
        spec = inspect.getfullargspec(f)
        if 'k' in spec.args and 'td' in spec.args and 't' in spec.args:
            time_args = self._discretized_time_arguments(call_args)
            call_args = dict(call_args, **time_args)
        return call_args


class SymbolicModel(sym2num.SymbolicModel):
    
    def _init_functions(self):
        '''Initialize model functions.'''
        super()._init_functions()
        
        # Add default implementation for Q = g * g.T
        if 'Q' not in self.functions:
            g = self.functions['g']
            Qexpr = np.dot(g.out, g.out.T)
            self.functions['Q'] = sym2num.SymbolicFunction(Qexpr, g.args, 'Q')


class SymbolicDiscretizedModel(SymbolicModel):
    
    k = 'k'
    '''Discretized time index.'''
    
    td = 'td'
    '''Discretization period.'''
    
    var_names = ['k', 'td']
    '''Name of model variables.'''
    
    def _init_functions(self):
        '''Initialize the model functions.'''
        super()._init_functions() # Initialize base class functions
        self._discretize() # Generate the discretized drift and diffusion
        
        # Add default implementation for Qd = gd * gd.T
        if 'Qd' not in self.functions:
            gd = self.functions['gd']
            Qd = np.dot(gd.out, gd.out.T)
            self.functions['Qd'] = sym2num.SymbolicFunction(Qd, gd.args, 'Qd')
    
    @abc.abstractmethod
    def _discretize(self):
        '''Discretize the drift and diffusion functions.'''
        raise NotImplementedError("Pure abstract method.")

    def discretized_args(self, args):
        # Check if first argument is time
        arg_items = list(args.items())
        if arg_items[0][0] != 't':
            msg = "First argument in discretized functions must be time."
            raise RuntimeError(msg)
        
        # Get additional arguments of the discretized function
        k = self.vars['k']
        td = self.vars['td']
        t = self.vars['t']
        
        # Build and return the new argument list
        return [('k', k)] + arg_items[1:] + [('t', t), ('td', td)]


class EulerDiscretizedModel(SymbolicDiscretizedModel):
    def _discretize(self):
        '''Discretize the drift and diffusion functions.'''
        # Get the discretization variables
        td = self.vars['td']
        k = self.vars['k']
        
        # Discretize the drift
        f = self.functions['f']
        fd = f.out * td
        fdargs = self.discretized_args(f.args)
        self.functions['fd'] = sym2num.SymbolicFunction(fd, fdargs, 'fd')
        
        # Discretize the diffusion
        g = self.functions['g']
        gd = g.out * td ** 0.5
        gdargs = self.discretized_args(g.args)
        self.functions['gd'] = sym2num.SymbolicFunction(gd, gdargs, 'gd')


