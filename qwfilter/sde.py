'''Stochastic differential equation module.'''


import abc

import numpy as np
import sym2num


class DiscretizedModel(sym2num.ParametrizedModel):
    pass


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


