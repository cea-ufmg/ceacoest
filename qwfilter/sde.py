'''Stochastic differential equation module.'''


import abc
import inspect

import numpy as np
import sym2num


model_append_template = '''\
    nx = {nx}
    """Length of the state vector."""

    ny = {ny}
    """Length of the output vector."""

    nw = {nw}
    """Length of the process noise vector."""

    nq = {nq}
    """Length of the unknown parameter vector."""'''

class SymbolicModel(sym2num.SymbolicModel):

    @property
    @abc.abstractmethod
    def t(self):
        '''The time variable.'''
        raise NotImplementedError("Pure abstract method.")

    @property
    @abc.abstractmethod
    def x(self):
        '''The state vector.'''
        raise NotImplementedError("Pure abstract method.")
    
    @abc.abstractmethod
    def f(self, t, x, *args):
        '''SDE drift function.'''
        raise NotImplementedError("Pure abstract method.")

    @abc.abstractmethod
    def g(self, t, x, *args):
        '''SDE diffusion.'''
        raise NotImplementedError("Pure abstract method.")
    
    def _init_functions(self):
        '''Initialize model functions.'''
        super()._init_functions()
        
        # Add default implementation for Q = g * g.T
        if 'Q' not in self.functions:
            g = self.functions['g']
            Qexpr = np.dot(g.out, g.out.T)
            self.functions['Q'] = sym2num.SymbolicFunction(Qexpr, g.args, 'Q')

    def print_class(self, printer, name=None, signature=''):
        base_code = super().print_class(printer, name, signature)
        nx = self.vars['x'].size
        ny = self.vars['y'].size
        nw = self.functions['g'].out.shape[1]
        nq = self.vars['q'].size if 'q' in self.vars else 0
        suffix = model_append_template.format(nx=nx, ny=ny, nw=nw, nq=nq)
        return '\n'.join([base_code, suffix])

class SymbolicDiscretizedModel(SymbolicModel):
    
    t_next = 't_next'
    '''Next time value in a discrete-time transition.'''
        
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
        if arg_items[0][0] != 't' or arg_items[1][0] != 'x':
            msg = "First two arguments in discretized functions must `(t, x)`."
            raise RuntimeError(msg)
        
        # Switch the time argument with the transition times
        td = np.hstack([self.vars['t'], self.vars['t_next']])
        return [('td', td)] + arg_items[1:]


class EulerDiscretizedModel(SymbolicDiscretizedModel):
    def _discretize(self):
        '''Discretize the drift and diffusion functions.'''
        # Get the discretization variables
        t = self.vars['t']
        t_next = self.vars['t_next']
        dt = t_next - t
        
        # Discretize the drift
        f = self.functions['f']
        fd = self.vars['x'] + f.out * dt
        fdargs = self.discretized_args(f.args)
        self.functions['fd'] = sym2num.SymbolicFunction(fd, fdargs, 'fd')
        
        # Discretize the diffusion
        g = self.functions['g']
        gd = g.out * dt ** 0.5
        gdargs = self.discretized_args(g.args)
        self.functions['gd'] = sym2num.SymbolicFunction(gd, gdargs, 'gd')

