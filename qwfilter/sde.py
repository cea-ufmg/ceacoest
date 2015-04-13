'''Stochastic differential equation module.'''


import abc
import inspect

import numpy as np
import scipy.linalg
import sympy
import sym2num


sde_model_append_template = '''\
    nx = {nx}
    """Length of the state vector."""

    ny = {ny}
    """Length of the output vector."""

    nw = {nw}
    """Length of the process noise vector."""

    nq = {nq}
    """Length of the unknown parameter vector."""'''


discretized_model_append_template = '''\
    nwd = {nwd}
    """Length of the state vector."""'''


class SymbolicModel(sym2num.SymbolicModel):

    var_names = {'t', 'x'}
    '''Name of the model variables.'''
    
    function_names = {'f', 'g'}
    '''Name of the model functions.'''
    
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

    def _init_variables(self):
        '''Initialize model variables.'''
        self.var_names = set.union(self.var_names, {'x', 't'})
        super()._init_variables()
    
    def _init_functions(self):
        '''Initialize model functions.'''
        self.function_names = set.union(self.function_names, {'f', 'g'})
        super()._init_functions()
        
        # Add default implementation for Q = g * g.T
        if 'Q' not in self.functions:
            g = self.functions['g']
            Qexpr = np.dot(g.out, g.out.T)
            self.functions['Q'] = sym2num.SymbolicFunction(Qexpr, g.args, 'Q')
    
    def print_class(self, printer, name=None, signature=''):
        base_code = super().print_class(printer, name, signature)
        nx = self.vars['x'].size
        nw = self.functions['g'].out.shape[1]
        nq = self.vars['q'].size if 'q' in self.vars else 0
        ny = self.vars['y'].size if 'y' in self.vars else 0
        suffix = sde_model_append_template.format(nx=nx, ny=ny, nw=nw, nq=nq)
        return '\n'.join([base_code, suffix])


class SymbolicDiscretizedModel(SymbolicModel):
    
    @property
    @abc.abstractmethod
    def dt(self):
        '''The time step in a discrete-time transition.'''
        raise NotImplementedError("Pure abstract method.")

    
    def _init_variables(self):
        '''Initialize model variables.'''
        self.var_names = set.union(self.var_names, {'dt'})
        super()._init_variables()
    
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
            msg = "First argument in discretized functions must be `t`."
            raise RuntimeError(msg)
        
        # Switch the time argument with the transition times
        td = np.hstack([self.vars['t'], self.vars['dt']])
        return [('td', td)] + arg_items[1:]

    def print_class(self, printer, name=None, signature=''):
        base_code = super().print_class(printer, name, signature)
        nwd = self.functions['gd'].out.shape[1]
        suffix = discretized_model_append_template.format(nwd=nwd)
        return '\n'.join([base_code, suffix])


class EulerDiscretizedModel(SymbolicDiscretizedModel):
    def _discretize(self):
        '''Discretize the drift and diffusion functions.'''
        # Get the discretization variables
        t = self.vars['t']
        dt = self.vars['dt']
        
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


class ItoTaylorAW3DiscretizedModel(SymbolicDiscretizedModel):
    '''Weak order 3 Ito--Taylor discretization for additive noise models.'''
    
    def _discretize(self):
        '''Discretize the drift and diffusion functions.'''
        # Compute the derivatives
        self.add_derivative('df_dt', 'f', 't')
        self.add_derivative('df_dx', 'f', 'x')
        self.add_derivative('d2f_dx2', 'df_dx', 'x')

        # Get the symbolic variables
        x = self.vars['x']
        dt = self.vars['dt']
        f = self.functions['f'].out
        g = self.functions['g'].out
        df_dt = self.functions['df_dt'].out
        df_dx = self.functions['df_dx'].out
        d2f_dx2 = self.functions['d2f_dx2'].out
        nw = g.shape[1]
        nx = f.size

        # Calculate the intermediates
        L0f = df_dt + np.dot(f, df_dx)
        for k, j, p, q in np.ndindex(nx, nw, nx, nx):
            L0f[k] += g[p, j] * g[q, j] * d2f_dx2[p, q, k]
        Lf = df_dx.T.dot(g)
        
        # Discretize the drift
        fd = x + f * dt + 0.5 * L0f * dt ** 2
        fdargs = self.discretized_args(self.functions['f'].args)
        self.functions['fd'] = sym2num.SymbolicFunction(fd, fdargs, 'fd')
        
        # Discretize the diffusion
        gd = np.hstack((g * dt ** 0.5 + 0.5 * Lf * dt ** 1.5,
                        0.5 * Lf * dt ** 1.5 / sympy.sqrt(3)))
        gdargs = self.discretized_args(self.functions['g'].args)
        self.functions['gd'] = sym2num.SymbolicFunction(gd, gdargs, 'gd')

