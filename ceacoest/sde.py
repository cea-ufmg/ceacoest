"""Stochastic differential equation module.

TODO
----
 * Merge signatures of f and g for the Ito-Taylor order 1.5 discretization.

"""


import abc
import collections

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


class SymbolicModel(sym2num.SymbolicModel):

    var_names = {'t', 'x'}
    """Name of the model variables."""
    
    function_names = {'f', 'g'}
    """Name of the model functions."""
    
    @property
    @abc.abstractmethod
    def t(self):
        """The time variable."""
        raise NotImplementedError("Pure abstract property, must be overloaded.")

    @property
    @abc.abstractmethod
    def x(self):
        """The state vector."""
        raise NotImplementedError("Pure abstract property, must be overloaded.")
    
    @abc.abstractmethod
    def f(self, t, x, *args):
        """SDE drift function."""
        raise NotImplementedError("Pure abstract method, must be overloaded.")

    @abc.abstractmethod
    def g(self, t, x, *args):
        """SDE diffusion."""
        raise NotImplementedError("Pure abstract method, must be overloaded.")

    def _init_variables(self):
        """Initialize model variables."""
        self.var_names = set.union(self.var_names, {'x', 't'})
        super()._init_variables()
    
    def _init_functions(self):
        """Initialize model functions."""
        self.function_names = set.union(self.function_names, {'f', 'g'})
        super()._init_functions()
        
        # Add default implementation for Q = g * g.T
        if 'Q' not in self.functions:
            self._add_default_Q()

    def _add_default_Q(self):
        """Add default implementation for `Q = g * g.T`."""
        g = self.functions['g']
        Qexpr = np.dot(g.out, g.out.T)
        self.functions['Q'] = sym2num.SymbolicFunction(Qexpr, g.args, 'Q')
    
    def print_class(self, printer):
        base_code = super().print_class(printer)
        nx = self.vars['x'].size
        nw = self.functions['g'].out.shape[1]
        nq = self.vars['q'].size if 'q' in self.vars else 0
        ny = self.vars['y'].size if 'y' in self.vars else 0
        suffix = sde_model_append_template.format(nx=nx, ny=ny, nw=nw, nq=nq)
        return '\n'.join([base_code, suffix])


class SymbolicDiscretizedModel(SymbolicModel):
    
    @property
    @abc.abstractmethod
    def k(self):
        """The discrete-time sample index."""
        raise NotImplementedError("Pure abstract property, must be overloaded.")
    
    @property
    @abc.abstractmethod
    def dt(self):
        """The time step in a discrete-time transition."""
        raise NotImplementedError("Pure abstract property, must be overloaded.")
    
    def _init_variables(self):
        """Initialize model variables."""
        self.var_names = set.union(self.var_names, {'dt', 'k'})
        super()._init_variables()
    
    def _init_functions(self):
        """Initialize the model functions."""
        super()._init_functions() # Initialize base class functions
        self._discretize_sde() # Generate the discretized drift and diffusion
        self._discretize_extra() # Discretize any remaining model functions
    
    @abc.abstractmethod
    def _discretize_sde(self):
        """Discretize the drift and diffusion functions."""
        raise NotImplementedError("Pure abstract method, must be overloaded.")

    def _discretize_extra(self):
        """Discretize other model functions."""
        for f in self.functions.values():
            fargs = f.args
            f.args = self._discretized_args(fargs)
    
    def _discretized_args(self, args):
        # If the function is not time-dependant the arguments are unchanged
        if 't' not in args:
            return args

        # Get the discretization variables
        dt = self.vars['dt']
        k = self.vars['k']
        
        # Switch the time argument with the sample index
        arg_items = list(args.items())
        t_index = list(args).index('t')
        t_item = arg_items[t_index]
        arg_items[t_index] = ('k', k)
        arg_items.extend([t_item, ('dt', dt)])
        return collections.OrderedDict(arg_items)


class EulerDiscretizedModel(SymbolicDiscretizedModel):
    def _discretize_sde(self):
        """Discretize the drift and diffusion functions."""
        # Get the discretization variables
        t = self.vars['t']
        dt = self.vars['dt']
        
        # Discretize the drift
        f = self.functions['f']
        fd = self.vars['x'] + f.out * dt
        fdargs = self._discretized_args(f.args)
        self.functions['f'] = sym2num.SymbolicFunction(fd, fdargs, 'f')
        
        # Discretize the diffusion
        g = self.functions['g']
        gd = g.out * dt ** 0.5
        gdargs = self._discretized_args(g.args)
        self.functions['g'] = sym2num.SymbolicFunction(gd, gdargs, 'g')
        
        # Add default implementation for transition covariance
        self._add_default_Q()


class ItoTaylorAS15DiscretizedModel(SymbolicDiscretizedModel):
    """Strong order 1.5 Ito--Taylor discretization for additive noise models."""
    
    def _discretize_sde(self):
        """Discretize the drift and diffusion functions."""
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
        fdargs = self._discretized_args(self.functions['f'].args)
        self.functions['f'] = sym2num.SymbolicFunction(fd, fdargs, 'f')
        
        # Discretize the diffusion
        gd = np.hstack((g * dt ** 0.5 + 0.5 * Lf * dt ** 1.5,
                        0.5 * Lf * dt ** 1.5 / sympy.sqrt(3)))
        gdargs = self._discretized_args(self.functions['g'].args)
        self.functions['g'] = sym2num.SymbolicFunction(gd, gdargs, 'g')
        
        # Add default implementation for transition covariance
        self._add_default_Q()



class DiscretizedModel(sym2num.ParametrizedModel):
    def __init__(self, params={}, sampled={}):
        super().__init__(params)
        self._sampled = {k: np.asarray(v) for k, v in sampled.items()}
    
    def call_args(self, f, *args, **kwargs):
        # Get the base call arguments
        call_args = super().call_args(f, *args, **kwargs)
    
        # Include the samples of the discretized variables
        if 'k' in call_args:
            k = np.asarray(call_args['k'], dtype=int)
            fargs = self.signatures[f.__name__]
            sampled_fargs = set(fargs).intersection(self._sampled)
            for arg_name in sampled_fargs.difference(call_args):
                call_args[arg_name] = self._sampled[arg_name][k]
        return call_args

    def parametrize(self, params={}, **kwparams):
        model = super().parametrize(params, **kwparams)
        model._sampled = self._sampled.copy()
        return model

