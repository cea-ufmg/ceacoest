"""Symbolic model building using stochastic differential equations."""

import numpy as np
import sympy
import sym2num.model


class DiscretizedSDEModelBase(sym2num.model.Base):

    generate_functions = ['f', 'g', 'Q', 'h', 'R']
    
    def __init__(self, ct_model=None):
        super().__init__() #Initialize base class
        
        if ct_model is None:
            ct_model = self.ContinuousTimeModel()
        self.ct_model = ct_model
        """The underlying continuous-time model."""
        
        v = self.variables
        v['self'] = ct_model.variables.self.copy()
        v['self']['dt'] = 'dt'
        v['k'] = 'k'
        v['x'] = ct_model.variables.x
        v['y'] = ct_model.variables.y
    
    @property
    def generate_assignments(self):
        a = {}
        a['nx'] = len(self.variables['x'])
        a['ny'] = len(self.ct_model.variables['y'])
        a['nw'] = self.default_function_output('g').shape[1]
        return a
    
    def Q(self, k, x):
        """State transition covariance matrix."""
        g = np.asarray(self.g(k, x))
        Q = g @ g.T
        return Q
    
    def h(self, k, x):
        """Measurement function."""
        t = k * self.dt
        return self.ct_model.h(t, x)
    
    def R(self):
        """Measurement covariance."""
        return self.ct_model.R()


class EulerDiscretizedSDEModel(DiscretizedSDEModelBase):
    """Euler--Maruyama SDE discretization."""
    
    generate_imports = ["ceacoest.modelling.gensde as _gensde"]
    generated_bases = ["_gensde.ConditionalGaussianTransition"]
    
    def f(self, k, x):
        """State transition function."""
        dt = self.dt
        t = dt * k
        f = self.ct_model.f(t, x)
        return f * dt + x
    
    def g(self, k, x):
        """State transition noise gain matrix."""
        dt = self.dt
        t = dt * k
        g = self.ct_model.g(t, x)
        return g * sympy.sqrt(dt)


class ItoTaylorAS15DiscretizedModel(DiscretizedSDEModelBase):
    """Strong order 1.5 Ito--Taylor discretization for additive noise models."""
    
    generate_imports = ["ceacoest.modelling.gensde as _gensde"]
    generated_bases = ["_gensde.ConditionalGaussianTransition"]
    
    def __init__(self, ct_model=None):
        super().__init__(ct_model)
        
        # Ensure the continuous model has all necessary derivatives
        if not hasattr(self.ct_model, 'df_dt'):
            self.ct_model.add_derivative('f', 't', 'df_dt')
        if not hasattr(self.ct_model, 'df_dx'):
            self.ct_model.add_derivative('f', 'x', 'df_dx')
        if not hasattr(self.ct_model, 'd2f_dx2'):
            self.ct_model.add_derivative('df_dx', 'x', 'd2f_dx2')        
    
    def f(self, k, x):
        dt = self.dt
        t = dt * k
        
        f = self.ct_model.f(t, x)
        g = self.ct_model.g(t, x)
        
        df_dt = self.ct_model.df_dt(t, x)
        df_dx = self.ct_model.df_dx(t, x)
        d2f_dx2 = self.ct_model.d2f_dx2(t, x)
        
        nx, nw = g.shape
        
        # Calculate the intermediates
        L0f = df_dt + df_dx.T @ f
        for i, j, p, q in np.ndindex(nx, nw, nx, nx):
            L0f[i] += 0.5 * g[p, j] * g[q, j] * d2f_dx2[p, q, i]
        
        # Discretize the drift
        fd = f * dt + 0.5 * L0f * dt ** 2 + x
        return fd
    
    def g(self, k, x):
        dt = self.dt
        t = dt * k
        
        f = self.ct_model.f(t, x)
        g = self.ct_model.g(t, x)
        
        df_dt = self.ct_model.df_dt(t, x)
        df_dx = self.ct_model.df_dx(t, x)
        d2f_dx2 = self.ct_model.d2f_dx2(t, x)
        
        nx, nw = g.shape
        
        # Calculate the intermediates
        Lf = df_dx.T @ g
        
        # Discretize the diffusion
        blk1 = g * dt ** 0.5 + 0.5 * Lf * dt ** 1.5
        blk2 = 0.5 * Lf * dt ** 1.5 / sympy.sqrt(3)
        return np.hstack((blk1, blk2))
    
