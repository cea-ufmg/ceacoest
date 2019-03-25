"""Symbolic model building using stochastic differential equations."""

import numpy as np
import sympy
from sym2num  import model


class DiscretizedSDEModelBase(model.Base):

    generate_functions = ['f', 'g', 'Q', 'h', 'R', 'x0', 'Px0']
    
    def __init__(self, ct_model=None):
        if ct_model is None:
            ct_model = self.ContinuousTimeModel()
        self.ct_model = ct_model
        """The underlying continuous-time model."""
        
        super().__init__() #Initialize base class
    
    @property
    def variables(self):
        v = self.ct_model.variables
        v.add_member('t', '(k)')
        v.add_member('dt', 'dt')
        v.add_member('ct_model_p', v['p'].tolist())
        v['k'] = 'k'
        return v

    @property
    def generate_assignments(self):
        a = getattr(self.ct_model, 'generate_assignments', {})
        a['nx'] = len(self.variables['x'])
        a['ny'] = len(self.ct_model.variables['y'])
        a['nq'] = len(self.ct_model.variables['p'])
        a['nw'] = self.default_function_output('g').shape[1]
        return a

    def Q(self, k, x):
        k = k[()]
        g = self.g(k, x).tomatrix()
        Q = g*g.T
        return sympy.Array(Q)

    def h(self, k, x):
        """Measurement function."""
        k = k[()]
        t = self.t(k)
        p = self.ct_model_p
        return self.ct_model.h(t, x, p)
    
    def R(self):
        """Measurement covariance."""
        p = self.ct_model_p
        return self.ct_model.R(p)
    
    def x0(self):
        """Initial state prior mean."""
        p = self.ct_model_p
        return self.ct_model.x0(p)
    
    def Px0(self):
        """Initial state prior covariance."""
        p = self.ct_model_p
        return self.ct_model.Px0(p)


class EulerDiscretizedSDEModel(DiscretizedSDEModelBase):
    """Euler--Maruyama SDE discretization."""
    
    def f(self, k, x):
        k = k[()]
        t = self.t(k)
        dt = self.dt[()]
        p = self.ct_model_p
        f = self.ct_model.f(t, x, p)
        return f * dt + x
    
    def g(self, k, x):
        t = self.t(k)
        dt = self.dt[()]
        p = self.ct_model_p
        g = self.ct_model.g(t, x, p)
        return g * sympy.sqrt(dt)


class ItoTaylorAS15DiscretizedModel(DiscretizedSDEModelBase):
    """Strong order 1.5 Ito--Taylor discretization for additive noise models."""

    def f(self, k, x):
        k = k[()]
        t = self.t(k)
        dt = self.dt[()]
        p = self.ct_model_p
        
        f = sympy.Matrix(self.ct_model.f(t, x, p))
        g = sympy.Matrix(self.ct_model.g(t, x, p))
        
        df_dt = sympy.Matrix(self.ct_model.df_dt(t, x, p))
        df_dx = self.ct_model.df_dx(t, x, p).tomatrix()
        d2f_dx2 = self.ct_model.d2f_dx2(t, x, p)
        
        x = sympy.Matrix(x)
        nx, nw = g.shape
        
        # Calculate the intermediates
        L0f = df_dt + df_dx.T * f
        for i, j, p, q in np.ndindex(nx, nw, nx, nx):
            L0f[i] += g[p, j] * g[q, j] * d2f_dx2[p, q, i]
        
        # Discretize the drift
        fd = f * dt + 0.5 * L0f * dt ** 2 + x
        return sympy.Array(fd, nx)

    def g(self, k, x):
        t = self.t(k)
        dt = self.dt[()]
        p = self.ct_model_p
        
        f = sympy.Matrix(self.ct_model.f(t, x, p))
        g = sympy.Matrix(self.ct_model.g(t, x, p))
        
        nx, nw = g.shape
        
        df_dt = sympy.Matrix(self.ct_model.df_dt(t, x, p))
        df_dx = self.ct_model.df_dx(t, x, p).tomatrix()
        d2f_dx2 = self.ct_model.d2f_dx2(t, x, p)
        
        # Calculate the intermediates
        Lf = df_dx.T * g
        
        # Discretize the diffusion
        blk1 = g * dt ** 0.5 + 0.5 * Lf * dt ** 1.5
        blk2 = 0.5 * Lf * dt ** 1.5 / sympy.sqrt(3)
        gd = blk1.col_insert(nw, blk2)
        return sympy.Array(gd)
    
