"""Symbolic model building using stochastic differential equations."""

import numpy as np
import sympy
from sym2num  import model


class DiscretizedSDEModelBase(model.Base):
    
    def __init__(self, ct_model=None):
        if ct_model is None:
            ct_model = self.ContinuousTimeModel()
        self.ct_model = ct_model
        """The underlying continuous-time model."""
    
    @property
    def variables(self):
        v = self.ct_model.variables
        v.add_member('t', '(k)')
        v.add_member('dt', 'dt')
        v.add_member('ct_model_p', v['p'].tolist())
        v['k'] = 'k'
        return v

    def Q(self, k, x):
        g = self.g(k, x).tomatrix()
        Q = g*g.T
        return sympy.Array(Q)


class EulerDiscretizedSDEModel(DiscretizedSDEModelBase):
    """Euler--Maruyama SDE discretization."""
    
    def f(self, k, x):
        t = self.t(k)
        dt = self.dt[()]
        p = self.ct_model_p
        f = self.ct_model.f(t, x, p)
        return f * dt
    
    def g(self, k, x):
        t = self.t(k)
        dt = self.dt[()]
        p = self.ct_model_p
        g = self.ct_model.g(t, x, p)
        return g * sympy.sqrt(dt)


class ItoTaylorAS15DiscretizedModel(DiscretizedSDEModelBase):
    """Strong order 1.5 Ito--Taylor discretization for additive noise models."""

    def f(self, k, x):
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
        for k, j, p, q in np.ndindex(nx, nw, nx, nx):
            L0f[k] += g[p, j] * g[q, j] * d2f_dx2[p, q, k]
        
        # Discretize the drift
        fd = x + f * dt + 0.5 * L0f * dt ** 2
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

