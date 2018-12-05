"""Symbolic model building using stochastic differential equations."""


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


class ItoTaylorAS15DiscretizedModel(SymbolicDiscretizedModel):
    """Strong order 1.5 Ito--Taylor discretization for additive noise models."""

    def f(self, k, x):
        t = self.t(k)
        dt = self.dt[()]
        p = self.ct_model_p
        f = self.ct_model.f(t, x, p)

        df_dx = self.ct_model.df_dx(t, x, p)
        return f * dt

