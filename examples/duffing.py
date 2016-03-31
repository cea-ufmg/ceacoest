"""Test various modules using a Duffing oscillator SDE model."""


import yaipopt
import numpy as np
import sympy
import sym2num
from numpy import ma
from scipy import stats

from ceacoest import kalman, sde, utils


class SymbolicDuffing(sde.SymbolicModel):
    """Symbolic Duffing oscillator model."""

    var_names = {'t', 'x', 'y', 'q', 'c'}
    """Name of model variables."""
    
    function_names = {'f', 'g', 'h', 'R', 'x0', 'Px0'}
    """Name of the model functions."""

    t = 't'
    """Time variable."""
    
    x = ['x', 'v']
    """State vector."""
    
    y = ['x_meas']
    """Measurement vector."""
    
    q = ['alpha', 'beta', 'delta', 'g2', 'x_meas_std', 
         'x0', 'v0']
    """Unknown parameter vector."""
    
    c = ['gamma', 'omega', 'x0_std', 'v0_std']
    """Constants vector."""
    
    def f(self, t, x, q, c):
        """Drift function."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        f1 = s.v
        f2 = (-s.delta * s.v - s.beta * s.x - s.alpha * s.x ** 3  +
              s.gamma * sympy.cos(s.t * s.omega))
        return [f1, f2]
    
    def g(self, t, x, q, c):
        """Diffusion matrix."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        return [[0, 0], [0, s.g2]]
    
    def h(self, t, x, q, c):
        """Measurement function."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        return [s.x]
    
    def R(self, q, c):
        """Measurement function."""
        s = self.symbols(q=q, c=c)
        return [[s.x_meas_std ** 2]]
    
    def x0(self, q, c):
        """Initial state."""
        s = self.symbols(q=q, c=c)
        return self.pack('x', dict(x=s.x0, v=s.v0))

    def Px0(self, q, c):
        """Initial state covariance."""
        s = self.symbols(q=q, c=c)
        return np.diag(self.pack('x', dict(x=s.x0_std, v=s.v0_std))**2)


class SymbolicDTDuffing(SymbolicDuffing, sde.ItoTaylorAS15DiscretizedModel):
    derivatives = [('df_dx', 'f', 'x'), ('df_dq', 'f', 'q'),
                   ('d2f_dx2', 'df_dx',  'x'), 
                   ('d2f_dx_dq', 'df_dx', 'q'),
                   ('d2f_dq2', 'df_dq',  'q'),
                   ('d3f_dx_dq2', 'd2f_dx_dq', 'q'),
                   ('d3f_dx2_dq', 'd2f_dx2',  'q'), 
                   ('dQ_dx', 'Q', 'x'), ('dQ_dq', 'Q', 'q'),
                   ('d2Q_dx2', 'dQ_dx',  'x'), 
                   ('d2Q_dx_dq', 'dQ_dx', 'q'),
                   ('d2Q_dq2', 'dQ_dq',  'q'),
                   ('dh_dx', 'h', 'x'), ('dh_dq', 'h', 'q'),
                   ('d2h_dx2', 'dh_dx',  'x'), 
                   ('d2h_dx_dq', 'dh_dx', 'q'),
                   ('d2h_dq2', 'dh_dq',  'q'),
                   ('dR_dq', 'R', 'q'), ('d2R_dq2', 'dR_dq', 'q'),
                   ('dx0_dq', 'x0', 'q'), ('d2x0_dq2', 'dx0_dq', 'q'),
                   ('dPx0_dq', 'Px0', 'q'), ('d2Px0_dq2', 'dPx0_dq', 'q')]
    """List of the model function derivatives to calculate / generate."""
    
    dt = 'dt'
    """Discretization time step."""

    k = 'k'
    """Discretized sample index."""

    generated_name = "GeneratedDTDuffing"
    """Name of the generated class."""
    
    meta = 'ceacoest.sde.DiscretizedModel.meta'
    """Generated model metaclass."""

    @property
    def imports(self):
        return super().imports + ('import ceacoest.sde',)
    

sym_duffing = SymbolicDuffing()
sym_dt_duffing = SymbolicDTDuffing()
printer = sym2num.ScipyPrinter()
GeneratedDTDuffing = sym2num.class_obj(sym_dt_duffing, printer)


def sim():
    np.random.seed(1)

    # Generate the time vector
    dt = 0.05
    N = int(30 // dt)
    k = np.arange(N)
    t = k * dt

    # Instantiate the model
    given = dict(
        alpha=1, beta=-1, delta=0.2, gamma=0.3, omega=1,
        g1=0, g2=0.1, x_meas_std=0.1,
        x0=1, v0=0, x0_std=0.1, v0_std=0.1
    )
    q = GeneratedDTDuffing.pack('q', given)
    c = GeneratedDTDuffing.pack('c', given)
    params = dict(q=q, c=c, dt=dt)
    sampled = dict(t=t)
    model = GeneratedDTDuffing(params, sampled)

    # Simulate the system
    w = np.random.randn(N - 1, model.nw)
    x = np.zeros((N, model.nx))
    x[0] = stats.multivariate_normal.rvs(model.x0(), model.Px0())
    for k in range(N - 1):
        x[k + 1] = model.f(k, x[k])  + model.g(k, x[k]).dot(w[k])
    
    # Sample the outputs
    R = model.R()
    v = np.random.multivariate_normal(np.zeros(model.ny), R, N)
    y = ma.asarray(model.h(k, x) + v)
    y[np.arange(N) % 4 != 0] = ma.masked
    return model, t, x, y, q


def pem(model, t, x, y, q):
    def merit(q, new=None):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedFilter(mq)
        return kf.pem_merit(y)
    
    def grad(q, new=None):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedFilter(mq)
        return kf.pem_gradient(y)
    
    hess_inds = np.tril_indices(model.nq)
    def hess(q, new_q=1, obj_factor=1, lmult=1, new_lmult=1):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedFilter(mq)
        return obj_factor * kf.pem_hessian(y)[hess_inds]
    
    q_lb = dict(g2=0, x_meas_std=0, x0_std=0, v0_std=0)
    q_ub = dict()
    q_fix = dict()
    q_bounds = [model.pack('q', dict(q_lb, **q_fix), fill=-np.inf),
                model.pack('q', dict(q_ub, **q_fix), fill=np.inf)]
    problem = yaipopt.Problem(q_bounds, merit, grad,
                              hess=hess, hess_inds=hess_inds)
    problem.num_option(b'obj_scaling_factor', -1)
    (qopt, solinfo) = problem.solve(q)
    
    return problem, qopt, solinfo


if __name__ == '__main__':
    [model, t, x, y, q] = sim()
    [problem, qopt, solinfo] = pem(model, t, x, y, q)
    mopt = model.parametrize(q=qopt)
    kfopt = kalman.DTUnscentedFilter(mopt)
    [xs, Pxs] = kfopt.smooth(y)
    
