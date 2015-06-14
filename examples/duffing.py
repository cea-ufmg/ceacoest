'''Test various modules using a Duffing oscillator SDE model.'''


import ipopt
import numpy as np
import sympy
import sym2num
from numpy import ma
from scipy import stats

from qwfilter import kalman, sde, utils


class SymbolicDuffing(sde.SymbolicModel):
    '''Symbolic Duffing oscillator model.'''

    var_names = {'t', 'x', 'y', 'q', 'c'}
    '''Name of model variables.'''
    
    function_names = {'f', 'g', 'h', 'R'}
    '''Name of the model functions.'''

    t = 't'
    '''Time variable.'''
    
    x = ['x', 'v']
    '''State vector.'''
    
    y = ['x_meas']
    '''Measurement vector.'''
    
    q = ['alpha', 'beta', 'delta', 'g2', 'x_meas_std']
    '''Unknown parameter vector.'''
    
    c = ['gamma', 'omega']
    '''Constants vector.'''
    
    def f(self, t, x, q, c):
        '''Drift function.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        f1 = s.v
        f2 = (-s.delta * s.v - s.beta * s.x - s.alpha * s.x ** 3  +
              s.gamma * sympy.cos(s.t * s.omega))
        return [f1, f2]
    
    def g(self, t, x, q, c):
        '''Diffusion matrix.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        return [[0, 0], [0, s.g2]]
    
    def h(self, t, x, q, c):
        '''Measurement function.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        return [s.x]
    
    def R(self, q, c):
        '''Measurement function.'''
        s = self.symbols(q=q, c=c)
        return [[s.x_meas_std ** 2]]


class SymbolicDTDuffing(SymbolicDuffing, sde.ItoTaylorAS15DiscretizedModel):
    derivatives = [('df_dx', 'f', 'x'), ('df_dq', 'f', 'q'),
                   ('d2f_dx2', 'df_dx',  'x'), 
                   ('d2f_dx_dq', 'df_dx', 'q'),
                   ('d2f_dq2', 'df_dq',  'q'),
                   ('dQ_dx', 'Q', 'x'), ('dQ_dq', 'Q', 'q'),
                   ('d2Q_dx2', 'dQ_dx',  'x'), 
                   ('d2Q_dx_dq', 'dQ_dx', 'q'),
                   ('d2Q_dq2', 'dQ_dq',  'q'),
                   ('dh_dx', 'h', 'x'), ('dh_dq', 'h', 'q'),
                   ('d2h_dx2', 'dh_dx',  'x'), 
                   ('d2h_dx_dq', 'dh_dx', 'q'),
                   ('d2h_dq2', 'dh_dq',  'q'),
                   ('dR_dq', 'R', 'q'), ('d2R_dq2', 'dR_dq', 'q')]
    '''List of the model function derivatives to calculate / generate.'''
    
    dt = 'dt'
    '''Discretization time step.'''

    k = 'k'
    '''Discretized sample index.'''


sym_duffing = SymbolicDuffing()
sym_dt_duffing = SymbolicDTDuffing()
printer = sym2num.ScipyPrinter()
GeneratedDTDuffing = sym2num.class_obj(
    sym_dt_duffing, printer,
    name='GeneratedDTDuffing', 
    meta=sde.DiscretizedModel.meta
)


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
        g1=0, g2=0.1, x_meas_std=0.1
    )
    q = GeneratedDTDuffing.pack('q', given)
    c = GeneratedDTDuffing.pack('c', given)
    params = dict(q=q, c=c, dt=dt)
    sampled = dict(t=t)
    model = GeneratedDTDuffing(params, sampled)

    # Simulate the system
    w = np.random.randn(N - 1, model.nw)
    x = np.zeros((N, model.nx))
    x[0] = [1, 0]
    for k in range(N - 1):
        x[k + 1] = model.f(k, x[k])  + model.g(k, x[k]).dot(w[k])

    # Sample the outputs
    R = model.R()
    v = np.random.multivariate_normal(np.zeros(model.ny), R, N)
    y = ma.asarray(model.h(k, x) + v)
    y[np.arange(N) % 4 != 0] = ma.masked
    return model, t, x, y, q


def pem(model, t, x, y, q):
    x0 = [1.2, 0.2]
    Px0 = np.diag([0.1, 0.1])
    
    def merit(q, new=None):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedKalmanFilter(mq, x0, Px0)
        return kf.pem_merit(y)
    
    def grad(q, new=None):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedKalmanFilter(mq, x0, Px0)
        return kf.pem_gradient(y)
    
    hess_inds = np.tril_indices(model.nq)
    def hess(q, new_q=1, obj_factor=1, lmult=1, new_lmult=1):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedKalmanFilter(mq, x0, Px0)
        return obj_factor * kf.pem_hessian(y)[hess_inds]
    
    q_bounds = np.tile([[-np.inf], [np.inf]], model.nq)
    problem = ipopt.Problem(q_bounds, merit, grad, 
                            hess=hess, hess_inds=hess_inds)
    problem.num_option(b'obj_scaling_factor', -1)
    (qopt, solinfo) = problem.solve(q)

    return problem, qopt, solinfo, x0, Px0

