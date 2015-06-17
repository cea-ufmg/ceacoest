'''Prediction Error Method for the HFB-320 aircraft longitudinal motion.'''


import os.path

import ipopt
import numpy as np
import sympy
import sym2num
from numpy import ma
from scipy import stats

from qwfilter import kalman, sde, utils


class SymbolicModel(sde.SymbolicModel):
    '''Symbolic HFB-320 longitudinal motion model.'''

    var_names = {'t', 'x', 'y', 'u', 'q', 'c'}
    '''Name of model variables.'''
    
    function_names = {'f', 'g', 'h', 'R'}
    '''Name of the model functions.'''

    t = 't'
    '''Time variable.'''
    
    x = ['vT', 'alpha', 'theta', 'q']
    '''State vector.'''
    
    y = ['vT_meas', 'alpha_meas', 'theta_meas', 'q_meas', 'ax_meas', 'az_meas']
    '''Measurement vector.'''

    u = ['de', 'Fe']
    '''Exogenous input vector.'''
    
    q = ['CD0', 'CDv', 'CDa', 'CL0', 'CLv', 'CLa',
         'Cm0', 'Cmv', 'Cma', 'Cmq', 'Cmde', 
         'alpha_png', 'vT_png', 'q_png',
         'vT_mng', 'alpha_mng', 'theta_mng', 'q_mng', 'ax_mng', 'az_mng']
    '''Unknown parameter vector.'''
    
    c = ['g0', 'Sbym', 'Scbyiy', 'Feiylt', 'v0', 'rm', 'sigmat', 'rho']
    '''Constants vector.'''

    def coefs(self, x, q, u, c):
        s = self.symbols(x=x, q=q, u=u, c=c)
        CD = s.CD0 + s.CDv*s.vT/s.v0 + s.CDa*s.alpha
        CL = s.CL0 + s.CLv*s.vT/s.v0 + s.CLa*s.alpha
        Cm = (s.Cm0 + s.Cmv*s.vT/s.v0 + s.Cma*s.alpha + s.Cmq*s.q/s.v0 +
              s.Cmde*s.de)
        Cx = CL*sympy.sin(s.alpha) - CD*sympy.cos(s.alpha)
        Cz = -CL*sympy.cos(s.alpha) - CD*sympy.sin(s.alpha)
        return [CD, CL, Cm, Cx, Cz]
    
    def f(self, t, x, q, u, c):
        '''Drift function.'''
        [CD, CL, Cm, Cx, Cz] = self.coefs(x, q, u, c)
        s = self.symbols(t=t, x=x, q=q, u=u, c=c)
        qbar = 0.5 * s.rho * s.vT ** 2
        vTdot = (-s.Sbym*qbar*CD + s.Fe*sympy.cos(s.alpha + s.sigmat)/s.rm +
                 s.g0*sympy.sin(s.alpha - s.theta))
        alphadot = (-s.Sbym*qbar*CL/s.vT + s.q -
                    s.Fe*sympy.sin(s.alpha + s.sigmat)/s.rm/s.vT +
                    s.g0*sympy.cos(s.alpha - s.theta)/s.vT)
        qdot = s.Scbyiy*qbar*Cm + s.Feiylt*s.Fe        
        derivs = dict(vT=vTdot, alpha=alphadot, theta=s.q, q=qdot)
        return self.pack('x', derivs)
    
    def g(self, t, x, q, u, c):
        '''Diffusion matrix.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        return np.c_[self.pack('x', vT=s.vT_png),
                     self.pack('x', alpha=s.alpha_png),
                     self.pack('x', q=s.q_png)]
    
    def h(self, t, x, q, u, c):
        '''Measurement function.'''
        [CD, CL, Cm, Cx, Cz] = self.coefs(x, q, u, c)
        s = self.symbols(t=t, x=x, q=q, u=u, c=c)
        qbar = 0.5 * s.rho * s.vT ** 2
        ax = qbar*s.Sbym*Cx + s.Fe*sympy.cos(s.sigmat)/s.rm
        az = qbar*s.Sbym*Cz - s.Fe*sympy.sin(s.sigmat)/s.rm
        meas = dict(vT_meas=s.vT, alpha_meas=s.alpha,
                    theta_meas=s.theta, q_meas=s.q, 
                    ax_meas=ax, az_meas=az)
        return self.pack('y', meas)
    
    def R(self, q, c):
        '''Measurement function.'''
        s = self.symbols(q=q, c=c)
        mng = dict(vT_meas=s.vT_mng, alpha_meas=s.alpha_mng, 
                   theta_meas=s.theta_mng, q_meas=s.q_mng, 
                   ax_meas=s.ax_mng, az_meas=s.az_mng)
        return np.diag(self.pack('y', mng)) ** 2


class SymbolicDTModel(SymbolicModel, sde.EulerDiscretizedModel):
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


sym_dt_model = SymbolicDTModel()
printer = sym2num.ScipyPrinter()
GeneratedDTModel = sym2num.class_obj(
    sym_dt_model, printer,
    name='GeneratedDTModel', 
    meta=sde.DiscretizedModel.meta
)


def load_data():
    module_dir = os.path.dirname(__file__)
    data = np.loadtxt(os.path.join(module_dir, 'data', 'hfb320_1_10.asc'))
    t = data[:, 0]
    u = data[:, [1, 3]]
    y = data[:, [4, 5, 6, 8, 9, 10]]
    return t, u, y


def pem(t, u, y):
    given = dict(
        g0=9.80665,
        Sbym=4.02800e-3,
        Scbyiy=8.00270e-4,
        Feiylt=-7.01530e-6,
        v0=104.67,
        rm=7472,
        sigmat=0.05240,
        rho=0.79200,
        CD0=1.2e-1, CDv=-6.5e-2, CDa=3.2e-1,
        CL0=-9.2e-2, CLv=1.5e-1, CLa=4.3,
        Cm0=1.1e-1, Cmv=4e-3, Cma=-9.7e-1,
        Cmq=-3.5e1, Cmde=-1.5,
        vT_png=1e-1, alpha_png=1e-3, q_png=2e-3,
        vT_mng=0.05, alpha_mng=1e-3, theta_mng=5e-4, q_mng=5e-4, 
        ax_mng=1e-2, az_mng=5e-2
    )
    q = GeneratedDTModel.pack('q', given)
    c = GeneratedDTModel.pack('c', given)
    dt = t[1] - t[0]
    params = dict(q=q, c=c, dt=dt)
    sampled = dict(t=t, u=u)
    model = GeneratedDTModel(params, sampled)
    
    x0 = [1.06023e2, 1.11685e-1, 1.04887e-1, -3.32659e-3]
    Px0 = np.diag([1, 0.005, 0.01, 0.02]) ** 2
    
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
    
    q_lb = dict(
        alpha_png=0, vT_png=0, q_png=0,
        vT_mng=0.3, alpha_mng=1e-3/np.sqrt(12),
        theta_mng=5e-4/np.sqrt(12), q_mng=5e-4/np.sqrt(12),
        ax_mng=1e-2/np.sqrt(12), az_mng=5e-2/np.sqrt(12)
    )
    q_ub = dict(q_mng=1e-2)
    q_fix = dict(theta_png=0)
    q_bounds = [model.pack('q', dict(q_lb, **q_fix), fill=-np.inf),
                model.pack('q', dict(q_ub, **q_fix), fill=np.inf)]
    problem = ipopt.Problem(q_bounds, merit, grad,
                            hess=hess, hess_inds=hess_inds)
    problem.num_option(b'obj_scaling_factor', -1)
    (qopt, solinfo) = problem.solve(q)
    return problem, qopt, solinfo

#qopt = array([1.24545680e-01, -6.57255675e-02, 3.14555447e-01,
#              -1.23033576e-01, 1.61966681e-01, 4.48616228e+00,
#              1.04982744e-01, 1.93904359e-02, -1.06164967e+00,
#              -5.44799807e+01, -1.70326794e+00, 2.47700970e-03,
#              3.37413258e-01, 3.50023101e-03, 6.07633176e-02,
#              1.59364385e-03, 3.87056032e-04, 2.14532443e-02,
#              1.04067154e-02, 7.41069189e-02])
