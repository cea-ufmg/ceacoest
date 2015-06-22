'''Path reconstruction of an Ascention Technology trakSTAR sensor.'''


import os

import ipopt
import numpy as np
import sympy
import sym2num
from numpy import ma
from scipy import io

from qwfilter import kalman, sde, utils


class SymbolicModel(sde.SymbolicModel):
    '''Symbolic SDE model.'''
    
    var_names = {'t', 'x', 'y', 'q', 'c'}
    '''Name of model variables.'''
    
    function_names = {'f', 'g', 'h', 'R'}
    '''Name of the model functions.'''
    
    t = 't'
    '''Time variable.'''
    
    x = ['x', 'y', 'z', 'u', 'v', 'w', 
         'phi', 'theta', 'psi', 'p', 'q', 'r']
    '''State vector.'''
    
    y = ['x_meas', 'y_meas', 'z_meas', 'phi_meas', 'theta_meas', 'psi_meas']
    '''Measurement vector.'''
    
    q = ['linvel_png', 'angvel_png', 'pos_meas_std', 'ang_meas_std']
    '''Parameter vector.'''
    
    c = []
    '''Constants vector.'''
    
    def f(self, t, x, q, c):
        '''Drift function.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        sinphi = sympy.sin(s.phi)
        cosphi = sympy.cos(s.phi)
        costheta = sympy.cos(s.theta)
        tantheta = sympy.tan(s.theta)
        derivs = dict(
            x=s.u, y=s.v, z=s.w, u=0, v=0, w=0, p=0, q=0, r=0,
            phi=s.p + s.q*tantheta*sinphi + s.r*tantheta*cosphi,
            theta=s.q*cosphi - s.r*sinphi,
            psi=s.q*sinphi/costheta + s.r*cosphi/costheta
        )
        return self.pack('x', derivs)
    
    def g(self, t, x, q, c):
        '''Diffusion matrix.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        g = np.zeros((x.size, 6), object)
        g[[3, 4, 5], [0, 1, 2]] = s.linvel_png
        g[[9, 10, 11], [3, 4, 5]] = s.angvel_png
        return g
    
    def h(self, t, x, q, c):
        '''Measurement function.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        meas = dict(x_meas=s.x, y_meas=s.y, z_meas=s.z, 
                    phi_meas=s.phi, theta_meas=s.theta, psi_meas=s.psi)
        return self.pack('y', meas)
    
    def R(self, q, c):
        '''Measurement function.'''
        s = self.symbols(q=q, c=c)
        R = np.diag(np.repeat([s.pos_meas_std, s.ang_meas_std], 3))**2
        return R


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


sym_model = SymbolicModel()
sym_dt_model = SymbolicDTModel()
printer = sym2num.ScipyPrinter()
GeneratedDTModel = sym2num.class_obj(
    sym_dt_model, printer,
    name='GeneratedDTModel', 
    meta=sde.DiscretizedModel.meta
)


def load_data():
    module_dir = os.path.dirname(__file__)
    filepath = os.path.join(module_dir, 'data', 'trakstar.mat')
    interval = slice(1, 3750)
    
    data = io.loadmat(filepath)
    tmeas = data['time'].flatten()[interval]
    tmeas -= tmeas[0]
    q0, q1, q2, q3 = data['q'][interval].T
    phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    theta = np.arcsin(2*(q0*q2 - q1*q3))
    psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    
    y_dict = dict(
        x_meas=data['x'].flatten()[interval],
        y_meas=data['y'].flatten()[interval],
        z_meas=data['z'].flatten()[interval],
        phi_meas=phi, theta_meas=theta, psi_meas=psi,
    )
    y_dict['x_meas'] -= y_dict['x_meas'][0]
    y_dict['y_meas'] -= y_dict['y_meas'][0]
    y_dict['z_meas'] -= y_dict['z_meas'][0]
    y = GeneratedDTModel.pack('y', y_dict, fill=np.zeros_like(tmeas))
    return tmeas, y


def pem(t, y):
    # Instantiate the model
    given = dict(
        pos_meas_std=0.04, ang_meas_std=5e-4,
        linvel_png=50, angvel_png=5,
    )
    dt = t[1] - t[0]
    q0 = GeneratedDTModel.pack('q', given)
    c = GeneratedDTModel.pack('c', given)
    params = dict(q=q0, c=c, dt=dt)
    sampled = dict(t=t)
    model = GeneratedDTModel(params, sampled)
    x0 = np.zeros(GeneratedDTModel.nx)
    x0[-3:] = y[0, -3:]
    Px0 = np.diag(np.repeat([1, 10, 1e-3, 1e-3], 3))
    
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
        pos_meas_std=0, ang_meas_std=0,
        linvel_png=0, angvel_png=0,
    )
    q_ub = dict()
    q_fix = dict()
    q_bounds = [model.pack('q', dict(q_lb, **q_fix), fill=-np.inf),
                model.pack('q', dict(q_ub, **q_fix), fill=np.inf)]
    problem = ipopt.Problem(q_bounds, merit, grad,
                            hess=hess, hess_inds=hess_inds)
    problem.num_option(b'obj_scaling_factor', -1)
    (qopt, solinfo) = problem.solve(q0)
    
    return problem, qopt, solinfo, model, q0, x0, Px0
