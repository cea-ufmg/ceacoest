"""Attitude reconstruction of an Ascention Technology trakSTAR sensor."""


import os

import yaipopt
import numpy as np
import sympy
import sym2num
from numpy import ma
from scipy import io

from ceacoest import kalman, sde, utils


class SymbolicModel(sde.SymbolicModel):
    """Symbolic SDE model."""
    
    var_names = {'t', 'x', 'y', 'q', 'c'}
    """Name of model variables."""
    
    function_names = {'f', 'g', 'h', 'R'}
    """Name of the model functions."""
    
    t = 't'
    """Time variable."""
    
    x = ['phi', 'theta', 'psi', 'p', 'q', 'r']
    """State vector."""
    
    y = ['phi_meas', 'theta_meas', 'psi_meas']
    """Measurement vector."""
    
    q = ['angvel_png', 'ang_meas_std']
    """Parameter vector."""
    
    c = []
    """Constants vector."""
    
    def f(self, t, x, q, c):
        """Drift function."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        sinphi = sympy.sin(s.phi)
        cosphi = sympy.cos(s.phi)
        costheta = sympy.cos(s.theta)
        tantheta = sympy.tan(s.theta)
        derivs = dict(
            phi=s.p + s.q*tantheta*sinphi + s.r*tantheta*cosphi,
            theta=s.q*cosphi - s.r*sinphi,
            psi=s.q*sinphi/costheta + s.r*cosphi/costheta,
            p=0, q=0, r=0
        )
        return self.pack('x', derivs)
    
    def g(self, t, x, q, c):
        """Diffusion matrix."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        g = np.zeros((x.size, 3), object)
        g[[3, 4, 5], [0, 1, 2]] = s.angvel_png
        return g
    
    def h(self, t, x, q, c):
        """Measurement function."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        meas = dict(phi_meas=s.phi, theta_meas=s.theta, psi_meas=s.psi)
        return self.pack('y', meas)
    
    def R(self, q, c):
        """Measurement function."""
        s = self.symbols(q=q, c=c)
        R = np.diag(np.repeat([s.ang_meas_std], 3))**2
        return R


class SymbolicDTModel(SymbolicModel, sde.ItoTaylorAS15DiscretizedModel):
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
    """List of the model function derivatives to calculate / generate."""
    
    dt = 'dt'
    """Discretization time step."""

    k = 'k'
    """Discretized sample index."""

    generated_name = "GeneratedDTModel"
    """Name of the generated class."""
    
    meta = 'ceacoest.sde.DiscretizedModel.meta'
    """Generated model metaclass."""
    
    @property
    def imports(self):
        return super().imports + ('import ceacoest.sde',)


sym_model = SymbolicModel()
sym_dt_model = SymbolicDTModel()
printer = sym2num.ScipyPrinter()
GeneratedDTModel = sym2num.class_obj(sym_dt_model, printer)


def load_data():
    module_dir = os.path.dirname(__file__)
    filepath = os.path.join(module_dir, 'data', 'trakstar.mat')
    interval = slice(95, 400)
    upsample = 4
    
    data = io.loadmat(filepath)
    dt = (data['time'].flat[1] - data['time'].flat[0]) / upsample
    q0, q1, q2, q3 = data['q'][interval].T
    phi = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    theta = np.arcsin(2*(q0*q2 - q1*q3))
    psi = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    
    N = len(q0)
    t = np.arange(N * upsample) * dt
    y_dict = dict(phi_meas=phi, theta_meas=theta, psi_meas=psi)
    y = ma.masked_all((N * upsample, GeneratedDTModel.ny))
    y[::upsample] = GeneratedDTModel.pack('y', y_dict, fill=np.zeros_like(q0))
    return t, y


def pem(t, y):
    # Instantiate the model
    given = dict(
        ang_meas_std=2.4e-4, angvel_png=3,
    )
    dt = t[1] - t[0]
    q0 = GeneratedDTModel.pack('q', given)
    c = GeneratedDTModel.pack('c', given)
    params = dict(q=q0, c=c, dt=dt)
    sampled = dict(t=t)
    model = GeneratedDTModel(params, sampled)
    x0 = np.zeros(GeneratedDTModel.nx)
    x0[:3] = y[0, :3]
    Px0 = np.diag(np.repeat([1e-3, 1e-3], 3))
    
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
    
    q_lb = dict(ang_meas_std=0, angvel_png=0)
    q_ub = dict()
    q_fix = dict(ang_meas_std=2.4e-4)
    q_bounds = [model.pack('q', dict(q_lb, **q_fix), fill=-np.inf),
                model.pack('q', dict(q_ub, **q_fix), fill=np.inf)]
    problem = yaipopt.Problem(q_bounds, merit, grad,
                            hess=hess, hess_inds=hess_inds)
    problem.num_option(b'obj_scaling_factor', -1)
    (qopt, solinfo) = problem.solve(q0)
    
    return problem, qopt, solinfo, model, q0, x0, Px0


if __name__ == '__main__':
    [t, y] = load_data()
    [problem, qopt, solinfo, model, q0, x0, Px0] = pem(t, y)
