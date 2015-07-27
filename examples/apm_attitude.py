"""Attitude reconstruction of an ArduPilot Mega."""


import os
import re

import yaipopt
import numpy as np
import sympy
import sym2num
from numpy import ma
from scipy import interpolate

from ceacoest import kalman, sde, utils


class SymbolicModel(sde.SymbolicModel):
    """Symbolic SDE model."""
    
    var_names = {'t', 'x', 'y', 'q', 'c'}
    """Name of model variables."""
    
    function_names = {'f', 'g', 'h', 'R'}
    """Name of the model functions."""
    
    t = 't'
    """Time variable."""
    
    x = ['ax', 'ay', 'az', 'p', 'q', 'r', 'q0', 'q1', 'q2', 'q3']
    """State vector."""
    
    y = ['ax_meas', 'ay_meas', 'az_meas', 'p_meas', 'q_meas', 'r_meas',
         'magx_meas', 'magy_meas', 'magz_meas']
    """Measurement vector."""
    
    q = ['acc_png', 'omega_png', 'acc_mng', 'omega_mng', 'mag_mng',
         'magex', 'magez', 'maghx', 'maghy', 'maghz']
    """Parameter vector."""
    
    c = ['quat_renorm_gain', 'g0']
    """Constants vector."""
    
    def f(self, t, x, q, c):
        """Drift function."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        renorm = s.quat_renorm_gain*(1 - s.q0**2 - s.q1**2 - s.q2**2 - s.q3**2)
        derivs = dict(
            p=0, q=0, r=0, ax=0, ay=0, az=0,
            q0=-0.5*(s.q1*s.p + s.q2*s.q + s.q3*s.r) + renorm*s.q0,
            q1=-0.5*(-s.q0*s.p - s.q2*s.r + s.q3*s.q) + renorm*s.q1,
            q2=-0.5*(-s.q0*s.q + s.q1*s.r - s.q3*s.p) + renorm*s.q2,
            q3=-0.5*(-s.q0*s.r - s.q1*s.q + s.q2*s.p) + renorm*s.q3,
        )
        return self.pack('x', derivs)
    
    def g(self, t, x, q, c):
        """Diffusion matrix."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        g = np.zeros((x.size, 6), object)
        g[[0, 1, 2], [0, 1, 2]] = s.acc_png
        g[[3, 4, 5], [3, 4, 5]] = s.omega_png
        return g
    
    def h(self, t, x, q, c):
        """Measurement function."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        q0 = s.q0
        q1 = s.q1
        q2 = s.q2
        q3 = s.q3
        e2b = np.array(
            [[q0**2+q1**2-q2**2-q3**2, 2*(q1*q2 + q0*q3), 2*(q1*q3 - q0*q2)],
             [2*(q1*q2 - q0*q3), q0**2-q1**2+q2**2-q3**2, 2*(q2*q3 + q0*q1)],
             [2*(q1*q3 + q0*q2), 2*(q2*q3 - q0*q1), q0**2-q1**2-q2**2+q3**2]],
            dtype=object
        )
        mag = np.dot(e2b, [s.magex, 0, s.magez]) + [s.maghx, s.maghy, s.maghz]
        gb = np.dot(e2b, [0, 0, s.g0])
        meas = dict(
            p_meas=s.p, q_meas=s.q, r_meas=s.r,
            magx_meas=mag[0], magy_meas=mag[1], magz_meas=mag[2],
            ax_meas=s.ax - gb[0],
            ay_meas=s.ay - gb[1],
            az_meas=s.az - gb[2],
        )
        return self.pack('y', meas)
    
    def R(self, q, c):
        """Measurement function."""
        s = self.symbols(q=q, c=c)
        std = dict(
            ax_meas=s.acc_mng, ay_meas=s.acc_mng, az_meas=s.acc_mng, 
            p_meas=s.omega_mng, q_meas=s.omega_mng, r_meas=s.omega_mng,
            magx_meas=s.mag_mng, magy_meas=s.mag_mng, magz_meas=s.mag_mng,
        )
        return np.diag(self.pack('y', std) ** 2)


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
    # Read the log file
    module_dir = os.path.dirname(__file__)
    filepath = os.path.join(module_dir, 'data', 'apm.log')
    lines = open(filepath).read().splitlines()

    # Parse the data
    data = dict(MAG=[], IMU=[], ATT=[])
    for line in lines:
        msgid, *fields = re.split(',\s*', line)
        if msgid in data:
            data[msgid].append([float(f) for f in fields])
    data = {key: np.asarray(val) for key, val in data.items()}
    imu = data['IMU']
    mag = data['MAG']
    
    
    # Build the output array
    t = np.sort(np.hstack((imu[:, 0], mag[:, 0])))
    imu_inds = np.array([tk in imu[:, 0] for tk in t])
    mag_inds = np.array([tk in mag[:, 0] for tk in t])
    y = ma.masked_all((t.size, GeneratedDTModel.ny))
    y[imu_inds, :6] = imu[:, [4, 5, 6, 1, 2, 3]]
    y[mag_inds, 6:] = mag[:, [1, 2, 3]]
    t *= 1e-3
    
    # Select the experiment interval
    range_ = np.s_[900:1200]#np.s_[900:1800]
    t = t[range_]
    y = y[range_]
    assert np.unique(t).size == t.size
    return t, y, data


def pem(t, y, data):
    # Instantiate the model
    given = dict(
        g0=9.81, quat_renorm_gain=4,
        acc_png=0.01, omega_png=0.0001,
        acc_mng=0.03, omega_mng=8e-3, mag_mng=1.6,
        magex=117, magez=0, maghx=0, maghy=0, maghz=0,
    )
    dt = np.diff(t)
    q0 = GeneratedDTModel.pack('q', given)
    c = GeneratedDTModel.pack('c', given)
    params = dict(q=q0, c=c)
    sampled = dict(dt=dt, t=t)
    model = GeneratedDTModel(params, sampled)

    # Build the initial state and covariance
    att = data['ATT']
    ang = interpolate.interp1d(att[:, 0]*1e-3, att[:, [2,4,6]], axis=0)
    phi, theta, psi = ang(t[0]) * np.pi / 180
    x0 = np.zeros(GeneratedDTModel.nx)
    x0[-4] = 1
    Px0 = np.diag(np.repeat([1, 1e-3, 0.1], (3,3,4)) ** 2)
    
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
    q_fix = dict(acc_mng=0.03, omega_mng=8e-3, mag_mng=1.6)
    q_bounds = [model.pack('q', dict(q_lb, **q_fix), fill=-np.inf),
                model.pack('q', dict(q_ub, **q_fix), fill=np.inf)]
    problem = yaipopt.Problem(q_bounds, merit, grad,
                            hess=hess, hess_inds=hess_inds)
    problem.num_option(b'obj_scaling_factor', -1)
    (qopt, solinfo) = problem.solve(q0)
    
    return problem, qopt, solinfo, model, q0, x0, Px0


if __name__ == '__main__':
    [t, y, data] = load_data()
    
    
