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
    
    function_names = {'f', 'g', 'h', 'R', 'x0'}
    """Name of the model functions."""
    
    t = 't'
    """Time variable."""
    
    x = ['ax', 'ay', 'az', 'p', 'q', 'r', 'phi', 'theta', 'psi']
    """State vector."""
    
    y = ['ax_meas', 'ay_meas', 'az_meas', 'p_meas', 'q_meas', 'r_meas',
         'magx_meas', 'magy_meas', 'magz_meas']
    """Measurement vector."""
    
    q = ['acc_png', 'omega_png', 'acc_mng', 'omega_mng', 'mag_mng',
         'magex', 'magez', 'maghx', 'maghy', 'maghz',
         'ax0', 'ay0', 'az0', 'p0', 'q0', 'r0', 'phi0', 'theta0', 'psi0']
    """Parameter vector."""
    
    c = ['g0']
    """Constants vector."""
    
    def f(self, t, x, q, c):
        """Drift function."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        cphi = sympy.cos(s.phi)
        sphi = sympy.sin(s.phi)
        cth = sympy.cos(s.theta)
        sth = sympy.sin(s.theta)
        tth = sympy.tan(s.theta)
        derivs = dict(
            p=0, q=0, r=0, ax=0, ay=0, az=0,
            phi=s.p + tth*(sphi*s.q + cphi*s.r),
            theta=cphi*s.q -sphi*s.r,
            psi=sphi/cth*s.q + cphi/cth*s.r,
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
        cphi = sympy.cos(s.phi)
        sphi = sympy.sin(s.phi)
        cth = sympy.cos(s.theta)
        sth = sympy.sin(s.theta)
        cpsi = sympy.cos(s.psi)
        spsi = sympy.sin(s.psi)
        e2b = np.array(
            [[cth*cpsi, cth*spsi, -sth],
             [-cphi*spsi + sphi*sth*cpsi, cphi*cpsi + sphi*sth*spsi, sphi*cth],
             [sphi*spsi + cphi*sth*cpsi, -sphi*cpsi + cphi*sth*spsi, cphi*cth]],
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
    
    def x0(self, q, c):
        """Initial state."""
        s = self.symbols(q=q, c=c)
        x0 = {
            'ax': s.ax0, 'ay': s.ay0, 'az': s.az0,
            'p': s.p0, 'q': s.q0, 'r': s.r0,
            'phi': s.phi0, 'theta': s.theta0, 'psi': s.psi0,
        }
        return self.pack('x', x0)


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
                   ('dR_dq', 'R', 'q'), ('d2R_dq2', 'dR_dq', 'q'),
                   ('dx0_dq', 'x0', 'q'), ('d2x0_dq2', 'dx0_dq', 'q')]
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
    range_ = np.s_[905:1200]#np.s_[900:1800]
    t = t[range_]
    y = y[range_]
    assert np.unique(t).size == t.size
    return t, y, data


def pem(t, y, data):
    # Build the initial state and covariance
    imu = data['IMU']
    att = data['ATT']
    d2r = np.pi / 180
    omega0 = interpolate.interp1d(imu[:, 0]*1e-3, imu[:,[1,2,3]],axis=0)(t[0])
    ang0 = interpolate.interp1d(att[:, 0]*1e-3, att[:,[2,4,6]]*d2r,axis=0)(t[0])
    Px0 = np.diag(np.repeat(1e-3 ** 2, GeneratedDTModel.nx))
    
    # Instantiate the model
    given = dict(
        g0=9.81,
        acc_png=0.05, omega_png=0.05,
        acc_mng=0.03, omega_mng=8e-3, mag_mng=2,
        magex=206, magez=147, maghx=-50, maghy=-50, maghz=-190,
        p0=omega0[0], q0=omega0[1], r0=omega0[2],
        phi0=ang0[0], theta0=ang0[1], psi0=ang0[2]+0.3941,
    )
    dt = np.diff(t)
    q0 = GeneratedDTModel.pack('q', given)
    c = GeneratedDTModel.pack('c', given)
    params = dict(q=q0, c=c)
    sampled = dict(dt=dt, t=t)
    model = GeneratedDTModel(params, sampled)
    
    def merit(q, new=None):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedKalmanFilter(mq, Px=Px0)
        return kf.pem_merit(y)
    
    def grad(q, new=None):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedKalmanFilter(mq, Px=Px0)
        return kf.pem_gradient(y)
    
    hess_inds = np.tril_indices(model.nq)
    def hess(q, new_q=1, obj_factor=1, lmult=1, new_lmult=1):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedKalmanFilter(mq, Px=Px0)
        return obj_factor * kf.pem_hessian(y)[hess_inds]
    
    q_lb = dict(acc_png=0, omega_png=0,
                acc_mng=0, omega_mng=0, mag_mng=0,
                ax0=-5, ay0=-5, az0=-5,
                phi0=-np.pi, psi0=0, theta0=-np.pi/2)
    q_ub = dict(ax0=5, ay0=5, az0=5,
                phi0=np.pi, psi0=2*np.pi, theta0=np.pi/2)
    q_fix = dict(acc_png=0.05, omega_png=0.05,
                 acc_mng=0.03, omega_mng=8e-3, mag_mng=2)
    q_bounds = [model.pack('q', dict(q_lb, **q_fix), fill=-np.inf),
                model.pack('q', dict(q_ub, **q_fix), fill=np.inf)]
    problem = yaipopt.Problem(q_bounds, merit, grad,
                              hess=hess, hess_inds=hess_inds)
    problem.num_option(b'obj_scaling_factor', -1)
    (qopt, solinfo) = problem.solve(q0)
    
    return problem, qopt, solinfo, model, q0, Px0


if __name__ == '__main__':
    [t, y, data] = load_data()
    
