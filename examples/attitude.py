"""Attitude reconstruction example."""


import yaipopt
import numpy as np
import sympy
import sym2num
from numpy import ma
from scipy import integrate, interpolate, signal, stats

from ceacoest import kalman, sde, utils


class SymbolicModel(sde.SymbolicModel):
    """Symbolic SDE model."""
    
    var_names = {'t', 'x', 'y', 'q', 'c'}
    """Name of model variables."""
    
    function_names = {'f', 'g', 'h', 'R', 'x0', 'Px0'}
    """Name of the model functions."""
    
    t = 't'
    """Time variable."""
    
    x = ['ax', 'ay', 'az', 'p', 'q', 'r', 'phi', 'theta', 'psi']
    """State vector."""
    
    y = ['ax_meas', 'ay_meas', 'az_meas', 'p_meas', 'q_meas', 'r_meas',
         'magx_meas', 'magy_meas', 'magz_meas']
    """Measurement vector."""
    
    q = ['magex', 'magez', 'maghx', 'maghy', 'maghz',
         'pbias', 'qbias', 'rbias']
    """Parameter vector."""
    
    c = ['accel_png', 'omega_png', 'accel_mng', 'omega_mng', 'mag_mng',
         'g0', 'ax0', 'ay0', 'az0', 'p0', 'q0', 'r0', 'phi0', 'theta0', 'psi0',
         'ax0_std', 'ay0_std', 'az0_std', 'p0_std', 'q0_std', 'r0_std',
         'phi0_std', 'theta0_std', 'psi0_std']
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
            theta=cphi*s.q - sphi*s.r,
            psi=sphi/cth*s.q + cphi/cth*s.r,
        )
        return self.pack('x', derivs)
    
    def g(self, t, x, q, c):
        """Diffusion matrix."""
        s = self.symbols(t=t, x=x, q=q, c=c)
        g = np.zeros((x.size, 6), object)
        g[[0, 1, 2], [0, 1, 2]] = s.accel_png
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
        magh = [s.maghx, s.maghy, s.maghz]
        mag = e2b.dot([s.magex, 0, s.magez]) + magh
        gb = np.dot(e2b, [0, 0, s.g0])
        meas = dict(
            magx_meas=mag[0], magy_meas=mag[1], magz_meas=mag[2],
            p_meas=s.p + s.pbias, 
            q_meas=s.q + s.qbias, 
            r_meas=s.r + s.rbias,
            ax_meas=s.ax - gb[0],
            ay_meas=s.ay - gb[1],
            az_meas=s.az - gb[2],
        )
        return self.pack('y', meas)
    
    def R(self, q, c):
        """Measurement function."""
        s = self.symbols(q=q, c=c)
        std = dict(
            ax_meas=s.accel_mng, ay_meas=s.accel_mng, az_meas=s.accel_mng, 
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

    def Px0(self, q, c):
        """Initial state covariance."""
        s = self.symbols(q=q, c=c)
        x0_std = {
            'ax': s.ax0_std, 'ay': s.ay0_std, 'az': s.az0_std,
            'p': s.p0_std, 'q': s.q0_std, 'r': s.r0_std,
            'phi': s.phi0_std, 'theta': s.theta0_std, 'psi': s.psi0_std,
        }
        return np.diag(self.pack('x', x0_std)**2)
    


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
                   ('dx0_dq', 'x0', 'q'), ('d2x0_dq2', 'dx0_dq', 'q'),
                   ('dPx0_dq', 'Px0', 'q'), ('d2Px0_dq2', 'dPx0_dq', 'q')]
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


def sim():
    np.random.seed(0)

    # Generate the time vector
    dt = 0.05
    N = int(30 / dt)
    k = np.arange(N)
    t = k * dt
    
    # Generate the accelerations and angular velocities
    [b, a] = signal.butter(2, 0.05)
    omega_png = 0.01
    accel_png = 0.1
    omega = signal.lfilter(b, a, omega_png*np.cumsum(np.random.randn(3, N), -1))
    accel = signal.lfilter(b, a, accel_png*np.cumsum(np.random.randn(3, N), -1))

    # Integrate the angular velocities to obtain the attitude
    omega_int = interpolate.interp1d(t, omega, fill_value='extrapolate')
    def odefun(angles, t):
        [phi, theta, psi] = angles
        [p, q, r] = omega_int(t)
        cphi = np.cos(phi)
        sphi = np.sin(phi)
        cth = np.cos(theta)
        sth = np.sin(theta)
        tth = np.tan(theta)
        phidot = p + tth*(sphi*q + cphi*r)
        thetadot = cphi*q - sphi*r
        psidot = sphi/cth*q + cphi/cth*r
        return [phidot, thetadot, psidot]
    angles = integrate.odeint(odefun, [0,0,0], t)

    # Generate the nominal model
    nominal = dict(
        g0=9.80665, accel_png=accel_png, omega_png=omega_png,
        accel_mng=0.5, omega_mng=0.01, mag_mng=1e-6,
        magex=18.982e-6, magez=-13.6305e-6,
        maghx=1e-6, maghy=-1e-6, maghz=0,
        pbias=0.02, qbias=-0.02, rbias=0,
        ax0_std=1, ay0_std=1, az0_std=1,
        p0_std=0.1, q0_std=0.1, r0_std=0.1,
        phi0_std=0.1, theta0_std=0.1, psi0_std=0.1,
    )
    c = GeneratedDTModel.pack('c', nominal)
    q = GeneratedDTModel.pack('q', nominal)
    model = GeneratedDTModel({'q': q, 'c': c, 'dt': dt}, {'t': t})
    
    x = np.vstack((accel, omega, angles.T)).T
    v = np.random.multivariate_normal(np.zeros(model.ny), model.R(), N)
    y = ma.asarray(model.h(k, x) + v)
    y[np.arange(N) % 2 != 0] = ma.masked
    
    return model, t, x, y, q


def pem(model, q0):
    def merit(q, new=None):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedKalmanFilter(mq)
        return kf.pem_merit(y)
    
    def grad(q, new=None):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedKalmanFilter(mq)
        return kf.pem_gradient(y)
    
    hess_inds = np.tril_indices(model.nq)
    def hess(q, new_q=1, obj_factor=1, lmult=1, new_lmult=1):
        mq = model.parametrize(q=q)
        kf = kalman.DTUnscentedKalmanFilter(mq)
        return obj_factor * kf.pem_hessian(y)[hess_inds]
    
    q_lb = dict(accel_png=0, omega_png=0,
                accel_mng=0, omega_mng=0, mag_mng=0)
    q_ub = dict()
    q_fix = dict()
    q_bounds = [model.pack('q', dict(q_lb, **q_fix), fill=-np.inf),
                model.pack('q', dict(q_ub, **q_fix), fill=np.inf)]
    problem = yaipopt.Problem(q_bounds, merit, grad,
                              hess=hess, hess_inds=hess_inds)
    problem.num_option(b'obj_scaling_factor', -1)
    (qopt, solinfo) = problem.solve(q0)
    
    return problem, qopt, solinfo


if __name__ == '__main__':
    guess = dict(magex=18.982e-6, magez=-13.6305e-6)
    q0 = GeneratedDTModel.pack('q', guess)
    
    model, t, x, y, q = sim()
    problem, qopt, solinfo = pem(model, q0)
    
    mopt = model.parametrize(q=qopt)
    kfopt = kalman.DTUnscentedKalmanFilter(mopt)
    [xs, Pxs] = kfopt.smooth(y)
