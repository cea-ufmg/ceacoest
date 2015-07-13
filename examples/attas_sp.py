"""Output Error Method for the ATTAS aircraft short-period mode estimation."""


import os.path

import ipopt
import numpy as np
import scipy.io
import sympy
import sym2num
from numpy import ma
from scipy import interpolate

from ceacoest import kalman, oem, symstats, utils


class SymbolicModel(sym2num.SymbolicModel):
    """Symbolic short-period-mode linear model."""
    
    var_names = {'t', 'x', 'y', 'u', 'q', 'c'}
    """Name of model variables."""
    
    function_names = {'f', 'L'}
    """Name of the model functions."""

    derivatives = [('df_dx', 'f', 'x'), ('df_dq', 'f', 'q'),
                   ('d2f_dx2', 'df_dx',  'x'), 
                   ('d2f_dx_dq', 'df_dx', 'q'),
                   ('d2f_dq2', 'df_dq',  'q'),
                   ('dL_dx', 'L', 'x'), ('dL_dq', 'L', 'q'),
                   ('d2L_dx2', 'dL_dx',  'x'), 
                   ('d2L_dx_dq', 'dL_dx', 'q'),
                   ('d2L_dq2', 'dL_dq',  'q'),]
    """List of the model function derivatives to calculate / generate."""
    
    t = 't'
    """Time variable."""
    
    x = ['q', 'alpha']
    """State vector."""
    
    y = ['q_meas', 'alpha_meas']
    """Measurement vector."""

    u = ['de']
    """Exogenous input vector."""
    
    q = ['Z0', 'Zalpha', 'Zq', 'Zde', 'M0', 'Malpha', 'Mq', 'Mde',
         'alpha_meas_std', 'q_meas_std']
    """Unknown parameter vector."""
    
    c = []
    """Constants vector."""

    meta = sym2num.ParametrizedModel.meta
    """Generated Model metaclass."""
    
    generated_name = "GeneratedModel"
    """Name of generated class."""
    
    def f(self, t, x, q, u, c):
        """ODE vector field."""
        s = self.symbols(t=t, x=x, q=q, u=u, c=c)
        derivs = {}
        derivs['alpha'] = s.Z0 + s.Zalpha*s.alpha + s.Zq*s.q + s.Zde*s.de
        derivs['q'] = s.M0 + s.Malpha*s.alpha + s.Mq*s.q + s.Mde*s.de
        return self.pack('x', derivs)
    
    def L(self, y, t, x, q, u, c):
        """Measurement log likelihood."""
        s = self.symbols(y=y, t=t, x=x, q=q, u=u, c=c)
        return (
            symstats.normal_logpdf1(s.alpha_meas, s.alpha, s.alpha_meas_std) +
            symstats.normal_logpdf1(s.q_meas, s.q, s.q_meas_std)
        )


sym_model = SymbolicModel()
printer = sym2num.ScipyPrinter()
GeneratedModel = sym2num.class_obj(sym_model, printer)
GeneratedModel.ny = len(sym_model.y)
GeneratedModel.nx = len(sym_model.x)
GeneratedModel.nq = len(sym_model.q)


def lininterp(x):
    N = np.size(x, axis=0)
    xi = np.zeros((2*N - 1,) + np.shape(x)[1:])
    xi[::2] = x
    xi[1::2] = 0.5 * (x[1:] + x[:-1])
    return xi


def load_data():
    d2r = np.pi / 180
    module_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(module_dir, 'data', 'fAttasElv1.mat')
    data = scipy.io.loadmat(data_file_path)['fAttasElv1']
    t = lininterp(data[:, 0])
    u = interpolate.interp1d(data[:, 0], data[:, [21]] * d2r, axis=0)
    y = ma.empty((t.size, 2))
    y[...] = ma.masked
    y[::2] = data[:, [12, 7]] * d2r
    return t, u, y


if __name__ == '__main__':
    [t, u, y] = load_data()
    
    given = dict(alpha_meas_std=2e-3, q_meas_std=2e-3)
    c = GeneratedModel.pack('c', given)
    params = dict(c=c)
    model = GeneratedModel(params)
    est = oem.CTEstimator(model, y, t, u, order=4)

    x0 = interpolate.interp1d(t[::2], y[::2], axis=0)(est.tc)
    q0 = GeneratedModel.pack('q', given)
    d0 = est.pack_decision(x0, q0)
    
