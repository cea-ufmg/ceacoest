"""Output Error Method for the ATTAS aircraft short-period mode estimation."""


import importlib
import os

import numpy as np
import scipy.interpolate
import scipy.io
import sympy
import sym2num.model

from ceacoest import optim, col, oem
from ceacoest.modelling import genoptim, symoptim, symcol, symoem, symstats


# Reload modules for testing
for m in (optim, col, oem, genoptim, symoptim, symcol, symoem, symstats):
    importlib.reload(m)


class SymbolicAttasShortPeriod(symoem.Model):
    """Symbolic linear short period model."""
    
    collocation_order = 2
    
    def __init__(self):
        v = self.Variables(
            x=['q', 'alpha'],
            y=['q_meas', 'alpha_meas'],
            u=['de'],
            p=['Z0', 'Zalpha', 'Zq', 'Zde', 'M0', 'Malpha', 'Mq', 'Mde',
               'alpha_meas_std', 'q_meas_std']
        )
        super().__init__(v)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        qdot = s.M0 + s.Malpha*s.alpha + s.Mq*s.q + s.Mde*s.de
        alphadot = s.Z0 + s.Zalpha*s.alpha + (s.Zq + 1)*s.q + s.Zde*s.de
        return [qdot, alphadot]
    
    @sym2num.model.collect_symbols
    def L(self, y, x, u, p, *, s):
        """Measurement log likelihood."""
        return (symstats.normal_logpdf1(s.alpha_meas, s.alpha, s.alpha_meas_std)
                + symstats.normal_logpdf1(s.q_meas, s.q, s.q_meas_std))


def load_data():
    d2r = np.pi / 180
    module_dir = os.path.dirname(__file__)
    data_file_path = os.path.join(module_dir, 'data', 'fAttasElv1.mat')
    data = scipy.io.loadmat(data_file_path)['fAttasElv1'][30:-30]
    t = data[:, 0] - data[0, 0]
    u = data[:, [21]] * d2r
    y = data[:, [7, 12]] * d2r
    return t, u, y


if __name__ == '__main__':
    # Compile and instantiate model
    symmodel = SymbolicAttasShortPeriod()
    GeneratedAttasShortPeriod = symmodel.compile_class()
    model = GeneratedAttasShortPeriod()
    
    # Load experiment data
    t, u, y = load_data()
    ufun = scipy.interpolate.interp1d(t, u, axis=0)
    
    # Create OEM problem
    problem = oem.Problem(model, t, y, ufun)
    tc = problem.tc
    
    # Define problem bounds
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    dec_L[-2] = 0.0001 # alpha_meas_std
    dec_L[-1] = 0.00025 # q_meas_std
    
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    
    # Define initial guess for decision variables
    dec0 = np.zeros(problem.ndec)
    dec0[-2] = 0.001 # alpha_meas_std
    dec0[-1] = 0.0025 # q_meas_std
    
    # Define problem scaling
    dec_scale = np.ones(problem.ndec)
    constr_scale = np.ones(problem.ncons)
    obj_scale = -1.0

    # Run ipopt
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 10.0)
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)

    # Unpack the solution
    opt = problem.variables(decopt)
    xopt = opt['x']
    popt = opt['p']

    # Show results
    from matplotlib import pylab
    pylab.figure(1).clear()
    pylab.plot(tc, xopt[:, 0], '-', t, y[:, 0], '.')

    pylab.figure(2).clear()
    pylab.plot(tc, xopt[:, 1], '-', t, y[:, 1], '.')
    pylab.show()
