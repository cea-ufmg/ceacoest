"""Output error method estimation of the longitudinal parameters of an HFB-320.

This example corresponds to the test case #4 of the 4th chapter of the book
Flight Vehicle System Identification: A Time-Domain Methodology, Second Edition
by Ravindra V. Jategaonkar, Senior Scientist, Institute of Flight Systems, DLR.

It uses flight test data obtained by the DLR that accompanies the book's
supporting materials (not provided here).
"""


import functools
import os.path

import numpy as np
import sympy
import sym2num.model
import sym2num.utils
from scipy import integrate, interpolate
from sym2num import var
from sympy import cos, sin

from ceacoest import oem, optim
from ceacoest.modelling import symcol, symoem, symstats


@symoem.collocate(order=3)
class HFB320Long:
    """Symbolic HFB-320 aircraft nonlinear longitudinal model."""
    
    @sym2num.utils.classproperty
    @functools.lru_cache()
    def variables(cls):
        """Model variables definition."""
        params = [
            'CD0', 'CDV', 'CDa', 'CL0', 'CLV', 'CLa',
            'Cm0', 'CmV', 'Cma', 'Cmq', 'Cmde', 
            'q_bias', 'qdot_bias', 'ax_bias', 'az_bias',
            'V_meas_std', 'alpha_meas_std', 'theta_meas_std', 'q_meas_std',
            'qdot_meas_std', 'ax_meas_std', 'az_meas_std', 
        ]
        consts = [
            'g0', 'Sbym', 'ScbyIy', 'FEIYLT', 'V0', 'mass', 'sigmaT', 'rho',
            'cbarH'
        ]
        y = ['V_meas', 'alpha_meas', 'theta_meas', 'q_meas', 
             'qdot_meas', 'ax_meas', 'az_meas']

        vars = [var.SymbolObject('self', var.SymbolArray('consts', consts)),
                var.SymbolArray('x', ['V', 'alpha', 'theta', 'q']),
                var.SymbolArray('y', y),
                var.SymbolArray('u', ['de', 'T']),
                var.SymbolArray('p', params)]
        return var.make_dict(vars)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        qbar = 0.5 * s.rho * s.V ** 2
        gamma = s.theta - s.alpha
        qhat = s.cbarH * s.q / s.V0
        
        CD = s.CD0 + s.CDV * s.V / s.V0 + s.CDa * s.alpha
        CL = s.CL0 + s.CLV * s.V / s.V0 + s.CLa * s.alpha
        Cm = s.Cm0 + s.CmV*s.V/s.V0 + s.Cma*s.alpha + s.Cmq*qhat + s.Cmde*s.de
        
        Vd = (-s.Sbym*qbar*CD + s.T*cos(s.alpha + s.sigmaT)/s.mass 
              - s.g0*sin(gamma))
        alphad = (-s.Sbym*qbar/s.V*CL - s.T*sin(s.alpha + s.sigmaT)/(s.mass*s.V)
                  + s.g0*cos(gamma)/s.V + s.q)
        qd = s.ScbyIy*qbar*Cm + s.T*s.FEIYLT
        return sympy.Array([Vd, alphad, s.q, qd])
    
    @sym2num.model.collect_symbols
    def L(self, y, x, u, p, *, s):
        """Measurement log likelihood."""
        qbar = 0.5 * s.rho * s.V ** 2
        qhat = s.cbarH * s.q / s.V0

        CD = s.CD0 + s.CDV * s.V / s.V0 + s.CDa * s.alpha
        CL = s.CL0 + s.CLV * s.V / s.V0 + s.CLa * s.alpha
        Cm = s.Cm0 + s.CmV*s.V/s.V0 + s.Cma*s.alpha + s.Cmq*qhat + s.Cmde*s.de
        
        salpha =  sin(s.alpha)
        calpha =  cos(s.alpha)
        CX =  CL*salpha - CD*calpha
        CZ = -CL*calpha - CD*salpha
        
        qdot = s.ScbyIy*qbar*Cm + s.T*s.FEIYLT + s.qdot_bias
        ax = s.Sbym*qbar*CX + s.T*cos(s.sigmaT)/s.mass + s.ax_bias
        az = s.Sbym*qbar*CZ - s.T*sin(s.sigmaT)/s.mass + s.az_bias
        return sympy.Array(
            symstats.normal_logpdf1(s.V_meas, s.V, s.V_meas_std)
            + symstats.normal_logpdf1(s.alpha_meas, s.alpha, s.alpha_meas_std)
            + symstats.normal_logpdf1(s.theta_meas, s.theta, s.theta_meas_std)
            + symstats.normal_logpdf1(s.q_meas, s.q + s.q_bias, s.q_meas_std)
            + symstats.normal_logpdf1(s.qdot_meas, qdot, s.qdot_meas_std)
            + symstats.normal_logpdf1(s.ax_meas, ax, s.ax_meas_std)
            + symstats.normal_logpdf1(s.az_meas, az, s.az_meas_std)
        )


if __name__ == '__main__':
    given = {'g0': 9.80665, 'Sbym': 4.0280e-3, 'ScbyIy': 8.0027e-4, 
             'FEIYLT': -7.0153e-6, 'V0': 104.67, 'mass':7472, 'sigmaT':0.0524,
             'rho': 0.7920, 'cbarH': 1.215}
    lower = {'V': 2, 'V_meas_std': 1e-3, 'alpha_meas_std': 1e-4,
             'theta_meas_std': 1e-4, 'q_meas_std': 1e-4, 'qdot_meas_std': 1e-4,
             'ax_meas_std': 1e-4, 'az_meas_std': 1e-4}
    
    # Compile and instantiate model
    symb_mdl = HFB320Long()
    GeneratedHFB320Long = sym2num.model.compile_class(symb_mdl)
    model = GeneratedHFB320Long(**given)
    
    # Load experiment data
    dirname = os.path.dirname(__file__)
    data = np.loadtxt(os.path.join(dirname, 'data', 'hfb320_1_10.asc'))
    Ts = 0.1
    t = np.arange(len(data)) * Ts
    y = data[:, 4:11]
    u = interpolate.interp1d(t, data[:, [1,3]], axis=0)

    # Create OEM problem
    problem = oem.Problem(model, t, y, u)
    tc = problem.tc
    
    # Set bounds
    constr_bounds = np.zeros((2, problem.ncons))
    dec_L, dec_U = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    for k,v in lower.items():
        problem.set_decision_item(k, v, dec_L)    
    
    # Set initial guess
    x0 = interpolate.interp1d(t, y.T[:4])(tc).T
    p0 = np.zeros(model.np)
    p0[-model.ny:] = 1 # guess for measurement standard deviations
    dec0 = np.zeros(problem.ndec)
    problem.set_decision('x', x0, dec0)
    problem.set_decision('p', p0, dec0)
    
    dec_scale = np.ones(problem.ndec)
    problem.set_decision_item('V', 1e-2, dec_scale)
    problem.set_decision_item('alpha', 20, dec_scale)
    problem.set_decision_item('q', 30, dec_scale)
    problem.set_decision_item('theta', 20, dec_scale)
    problem.set_decision_item('V_meas_std', 1/0.2, dec_scale)
    problem.set_decision_item('alpha_meas_std', 1/0.03, dec_scale)
    problem.set_decision_item('theta_meas_std', 1/0.002, dec_scale)
    problem.set_decision_item('q_meas_std', 1/0.001, dec_scale)
    problem.set_decision_item('qdot_meas_std', 1/0.025, dec_scale)
    problem.set_decision_item('ax_meas_std', 1/0.03, dec_scale)
    problem.set_decision_item('az_meas_std', 1/0.03, dec_scale)
    
    constr_scale = np.ones(problem.ncons)
    problem.set_defect_scale('V', 1e-2, constr_scale)
    problem.set_defect_scale('alpha', 20, constr_scale)
    problem.set_defect_scale('q', 30, constr_scale)
    problem.set_defect_scale('theta', 20, constr_scale)
    
    with problem.ipopt((dec_L, dec_U), constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(-1, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    popt = opt['p']
