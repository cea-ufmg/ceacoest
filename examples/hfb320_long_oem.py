"""Output error method estimation of the longitudinal parameters of an HFB-320.

This example corresponds to the test case #4 of the 4th chapter of the book
Flight Vehicle System Identification: A Time-Domain Methodology, Second Edition
by Ravindra V. Jategaonkar, Senior Scientist, Institute of Flight Systems, DLR.

It uses flight test data obtained by the DLR that accompanies the book's
supporting materials (not provided here).
"""


import functools

import numpy as np
import sympy
import sym2num.model
import sym2num.utils
from sym2num import var
from sympy import cos, sin

from ceacoest import oem, optim
from ceacoest.modelling import symcol, symoem, symstats


import imp; [imp.reload(m) for m in [symcol, symoem, oem]]


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
            'V_meas_std', 'alpha_meas_std', 'theta_meas_std', 'q_meas_std',
            'qdot_meas_std', 'ax_meas_std', 'az_meas_std'
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
    def L(self, y, xm, um, p, *, s):
        """Measurement log likelihood."""
        qbar = 0.5 * s.rho * s.V ** 2
        qhat = s.cbarH * s.q / s.V0

        CD = s.CD0 + s.CDV * s.V / s.V0 + s.CDa * s.alpha
        CL = s.CL0 + s.CLV * s.V / s.V0 + s.CLa * s.alpha
        Cm = s.Cm0 + s.CmV*s.V/s.V0 + s.Cma*s.alpha + s.Cmq*qhat + s.Cmde*s.de
        
        salpha =  sin(s.alpha);
        calpha =  cos(s.alpha);
        CX =  CL*salpha - CD*calpha
        CZ = -CL*calpha - CD*salpha
        
        qdot = s.ScbyIy*qbar*Cm + s.T*s.FEIYLT
        ax = s.Sbym*qbar*CX + s.T*cos(s.sigmaT)/s.mass
        az = s.Sbym*qbar*CZ - s.T*sin(s.sigmaT)/s.mass
        return sympy.Array(
            symstats.normal_logpdf1(s.V_meas, s.V, s.V_meas_std)
            + symstats.normal_logpdf1(s.alpha_meas, s.alpha, s.alpha_meas_std)
            + symstats.normal_logpdf1(s.theta_meas, s.theta, s.theta_meas_std)
            + symstats.normal_logpdf1(s.q_meas, s.q, s.q_meas_std)
            + symstats.normal_logpdf1(s.qdot_meas, qdot, s.qdot_meas_std)
            + symstats.normal_logpdf1(s.ax_meas, ax, s.ax_meas_std)
            + symstats.normal_logpdf1(s.az_meas, az, s.az_meas_std)
        )


if __name__ == '__main__':
    symb_mdl = HFB320Long()
    GeneratedHFB320Long = sym2num.model.compile_class(symb_mdl)
    model = GeneratedHFB320Long()
