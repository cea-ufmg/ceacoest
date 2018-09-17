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
from ceacoest.modelling import symstats


class HFB320Long:
    """Symbolic HFB-320 aircraft nonlinear longitudinal model."""
    
    @sym2num.utils.classproperty
    @functools.lru_cache()
    def variables(cls):
        """Model variables definition."""
        param_names = [
            'CD0', 'CDV', 'CDa', 'CL0', 'CLV', 'CLa',
            'Cm0', 'CmV', 'Cma', 'Cmq', 'Cmde', 
            'V_meas_std', 'alpha_meas_std', 'theta_meas_std', 'q_meas_std'
        ]
        consts_names = ['g0', 'Sbym', 'ScbyIy', 'FEIYLT', 'V0', 'mass', 'sigmaT', 'rho']

        vars = [var.SymbolObject( 'self', var.SymbolArray('consts', consts_names)),
                var.SymbolArray('x', ['V', 'alpha', 'theta', 'q']),
                var.SymbolArray('u', ['de', 'T']),
                var.SymbolArray('p', param_names)]
        return var.make_dict(vars)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        qbar = 0.5 * s.rho * s.V ** 2
        gamma = s.theta - s.alpha
        qhat = 1.215 * s.q / s.V0
        
        CD = s.CD0 + s.CDV * s.V / s.V0 + s.CDa * s.alpha
        CL = s.CL0 + s.CLV * s.V / s.V0 + s.CLa * s.alpha
        Cm = s.Cm0 + s.CmV * s.V / s.V0 + s.Cma * s.alpha + s.Cmq * qhat + s.Cmde * s.de
        
        Vd = -s.Sbym*qbar*CD + s.T*cos(s.alpha + s.sigmaT)/s.mass - s.g0*sin(gamma)
        alphad = (-s.Sbym*qbar/s.V*CL - s.T*sin(s.alpha + s.sigmaT)/(s.mass*s.V) 
                  + s.g0*cos(gamma)/s.V + s.q)
        qd = ScbyIy*qbar*Cm + s.T*s.FEIYLT
        return sympy.Array([Vd, alphad, q, qd])
    
    @sym2num.model.collect_symbols
    def L(self, y, x, u, p, *, s):
        """Measurement log likelihood."""
        return (
            symstats.normal_logpdf1(s.alpha_meas, s.alpha, s.alpha_meas_std)
            + symstats.normal_logpdf1(s.q_meas, s.q, s.q_meas_std)
        )

    @sym2num.model.collect_symbols
    def g(self, x, u, p, *, s):
        """Path constraints."""
        return sympy.Array([], 0)
    
    @sym2num.model.collect_symbols
    def h(self, xe, p, *, s):
        """Endpoint constraints."""
        return sympy.Array([], 0)
    
    @sym2num.model.collect_symbols
    def L(self, x, u, p, *, s):
        """Lagrange (running) cost."""
        return sympy.Array(0.5*s.u1**2*s.T) 
    
    @sym2num.model.collect_symbols
    def M(self, xe, p, *, s):
        """Mayer (endpoint) cost."""
        return sympy.Array(0)

