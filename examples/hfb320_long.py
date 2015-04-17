'''Prediction Error Method for the HFB-320 aircraft longitudinal motion.'''


import ipopt
import numpy as np
import sympy
import sym2num
from numpy import ma
from scipy import stats

from qwfilter import kalman, sde, utils


class SymbolicModel(sde.SymbolicModel):
    '''Symbolic HFB-320 longitudinal motion model.'''

    var_names = {'t', 'x', 'y', 'u', 'q', 'c'}
    '''Name of model variables.'''
    
    function_names = {'f', 'g', 'h', 'R'}
    '''Name of the model functions.'''

    t = 't'
    '''Time variable.'''
    
    x = ['vT', 'alpha', 'theta', 'q']
    '''State vector.'''
    
    y = ['vT_meas', 'alpha_meas', 'theta_meas', 'q_meas']
    '''Measurement vector.'''

    u = ['de', 'Fe']
    '''Exogenous input vector.'''
    
    q = ['CD0', 'CDv', 'CDa', 'CL0', 'CLv', 'CLa',
         'Cm0', 'Cmv', 'Cma', 'Cmq', 'Cmde', 
         'alpha_png', 'vT_png', 'q_png']
    '''Unknown parameter vector.'''
    
    c = ['g0', 'Sbym', 'Scbyiy', 'Feiylt', 'v0', 'rm', 'sigmat', 'rho']
    '''Constants vector.'''
    
    def f(self, t, x, q, u, c):
        '''Drift function.'''
        s = self.symbols(t=t, x=x, q=q, u=u, c=c)        
        qbar = 0.5 * s.rho * s.vT ** 2
        CD = s.CD0 + s.CDv*s.vT/s.v0 + s.CDa*s.alpha
        CL = s.CL0 + s.CLv*s.vT/s.v0 + s.CLa*s.alpha
        Cm = (s.Cm0 + s.Cmv*s.vT/s.v0 + s.Cma*s.alpha + s.Cmq*s.q/s.v0 +
              s.Cmde*s.de)
        vTdot = (-s.Sbym*qbar*CD + s.Fe*sympy.cos(s.alpha + s.sigmat)/s.rm +
                 s.g0*sympy.sin(s.alpha - s.theta))
        alphadot = (-s.Sbym*qbar*CL/s.vT + s.q -
                    s.Fe*sympy.sin(s.alpha + s.sigmat)/s.rm/s.vT +
                    s.g0*sympy.cos(s.alpha - s.theta)/vT)
        qdot = s.Scbyiy*qbar*Cm + Feiylt*s.Fe        
        derivs = dict(vT=vTdot, alpha=alphadot, theta=q, q=qdot)
        return self.pack('x', derivs)
    
    def g(self, t, x, q, c):
        '''Diffusion matrix.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        return np.r[self.pack('x', vT=s.vT_png),
                    self.pack('x', alpha=s.alpha_png),
                    self.pack('x', q=s.q_png)]
    
    def h(self, t, x, q, c):
        '''Measurement function.'''
        s = self.symbols(t=t, x=x, q=q, c=c)
        meas = dict(vT_meas=s.vT, alpha_meas=s.alpha, 
                    theta_meas=s.theta, q_meas=s.q)
        return self.pack('y', meas)
    
    def R(self, q, c):
        '''Measurement function.'''
        s = self.symbols(q=q, c=c)
        return [[s.x_meas_std ** 2]]
