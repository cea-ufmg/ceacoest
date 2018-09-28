"""Shuttle Reentry maximum crossrange optimal control problem."""


import functools

import numpy as np
import sympy
from sympy import sin, cos, exp
from scipy import constants, integrate, interpolate

import sym2num.model
import sym2num.utils
from sym2num import var
from ceacoest import oc
from ceacoest.modelling import symoc


@symoc.collocate(order=3)
class ShuttleReentry:
    """Shuttle Reentry maximum crossrange optimal control model."""
    
    @sym2num.utils.classproperty
    @functools.lru_cache()
    def variables(cls):
        """Model variables definition."""
        consts = ['a0', 'a1', 'mu', 'b0', 'b1', 'b2', 'S', 'Re', 'rho0', 
                  'hr', 'm']
        x = ['h', 'phi', 'theta', 'v', 'gamma', 'psi']
        vars = [var.SymbolObject('self', var.SymbolArray('consts', consts)),
                sym2num.var.SymbolArray('x', x),
                sym2num.var.SymbolArray('u', ['alpha', 'beta']),
                sym2num.var.SymbolArray('p', ['tf'])]
        return sym2num.var.make_dict(vars)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        
        rho = s.rho0 * exp(-s.h / s.hr)
        qbar = 0.5 * rho * s.v**2
        CL = s.a0 + s.a1 * s.alpha
        CD = s.b0 + s.b1 * s.alpha + s.b2 * s.alpha ** 2
        L = 0.5 * qbar * s.S * CL
        D = 0.5 * qbar * s.S * CD
        
        r = s.Re + s.h
        g = s.mu / r**2        
        
        hd = s.v * sin(s.gamma)
        phid = s.v / r * cos(s.gamma) * sin(s.psi) / cos(s.theta)
        thetad = s.v / r * cos(s.gamma) * cos(s.psi)
        vd = -D / s.m - g * sin(s.gamma)
        gammad = L / (s.m * s.v) * cos(s.beta) + cos(s.gamma) * (s.v/r - g/s.v)
        psid = (L * sin(s.beta) / (s.m * s.v * cos(s.gamma))
                + s.v / (r*cos(s.theta)) * cos(s.gamma)*sin(s.psi)*sin(s.theta))
        return sympy.Array([hd, phid, thetad, vd, gammad, psid]) * s.tf
    
    @sym2num.model.collect_symbols
    def g(self, x, u, p, *, s):
        """Path constraints."""
        return []
    
    @sym2num.model.collect_symbols
    def h(self, xe, p, *, s):
        """Endpoint constraints."""
        return []
    
    @sym2num.model.collect_symbols
    def M(self, xe, p, *, s):
        """Mayer (endpoint) cost."""
        return s.theta_final
    
    @sym2num.model.collect_symbols
    def L(self, x, u, p, *, s):
        """Lagrange (running) cost."""
        return 0


def guess(problem):
    
    u = np.array([20, -30])
    def xdot(t, x):
        return problem.model.f(x, u, [1])
    
    tf = 2000
    x0 = [260e3, 0, 0, 25.6e3, 0, 0]
    tspan = [0, tf]
    sol = integrate.solve_ivp(xdot, tspan, x0, max_step=1)
    x = interpolate.interp1d(sol.t / tf, sol.y)(problem.tc).T
    
    dec0 = np.zeros(problem.ndec)
    problem.set_decision_item('tf', tf, dec0)
    problem.set_decision('u', u, dec0)
    problem.set_decision('x', x, dec0)

    return dec0


if __name__ == '__main__':
    symbolic_model = ShuttleReentry()
    GeneratedShuttleReentry = sym2num.model.compile_class(symbolic_model)

    d2r = constants.degree
    r2d = 1 / d2r
    given = {
        'rho0': 0.002378, 'hr': 23800, 'Re': 20902900, 'S': 2690,
        'a0': -0.20704 * r2d, 'a1': 0.029244 * r2d, 'mu': 0.14076539e17,
        'b0': 0.07854 * r2d, 'b1': -0.61592e-2 * r2d, 'b2': 0.621408 * r2d,
        'm': 203e3 / 32.174
    }
    model = GeneratedShuttleReentry(**given)
    
    
    t = np.linspace(0, 1, 300)
    problem = oc.Problem(model, t)
    
    dec_L, dec_U = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    problem.set_decision_item('tf', 0, dec_L)
    problem.set_decision_item('h', 0, dec_L)
    problem.set_decision_item('h', 300e3, dec_U)
    problem.set_decision_item('v', 1e3, dec_L)
    problem.set_decision_item('v', 300e3, dec_U)
    problem.set_decision_item('alpha', -90*d2r, dec_L)
    problem.set_decision_item('alpha', 90*d2r, dec_U)
    problem.set_decision_item('beta', -89*d2r, dec_L)
    problem.set_decision_item('beta', 1*d2r, dec_U)
    problem.set_decision_item('theta', -89*d2r, dec_L)
    problem.set_decision_item('theta', 89*d2r, dec_U)
    problem.set_decision_item('gamma', -89*d2r, dec_L)
    problem.set_decision_item('gamma', 89*d2r, dec_U)
    problem.set_decision_item('h_initial', 260e3, dec_L)
    problem.set_decision_item('h_initial', 260e3, dec_U)
    problem.set_decision_item('v_initial', 25.6e3, dec_L)
    problem.set_decision_item('v_initial', 25.6e3, dec_U)
    problem.set_decision_item('gamma_initial', -1*d2r, dec_L)
    problem.set_decision_item('gamma_initial', -1*d2r, dec_U)
    problem.set_decision_item('phi_initial', 0, dec_L)
    problem.set_decision_item('phi_initial', 0, dec_U)
    problem.set_decision_item('theta_initial', 0, dec_L)
    problem.set_decision_item('theta_initial', 0, dec_U)
    problem.set_decision_item('psi_initial', 90*d2r, dec_L)
    problem.set_decision_item('psi_initial', 90*d2r, dec_U)
    problem.set_decision_item('h_final', 80e3, dec_L)
    problem.set_decision_item('h_final', 80e3, dec_U)
    problem.set_decision_item('v_final', 2.5e3, dec_L)
    problem.set_decision_item('v_final', 2.5e3, dec_U)
    problem.set_decision_item('gamma_final', -5*d2r, dec_L)
    problem.set_decision_item('gamma_final', -5*d2r, dec_U)
    
    dec_scale = np.ones(problem.ndec)
    problem.set_decision_item('tf', 1 / 2000, dec_scale)
    problem.set_decision_item('h', 1 / 180e3, dec_scale)
    problem.set_decision_item('v', 1 / 20e3, dec_scale)
    problem.set_decision_item('gamma', 10, dec_scale)
    problem.set_decision_item('alpha', 2, dec_scale)
    
    constr_scale = np.ones(problem.ncons)
    problem.set_defect_scale('h', 1 / 180e3, constr_scale)
    problem.set_defect_scale('v', 1 / 20e3, constr_scale)
    problem.set_defect_scale('gamma', 10, constr_scale)
    
    constr_bounds = np.zeros((2, problem.ncons))
    dec0 = np.zeros(problem.ndec)
    problem.set_decision_item('tf', 2000, dec0)
    problem.set_decision_item('h', 260e3, dec0)
    problem.set_decision_item('v', 26e3, dec0)
    dec0 = guess(problem)
    
    with problem.ipopt((dec_L, dec_U), constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-5)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(-1, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    uopt = opt['u']
    tfopt = opt['p']
    topt = problem.tc * tfopt
