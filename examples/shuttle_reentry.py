"""Shuttle Reentry maximum crossrange optimal control problem."""


import functools

import numpy as np
import sympy
from sympy import sin, cos, exp, sqrt
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
                  'hr', 'm', 'c0', 'c1', 'c2', 'c3', 'qU']
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
        L = qbar * s.S * CL
        D = qbar * s.S * CD
        
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
        rho = s.rho0 * exp(-s.h / s.hr)
        qr = 17700 * sqrt(rho) * (0.0001 * s.v) ** 3.07
        qa = s.c0 + s.c1 * s.alpha + s.c2 * s.alpha ** 2 + s.c3 * s.alpha ** 3
        return [qa*qr / s.qU]
    
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

    def u(x):
        ret = np.zeros(x.shape[:-1] + (2,))
        ret[..., 0] = -2*constants.degree - x[..., 4]
        ret[..., 1] = -5*constants.degree
        return ret
    
    def xdot(t, x):
        return problem.model.f(x, u(x), [1])
    
    tf = 80
    x0 = [260e3, 0, 0, 25.6e3, -1*constants.degree, 0]
    tspan = [0, tf]
    sol = integrate.solve_ivp(xdot, tspan, x0, max_step=1)
    x = interpolate.interp1d(sol.t / tf, sol.y)(problem.tc).T
    u = interpolate.interp1d(sol.t / tf, u(sol.y.T).T)(problem.tc).T
    
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
        'a0': -0.20704, 'a1': 0.029244 * r2d, 'mu': 0.14076539e17,
        'b0': 0.07854, 'b1': -0.61592e-2 * r2d, 'b2': 0.621408 * r2d**2,
        'c0': 1.0672181, 'c1': -0.19213774e-1 * r2d, 
        'c2': 0.21286289e-3 * r2d**2, 'c3': -0.10117249e-5 * r2d**3, 
        'qU': 70, 'm': 203e3 / 32.174, 
    }
    model = GeneratedShuttleReentry(**given)
    
    t = np.linspace(0, 1, 1000)
    problem = oc.Problem(model, t)
    ntc = problem.tc.size
    
    dec_L, dec_U = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    problem.set_decision_item('tf', 0, dec_L)
    problem.set_decision_item('h', 0, dec_L)
    problem.set_decision_item('h', 300e3, dec_U)
    problem.set_decision_item('v', 100, dec_L)
    problem.set_decision_item('alpha', -70*d2r, dec_L)
    problem.set_decision_item('alpha', 70*d2r, dec_U)
    problem.set_decision_item('beta', -89*d2r, dec_L)
    problem.set_decision_item('beta', 1*d2r, dec_U)
    problem.set_decision_item('theta', 0*d2r, dec_L)
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
    problem.set_decision_item('psi_final', 50*d2r, dec_U)
    problem.set_decision_item('theta_final', 10*d2r, dec_L)
    
    dec_scale = np.ones(problem.ndec)
    problem.set_decision_item('tf', 1 / 1000, dec_scale)
    problem.set_decision_item('h', 1 / 260e3, dec_scale)
    problem.set_decision_item('v', 1 / 20e3, dec_scale)
    problem.set_decision_item('gamma', 2, dec_scale)
    problem.set_decision_item('alpha', 2, dec_scale)
    
    constr_scale = np.ones(problem.ncons)
    problem.set_defect_scale('h', 1 / 260e3, constr_scale)
    problem.set_defect_scale('v', 1 / 20e3, constr_scale)
    problem.set_defect_scale('gamma', 2, constr_scale)
    
    constr_L, constr_U = np.zeros((2, problem.ncons))
    problem.set_constraint('g', 1, constr_U)

    dec0 = np.zeros(problem.ndec)
    problem.set_decision_item('tf', 1000, dec0)
    problem.set_decision_item('h', np.linspace(260e3, 80e3, ntc), dec0)
    problem.set_decision_item('v', np.linspace(25.6e3, 2.5e3, ntc), dec0)
    problem.set_decision_item('phi', np.linspace(0, 80*d2r, ntc), dec0)
    problem.set_decision_item('theta', np.linspace(0, 30*d2r, ntc), dec0)
    problem.set_decision_item('psi', np.linspace(90*d2r, 10*d2r, ntc), dec0)
    problem.set_decision_item('alpha', 15*d2r, dec0)
    problem.set_decision_item('beta', np.linspace(-80*d2r, 0*d2r, ntc), dec0)
    
    with problem.ipopt((dec_L, dec_U), (constr_L, constr_U)) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_str_option('ma57_automatic_scaling', 'yes')
        nlp.add_num_option('ma57_pre_alloc', 10)
        nlp.add_int_option('ma57_small_pivot_flag', 1)
        nlp.add_num_option('tol', 1e-5)
        nlp.add_int_option('max_iter', 5000)
        nlp.set_scaling(-1, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    uopt = opt['u']
    tfopt = popt = opt['p']
    topt = problem.tc * tfopt
