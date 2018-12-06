"""Hang glider range maximization example.

Based on Betts (2010) Practical Methods for Optimal Control and Estimation
Using Nonlinear Programming, Second Edition, Section 6.5.
"""


import numpy as np
import sympy
import sym2num.model
from sympy import sqrt, exp

from ceacoest import oc, optim
from ceacoest.modelling import symoc


@symoc.collocate(order=3)
class HangGlider(sym2num.model.Base):
    """Hang glider range maximization optimal control problem."""

    @property
    def variables(self):
        v = super().variables
        v['self'] = {'consts': 'uM m R S CD0 rho k g'}
        v['x'] = 'X y vx vy'
        v['u'] = 'CL',
        v['p'] = 'tf',
        return v
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        CD = s.CD0 + s.k * s.CL**2
        XX = (s.X/s.R - 2.5) ** 2
        ua = s.uM * (1 - XX) * exp(-XX)
        Vy = s.vy - ua
        vr = sqrt(s.vx**2 + Vy)
        D = 0.5 * CD * s.rho * s.S * vr**2
        L = 0.5 * s.CL * s.rho * s.S * vr**2
        sin_eta = Vy / vr
        cos_eta = s.vx / vr
        W = s.m * s.g
        Xdot = s.vx
        ydot = s.vy
        vxdot = (-L * sin_eta - D * cos_eta) / s.m
        vydot = (L * cos_eta - D * sin_eta - W) / s.m
        return sympy.Array([Xdot, ydot, vxdot, vydot]) * s.tf
    
    @sym2num.model.collect_symbols
    def g(self, x, u, p, *, s):
        """Path constraints."""
        return [s.vx**2 + s.vy**2]
    
    @sym2num.model.collect_symbols
    def h(self, xe, p, *, s):
        """Endpoint constraints."""
        return sympy.Array([], 0)
    
    @sym2num.model.collect_symbols
    def L(self, x, u, p, *, s):
        """Lagrange (running) cost."""
        return 0
    
    @sym2num.model.collect_symbols
    def M(self, xe, p, *, s):
        """Mayer (endpoint) cost."""
        return s.X_final


if __name__ == '__main__':
    symb_mdl = HangGlider()
    GeneratedHangGlider = sym2num.model.compile_class(symb_mdl)

    consts = dict(
        uM=2.5, m=100, R=100, S=14, CD0=0.034,
        rho=1.13, k=0.069662, g=9.80665,
    )
    mdl = GeneratedHangGlider(**consts)
    
    t = np.linspace(0, 1, 250)
    problem = oc.Problem(mdl, t)
    tc = problem.tc
    ntc = tc.size

    dec_L, dec_U = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    problem.set_decision_item('tf', 10, dec_L)
    problem.set_decision_item('tf', 500, dec_U)
    problem.set_decision_item('CL', 0, dec_L)
    problem.set_decision_item('CL', 1.4, dec_U)
    
    problem.set_decision_item('X', 0, dec_L)
    problem.set_decision_item('y', 0, dec_L)
    problem.set_decision_item('vx', 0, dec_L)
    
    problem.set_decision_item('X_initial', 0, dec_L)
    problem.set_decision_item('X_initial', 0, dec_U)
    problem.set_decision_item('y_initial', 1000, dec_U)
    problem.set_decision_item('y_initial', 1000, dec_L)
    problem.set_decision_item('vx_initial', 13.2, dec_U)
    problem.set_decision_item('vx_initial', 13.2, dec_L)
    problem.set_decision_item('vy_initial', -1.3, dec_U)
    problem.set_decision_item('vy_initial', -1.3, dec_L)
    
    problem.set_decision_item('X_final', 0, dec_L)
    problem.set_decision_item('y_final', 900, dec_L)
    problem.set_decision_item('y_final', 900, dec_U)
    problem.set_decision_item('vx_final', 13.2, dec_L)
    problem.set_decision_item('vx_final', 13.2, dec_U)
    problem.set_decision_item('vy_final', -1.3, dec_L)
    problem.set_decision_item('vy_final', -1.3, dec_U)

    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    problem.set_constraint('g', 1, constr_L)
    problem.set_constraint('g', np.inf, constr_U)

    dec0 = np.zeros(problem.ndec)
    problem.set_decision_item('X', np.linspace(0, 1000, ntc), dec0)
    problem.set_decision_item('y', 950, dec0)
    problem.set_decision_item('vx', 13.2, dec0)
    problem.set_decision_item('vy', -1.3, dec0)
    problem.set_decision_item('tf', 100, dec0)
    problem.set_decision_item('CL', 1, dec0)
    
    dec_scale = np.ones(problem.ndec)
    problem.set_decision_item('tf', 1e-2, dec_scale)
    problem.set_decision_item('X', 1e-2, dec_scale)
    problem.set_decision_item('y', 1e-2, dec_scale)
    problem.set_decision_item('vx', 1e-1, dec_scale)
    problem.set_decision_item('vy', 0.5, dec_scale)

    constr_scale = np.ones(problem.ncons)
    problem.set_defect_scale('X', 1e-2, constr_scale)
    problem.set_defect_scale('y', 1e-1, constr_scale)
    problem.set_defect_scale('vx', 1e-1, constr_scale)
    problem.set_defect_scale('vy', 0.5, constr_scale)
    
    with problem.ipopt((dec_L, dec_U), constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 10.0)
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(-1e-3, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)

    opt = problem.variables(decopt)
    xopt = opt['x']
    uopt = opt['u']
    Topt = opt['p']
    topt = problem.tc * Topt        
