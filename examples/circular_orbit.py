"""Limited-thrust circular orbit transfer."""


import functools

import numpy as np
import sympy
from sympy import sin, cos
from scipy import constants, integrate, interpolate

import sym2num.model
import sym2num.utils
import sym2num.var
from ceacoest import oc
from ceacoest.modelling import symoc


@symoc.collocate(order=2)
class CircularOrbit:
    """Symbolic limited-thrust circular orbit transfer optimal control model."""
    
    @sym2num.utils.classproperty
    @functools.lru_cache()
    def variables(cls):
        """Model variables definition."""
        consts = ['mu', 've', 'T_max', 'R_final']
        obj  = sym2num.var.SymbolObject(
            'self', 
            sym2num.var.SymbolArray('consts', consts)
        )
        vars = [obj,
                sym2num.var.SymbolArray('x', ['X', 'Y', 'vx', 'vy', 'm']),
                sym2num.var.SymbolArray('u', ['T', 'Txn', 'Tyn']),
                sym2num.var.SymbolArray('p', ['tf'])]
        return sym2num.var.make_dict(vars)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        R3 = (s.X**2 + s.Y**2) ** 1.5        
        gx = - s.mu * s.X / R3
        gy = - s.mu * s.Y / R3
        
        Tx = s.T * s.Txn * s.T_max
        Ty = s.T * s.Tyn * s.T_max
        
        ax = gx + Tx / s.m
        ay = gy + Ty / s.m
        
        mdot = - s.T * s.T_max / s.ve

        f = [s.vx, s.vy, ax, ay, mdot]
        return sympy.Array(f) * s.tf
    
    @sym2num.model.collect_symbols
    def g(self, x, u, p, *, s):
        """Path constraints."""
        return sympy.Array([s.Txn**2 + s.Tyn**2 - 1])
    
    @sym2num.model.collect_symbols
    def h(self, xe, p, *, s):
        """Endpoint constraints."""
        R_error = (s.X_final ** 2 + s.Y_final ** 2) / s.R_final ** 2 - 1
        v_dot_r = s.X_final * s.vx_final + s.Y_final * s.vy_final
        
        r_cross_v = s.X_final * s.vy_final - s.Y_final * s.vx_final
        V = sympy.sqrt(s.vx_final**2 + s.vy_final**2)
        V_error = r_cross_v * V - s.mu
        return sympy.Array([R_error, v_dot_r, V_error])
    
    @sym2num.model.collect_symbols
    def M(self, xe, p, *, s):
        """Mayer (endpoint) cost."""
        return sympy.Array(0)
    
    @sym2num.model.collect_symbols
    def L(self, x, u, p, *, s):
        """Lagrange (running) cost."""
        P = 0.5 * s.T * s.ve * s.T_max
        return sympy.Array(P) * s.tf


if __name__ == '__main__':
    symb_mdl = CircularOrbit()
    GeneratedCircularOrbit = sym2num.model.compile_class(symb_mdl)

    mu = 1
    ve = 50
    T_max = 0.05
    R_final = 2
    mdl_consts = dict(mu=mu, ve=ve, T_max=T_max, R_final=R_final)
    mdl = GeneratedCircularOrbit(**mdl_consts)
    
    t = np.linspace(0, 1, 500)
    problem = oc.Problem(mdl, t)
    tc = problem.tc
    
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    problem.set_decision_item('tf', 0, dec_L)
    problem.set_decision_item('tf', 13, dec_U)
    problem.set_decision_item('m', 0, dec_L)
    problem.set_decision_item('T', 0, dec_L)
    problem.set_decision_item('T', 1, dec_U)
    problem.set_decision_item('Txn', -1.5, dec_L)
    problem.set_decision_item('Txn', 1.5, dec_U)
    problem.set_decision_item('Tyn', -1.5, dec_L)
    problem.set_decision_item('Tyn', 1.5, dec_U)
    problem.set_decision_item('X_initial', 1, dec_L)
    problem.set_decision_item('X_initial', 1, dec_U)
    problem.set_decision_item('Y_initial', 0, dec_L)
    problem.set_decision_item('Y_initial', 0, dec_U)
    problem.set_decision_item('vx_initial', 0, dec_L)
    problem.set_decision_item('vx_initial', 0, dec_U)
    problem.set_decision_item('vy_initial', 1, dec_L)
    problem.set_decision_item('vy_initial', 1, dec_U)
    problem.set_decision_item('m_initial', 1, dec_L)
    problem.set_decision_item('m_initial', 1, dec_U)
    
    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    
    dec_scale = np.ones(problem.ndec)
    problem.set_decision_item('m', 1, dec_scale)
    
    constr_scale = np.ones(problem.ncons)
    problem.set_constraint('h', 1, constr_scale)
    problem.set_defect_scale('m', 1, dec_scale)
    
    obj_scale = 1
    
    dec0 = np.zeros(problem.ndec)
    problem.set_decision_item('m', 1, dec0)
    problem.set_decision_item('tf', 2*np.pi, dec0)
    problem.set_decision_item('X', np.cos(2*np.pi*tc), dec0)
    problem.set_decision_item('Y', np.sin(2*np.pi*tc), dec0)
    problem.set_decision_item('vx', -np.sin(2*np.pi*tc), dec0)
    problem.set_decision_item('vy', np.cos(2*np.pi*tc), dec0)
    problem.set_decision_item('Txn', 1, dec0)
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 100.0)
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 4000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    uopt = opt['u']
    Topt = opt['p']
    iopt = mdl.g(xopt, uopt, Topt)
    topt = problem.tc * Topt
