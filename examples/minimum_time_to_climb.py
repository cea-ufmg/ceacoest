"""Minimum time to climb optimal control problem."""


import numpy as np
import sympy
from sympy import sin, cos
from scipy import  constants, integrate, interpolate

import sym2num.model
import sym2num.var
import sym2num.spline
from ceacoest import oc, symb_oc


hS = 5e5
vS = 1e3


@symb_oc.collocate(order=3)
class MinimumTimeToClimb:
    """Symbolic minimum time to climb optimal control model."""
    
    @sym2num.model.make_variables_dict
    def variables():
        """Model variables definition."""
        obj  = sym2num.var.SymbolObject(
            'self', 
            sym2num.spline.BivariateSpline('T'), 
            sym2num.spline.UnivariateSpline('CLa'),
            sym2num.spline.UnivariateSpline('CD0'),
            sym2num.spline.UnivariateSpline('eta'),
            sym2num.spline.UnivariateSpline('speed_of_sound'),
            sym2num.spline.UnivariateSpline('rho'),
            sym2num.var.SymbolArray('consts', ['S', 'Isp', 'Re', 'mu', 'g0'])
        )
        return [obj,
                sym2num.var.SymbolArray('x', ['h', 'v', 'gamma', 'w']),
                sym2num.var.SymbolArray('u', ['alpha']),
                sym2num.var.SymbolArray('p', ['tf'])]
    
    @sym2num.model.symbols_from('x, u, p')
    def f(self, s):
        """ODE function."""
        m = s.w / s.g0
        v = s.v * vS
        h = s.h * hS
        M = v / s.speed_of_sound(s.h)
        T = s.T(s.h, M)
        qbar = 0.5 * s.rho(s.h) * v**2
        L = qbar * s.S * s.CLa(M) * s.alpha
        D = qbar * s.S * (s.CD0(M) + s.eta(M)*s.CLa(M)*s.alpha**2)
        g = s.mu / (s.Re + h) ** 2
        f = [v * sin(s.gamma) / hS,
             ((T * cos(s.alpha) - D)/m - g * sin(s.gamma)) / vS,
             (T*sin(s.alpha) + L)/m/v + cos(s.gamma)*(v/(s.Re+h) - g/v),
             -T / s.Isp]
        return sympy.Array(f) * s.tf
    
    @sym2num.model.symbols_from('x, u, p')
    def g(self, s):
        """Path constraints."""
        return sympy.Array([], 0)
    
    @sym2num.model.symbols_from('xe, p')
    def h(self, s):
        """Endpoint constraints."""
        return sympy.Array([s.h_start, 
                            s.v_start - 424.26 / vS, 
                            s.gamma_start, 
                            (s.w_start - 42e3) / 40e3,
                            s.h_end - 65600 / hS,
                            s.v_end - 968.148 / vS, 
                            s.gamma_end])
    
    @sym2num.model.symbols_from('xe, p')
    def M(self, s):
        """Mayer (endpoint) cost."""
        return sympy.Array(s.tf)


def guess(problem):
    def u(x):
        return 4*constants.degree - x[[2]]
    
    def xdot(t, x):
        return problem.model.f(x, u(x), [1])
    
    tf = 400
    x0 = [0, 424.26 / vS, 0, 42e3]
    tspan = [0, tf]
    sol = integrate.solve_ivp(xdot, tspan, x0, max_step=1)
    
    x = interpolate.interp1d(sol.t / tf, sol.y)(problem.tc).T
    u = interpolate.interp1d(sol.t / tf, u(sol.y))(problem.tc).T
    p = np.array([tf])
    
    dec0 = np.zeros(problem.ndec)
    problem.set_decision('p', p, dec0)
    problem.set_decision('u', u, dec0)
    problem.set_decision('x', x, dec0)

    return dec0


if __name__ == '__main__':
    T_h = np.r_[0:31e3:5e3, 40e3, 50e3, 70e3] / hS
    T_M = np.r_[0:1.9:0.2]
    T = np.array([[24.2, 22.2, 19.8, 17.8, 14.8, 12.3, 10.3, 6.3, 3.3, 0.3],
                  [28, 24.6, 21.1, 18.1, 15.2, 12.8, 10.7, 6.7, 3.7, 0.5],
                  [28.3, 25.2, 21.9, 18.7, 15.9, 13.4, 11.2, 7.3, 4.4, 0.7],
                  [30.8, 27.2, 23.8, 20.5, 17.3, 14.7, 12.3, 8.1, 4.9, 0.9],
                  [34.5, 30.3, 26.6, 23.2, 19.8, 16.8, 14.1, 9.4, 5.6, 1.1],
                  [37.9, 34.3, 30.4, 26.8, 23.3, 19.8, 16.8, 11.2, 6.8, 1.4],
                  [36.1, 38.0, 34.9, 31.3, 27.3, 23.6, 20.1, 13.4, 8.3, 1.7],
                  [34.3, 36.6, 38.5, 36.1, 31.6, 28.1, 24.2, 16.2, 10,  2.2],
                  [32.5, 35.2, 37.5, 38.7, 35.7, 32.0, 28.1, 19.3, 11.9, 2.9],
                  [30.7, 33.8, 36.5, 38, 37, 34.6, 31.1, 21.7, 13.3, 3.1]])*1e3
    T_spline = interpolate.RectBivariateSpline(T_h, T_M, T.T)
    
    aero_M = np.r_[0:0.9:0.4, 0.9, 1:2:0.2]
    CLa = np.array([3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44])
    CD0 = np.array([13, 13, 13, 14, 31, 41, 39, 36, 35]) * 1e-3
    eta = np.array([54, 54, 54, 75, 79, 78, 89, 93, 93]) * 1e-2
    CLa_pchip = interpolate.PchipInterpolator(aero_M, CLa)
    CD0_pchip = interpolate.PchipInterpolator(aero_M, CD0)
    eta_pchip = interpolate.PchipInterpolator(aero_M, eta)

    temp_h = np.array([0, 11e3, 20e3, 32e3]) / constants.foot / hS
    temp = np.array([288.15, 216.65, 216.65, 226.650])
    a = np.sqrt(287.058 *  1.4 * temp) / constants.foot
    a_spline = interpolate.InterpolatedUnivariateSpline(temp_h, a, k=1)

    rho_h, rho = np.r_[0.00000, 0.00237717,
                       5000.00, 0.00204834,
                       10000.0, 0.00175549,
                       15000.0, 0.00149581,
                       20000.0, 0.00126659,
                       25000.0, 0.00106526,
                       30000.0, 0.000889378,
                       35000.0, 0.000736627,
                       40000.0, 0.000585189,
                       45000.0, 0.000460180,
                       50000.0, 0.000361876,
                       55000.0, 0.000284571,
                       60000.0, 0.000223781,
                       65000.0, 0.000175976,
                       70000.0, 0.000137625].reshape((-1,2)).T / [[hS],[1]]
    rho_spline = interpolate.UnivariateSpline(rho_h, rho)
    
    symb_mdl = MinimumTimeToClimb()
    GeneratedMinimumTimeToClimb = sym2num.model.compile_class(
        'GeneratedMinimumTimeToClimb', symb_mdl
    )
    mdl = GeneratedMinimumTimeToClimb()
    mdl.T = T_spline
    mdl.CLa = CLa_pchip
    mdl.CD0 = CD0_pchip
    mdl.eta = eta_pchip
    mdl.rho = rho_spline
    mdl.speed_of_sound = lambda h, dx=0: 0 if dx > 1 else a_spline(h, dx)
    mdl.consts = np.array([530, 1600, 20902900, 0.14076539e17, 32.174])
    #   consts:           ['S', 'Isp', 'Re',    'mu',          'g0']
    
    t = np.linspace(0, 1, 250)
    problem = oc.Problem(mdl, t)
    
    d2r = constants.degree
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    problem.set_decision('p', 0, dec_bounds[0])
    problem.set_decision('x', [0, 1/vS, -89*d2r, 0],    dec_bounds[0])
    problem.set_decision('x', [69e3/hS, 2e3/vS, 89*d2r, 45e3], dec_bounds[1])
    problem.set_decision('u', -20*d2r, dec_bounds[0])
    problem.set_decision('u',  20*d2r, dec_bounds[1])
    constr_bounds = np.zeros((2, problem.ncons))
    
    obj = lambda dec, new: problem.merit(dec)
    grad = lambda dec, new: problem.merit_gradient(dec)
    constr = lambda dec, new: problem.constraint(dec)
    jac_ind = problem.constraint_jacobian_ind()[[1,0]]
    jac_val = lambda dec, new: problem.constraint_jacobian_val(dec)
    hess_ind = np.c_[problem.merit_hessian_ind(),
                     problem.constraint_hessian_ind()][[1,0]]
    def hess_val(dec, newd, obj_factor, mult, newm):
        return np.r_[problem.merit_hessian_val(dec) * obj_factor,
                     problem.constraint_hessian_val(dec, mult)]
    
    import yaipopt
    nlp = yaipopt.Problem(dec_bounds, obj, grad, constr_bounds, constr,
                          jac_val, jac_ind, hess_val, hess_ind)
    nlp.str_option('linear_solver', 'ma57')
    nlp.num_option('tol', 1e-6)
    #nlp.int_option('max_iter', 1000)
    
    ntc = problem.tc.size
    dec0 = guess(problem)
    decopt, info = nlp.solve(dec0)
    opt = problem.variables(decopt)
    xopt = opt['x'] 
    uopt = opt['u']
    tc = problem.tc * opt['p']
