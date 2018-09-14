"""Minimum time to climb optimal control problem."""


import functools

import numpy as np
import sympy
from sympy import sin, cos
from scipy import constants, integrate, interpolate

import sym2num.model
import sym2num.utils
import sym2num.var
from ceacoest import oc, symb_oc


hS = 5e5
vS = 1e3

# Propulsion model tables
T_h = np.r_[0, 5, 10, 15, 20, 25, 30, 40, 50, 70] * 1e3 / hS
T_M = np.r_[0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8]
T_data = np.array(
    [[24.2, 28.0, 28.3, 30.8, 34.5, 37.9, 36.1, 34.3, 32.5, 30.7],
     [22.2, 24.6, 25.2, 27.2, 30.3, 34.3, 38.0, 36.6, 35.2, 33.8],
     [19.8, 21.1, 21.9, 23.8, 26.6, 30.4, 34.9, 38.5, 37.5, 36.5],
     [17.8, 18.1, 18.7, 20.5, 23.2, 26.8, 31.3, 36.1, 38.7, 38.0],
     [14.8, 15.2, 15.9, 17.3, 19.8, 23.3, 27.3, 31.6, 35.7, 37.0],
     [12.3, 12.8, 13.4, 14.7, 16.8, 19.8, 23.6, 28.1, 32.0, 34.6],
     [10.3, 10.7, 11.2, 12.3, 14.1, 16.8, 20.1, 24.2, 28.1, 31.1],
     [6.3, 6.7, 7.3, 8.1, 9.4, 11.2, 13.4, 16.2, 19.3, 21.7],
     [3.3, 3.7, 4.4, 4.9, 5.6, 6.8, 8.3, 10.0, 11.9, 13.3],
     [0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.7, 2.2, 2.9, 3.1]]
) * 1e3

# Aerodynamic model tables
aero_M = np.r_[0, 0.4, 0.8, 0.9, 1, 1.2, 1.4, 1.6, 1.8]
CLa_data = np.array([3.44, 3.44, 3.44, 3.58, 4.44, 3.44, 3.01, 2.86, 2.44])
CD0_data = np.array([13, 13, 13, 14, 31, 41, 39, 36, 35]) * 1e-3
eta_data = np.array([54, 54, 54, 75, 79, 78, 89, 93, 93]) * 1e-2

# Air density table
rho_h = np.r_[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70] *1e3/hS
rho_data = np.r_[0.00237717, 0.00204834, 0.00175549, 0.00149581, 0.00126659, 
                 0.00106526, 0.00088938, 0.00073663, 0.00058519, 0.00046018, 
                 0.00036188, 0.00028457, 0.00022378, 0.00017598, 0.00013762]

# Speed of sound table
a_h = np.r_[0,  36089,  65617, 104986] / hS
a_data = np.r_[1116.46, 968.08, 968.08, 990.17]


@symb_oc.collocate(order=3)
class MinimumTimeToClimb:
    """Symbolic minimum time to climb optimal control model."""
    
    @sym2num.utils.classproperty
    @functools.lru_cache()
    def variables(cls):
        """Model variables definition."""
        obj  = sym2num.var.SymbolObject(
            'self', 
            sym2num.var.BivariateCallable('T'), 
            sym2num.var.UnivariateCallable('CLa'),
            sym2num.var.UnivariateCallable('CD0'),
            sym2num.var.UnivariateCallable('eta'),
            sym2num.var.UnivariateCallable('speed_of_sound'),
            sym2num.var.UnivariateCallable('rho'),
            sym2num.var.SymbolArray('consts', ['S', 'Isp', 'Re', 'mu', 'g0'])
        )
        vars = [obj,
                sym2num.var.SymbolArray('x', ['h', 'v', 'gamma', 'w']),
                sym2num.var.SymbolArray('u', ['alpha']),
                sym2num.var.SymbolArray('p', ['tf'])]
        return sym2num.var.make_dict(vars)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
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
    
    @sym2num.model.collect_symbols
    def g(self, x, u, p, *, s):
        """Path constraints."""
        return sympy.Array([], 0)
    
    @sym2num.model.collect_symbols
    def h(self, xe, p, *, s):
        """Endpoint constraints."""
        return sympy.Array([s.h_initial, 
                            s.v_initial - 424.26 / vS, 
                            s.gamma_initial, 
                            (s.w_initial - 42e3) / 40e3,
                            s.h_final - 65600 / hS,
                            s.v_final - 968.148 / vS, 
                            s.gamma_final])
    
    @sym2num.model.collect_symbols
    def M(self, xe, p, *, s):
        """Mayer (endpoint) cost."""
        return sympy.Array(s.tf)
    
    @sym2num.model.collect_symbols
    def L(self, x, u, p, *, s):
        """Lagrange (running) cost."""
        return sympy.Array(0)


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
    T_spline = interpolate.RectBivariateSpline(T_h, T_M, T_data)
    CLa_pchip = interpolate.PchipInterpolator(aero_M, CLa_data)
    CD0_pchip = interpolate.PchipInterpolator(aero_M, CD0_data)
    eta_pchip = interpolate.PchipInterpolator(aero_M, eta_data)
    a_spline = interpolate.InterpolatedUnivariateSpline(a_h, a_data, k=1)
    rho_spline = interpolate.UnivariateSpline(rho_h, rho_data)
    
    symb_mdl = MinimumTimeToClimb()
    GeneratedMinimumTimeToClimb = sym2num.model.compile_class(symb_mdl)

    mdl = GeneratedMinimumTimeToClimb(S=530, Isp=1600, Re=20902900, 
                                      mu=0.14076539e17, g0=32.174)
    mdl.T = T_spline.ev
    mdl.CLa = CLa_pchip
    mdl.CD0 = CD0_pchip
    mdl.eta = eta_pchip
    mdl.rho = rho_spline
    mdl.speed_of_sound = lambda h, dx=0: 0 if dx > 1 else a_spline(h, dx)
    
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
    dec0 = guess(problem)
    
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        decopt, info = nlp.solve(dec0)
    
    opt = problem.variables(decopt)
    xopt = opt['x']
    uopt = opt['u']
    Topt = opt['p']
    topt = problem.tc * Topt
