"""Free flying robot optimal control example.

Based on Betts (2010) Practical Methods for Optimal Control and Estimation
Using Nonlinear Programming, Second Edition, Section 6.13.
"""


import numpy as np
import sympy
import sym2num.model
from sympy import sqrt, exp

from ceacoest import oc, optim
from ceacoest.modelling import symoc


@symoc.collocate(order=2)
class FreeFlyingRobot(sym2num.model.Base):
    """Free flying robot optimal control model."""

    @property
    def variables(self):
        v = super().variables
        v['x'] = 'x1 x2 theta v1 v2 omega'
        v['u'] = 'u1 u2 u3 u4'
        v['p'] = []
        return v

    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        alpha = 0.2
        beta = 0.2
        
        T1 = s.u1 - s.u2
        T2 = s.u3 - s.u4
        
        x1dot = s.v1
        x2dot = s.v2
        thetadot = s.omega
        v1dot = (T1 + T2) * sympy.cos(s.theta)
        v2dot = (T1 + T2) * sympy.sin(s.theta)
        omegadot = alpha * T1 - beta * T2
        return [x1dot, x2dot, thetadot, v1dot, v2dot, omegadot]
    
    @sym2num.model.collect_symbols
    def g(self, x, u, p, *, s):
        """Path constraints."""
        abs_T1 = s.u1 + s.u2
        abs_T2 = s.u3 + s.u4
        return [abs_T1, abs_T2]
    
    @sym2num.model.collect_symbols
    def h(self, xe, p, *, s):
        """Endpoint constraints."""
        return sympy.Array([], 0)
    
    @sym2num.model.collect_symbols
    def L(self, x, u, p, *, s):
        """Lagrange (running) cost."""
        abs_T1 = s.u1 + s.u2
        abs_T2 = s.u3 + s.u4
        return abs_T1 + abs_T2
    
    @sym2num.model.collect_symbols
    def M(self, xe, p, *, s):
        """Mayer (endpoint) cost."""
        return 0


if __name__ == '__main__':
    symb_mdl = FreeFlyingRobot()
    GeneratedFreeFlyingRobot = sym2num.model.compile_class(symb_mdl)
    mdl = GeneratedFreeFlyingRobot()

    t = np.linspace(0, 12, 500)
    problem = oc.Problem(mdl, t)
    tc = problem.tc
    ntc = tc.size

    # Define problem bounds
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    problem.set_decision_item('u1', 0, dec_L)
    problem.set_decision_item('u2', 0, dec_L)
    problem.set_decision_item('u3', 0, dec_L)
    problem.set_decision_item('u4', 0, dec_L)

    #problem.set_decision_item('x1_initial', -10, dec_L)
    #problem.set_decision_item('x1_initial', -10, dec_U)
    problem.set_decision_item('x2_initial', -10, dec_L)
    problem.set_decision_item('x2_initial', -10, dec_U)
    problem.set_decision_item('theta_initial', np.pi/2, dec_L)
    problem.set_decision_item('theta_initial', np.pi/2, dec_U)
    problem.set_decision_item('v1_initial', 0, dec_L)
    problem.set_decision_item('v1_initial', 0, dec_U)
    problem.set_decision_item('v2_initial', 0, dec_L)
    problem.set_decision_item('v2_initial', 0, dec_U)
    problem.set_decision_item('omega_initial', 0, dec_L)
    problem.set_decision_item('omega_initial', 0, dec_U)

    problem.set_decision_item('x1_final', 0, dec_L)
    problem.set_decision_item('x1_final', 0, dec_U)
    problem.set_decision_item('x2_final', 0, dec_L)
    problem.set_decision_item('x2_final', 0, dec_U)
    problem.set_decision_item('theta_final', 0, dec_L)
    problem.set_decision_item('theta_final', 0, dec_U)
    problem.set_decision_item('v1_final', 0, dec_L)
    problem.set_decision_item('v1_final', 0, dec_U)
    problem.set_decision_item('v2_final', 0, dec_L)
    problem.set_decision_item('v2_final', 0, dec_U)
    problem.set_decision_item('omega_final', 0, dec_L)
    problem.set_decision_item('omega_final', 0, dec_U)

    constr_bounds = np.zeros((2, problem.ncons))
    constr_L, constr_U = constr_bounds
    problem.set_constraint('g', -np.inf, constr_L)
    problem.set_constraint('g', 1, constr_U)
    problem.set_constraint('h', -10, constr_L)
    problem.set_constraint('h', -10, constr_U)
    
    # Define initial guess for decision variables
    dec0 = np.zeros(problem.ndec)
    problem.set_decision_item('x1', np.linspace(-10, 0, ntc), dec0)
    problem.set_decision_item('x2', np.linspace(-10, 0, ntc), dec0)
    problem.set_decision_item('theta', np.linspace(np.pi/2, 0, ntc), dec0)
    problem.set_decision_item('v1', 10/12, dec0)
    problem.set_decision_item('v2', 10/12, dec0)
    problem.set_decision_item('omega', -np.pi/2/12, dec0)
    
    # Define problem scaling
    dec_scale = np.ones(problem.ndec)
    constr_scale = np.ones(problem.ncons)
    obj_scale = 1.0
    
    # Run ipopt
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('ma57_pre_alloc', 10.0)
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(obj_scale, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)

    # Unpack the solution
    opt = problem.variables(decopt)
    xopt = opt['x']
    uopt = opt['u']
    popt = opt['p']
    
    # Show results
    from matplotlib import pylab
    pylab.figure(1).clear()
    u1,u2,u3,u4 = uopt.T
    pylab.plot(tc, u1 - u2)

    pylab.figure(2).clear()
    pylab.plot(tc, xopt[:, 5])
    pylab.show()
