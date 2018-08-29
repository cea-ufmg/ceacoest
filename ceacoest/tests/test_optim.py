"""Optimization problem tests and test infrastructure."""


import numpy as np
from scipy import sparse

from .. import utils


def sparse_fun_to_full(val, ind, shape, fix_sym=False):
    def wrapper(*args, **kwargs):
        M = sparse.coo_matrix((val(*args, **kwargs), ind), shape=shape)
        A = M.toarray()
        if fix_sym:
            A += A.T - np.diag(np.diag(A))
        return A
    return wrapper


def full_hessian(problem):
    ind = problem.merit_hessian_ind()
    shape = (problem.ndec,) * 2
    return sparse_fun_to_full(problem.merit_hessian_val, ind, shape, True)


def full_jac(problem):
    ind = problem.constraint_jacobian_ind()
    shape = (problem.ndec, problem.ncons)
    return sparse_fun_to_full(problem.constraint_jacobian_val, ind, shape)


def full_cons_hessian(problem):
    ind = problem.constraint_hessian_ind()
    shape = (problem.ndec,) * 2
    return sparse_fun_to_full(problem.constraint_hessian_val, ind, shape, True)


def test_merit_gradient(problem, dec=None):
    dec = np.random.randn(problem.ndec) if dec is None else dec
    grad = problem.merit_gradient(dec)
    merit_diff = utils.central_diff(problem.merit, dec)
    np.testing.assert_almost_equal(grad, merit_diff, decimal=6)
    return [grad, merit_diff]


def test_merit_hessian(problem, dec=None):
    dec = np.random.randn(problem.ndec) if dec is None else dec
    hess = full_hessian(problem)(dec)
    grad_diff = utils.central_diff(problem.merit_gradient, dec)
    np.testing.assert_almost_equal(hess, grad_diff, decimal=6)
    return [hess, grad_diff]


def test_constraint_jacobian(problem, dec=None):
    dec = np.random.randn(problem.ndec) if dec is None else dec
    jac = full_jac(problem)(dec)
    cons_diff = utils.central_diff(problem.constraint, dec)
    np.testing.assert_almost_equal(jac, cons_diff, decimal=6)
    return [jac, cons_diff]


def test_constraint_hessian(problem, dec=None):
    dec = np.random.randn(problem.ndec) if dec is None else dec
    jac_diff = utils.central_diff(full_jac(problem), dec)
    hess = full_cons_hessian(problem)
    for i in range(problem.ncons):
        mult = np.zeros(problem.ncons)
        mult[i] = 1
        H = hess(dec,mult)
        np.testing.assert_almost_equal(H, jac_diff[:,:,i], decimal=6)
