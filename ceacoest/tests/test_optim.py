"""Optimization problem tests and test infrastructure."""


import numpy as np
import pytest
from scipy import sparse

from ceacoest import utils
from ceacoest.testsupport.array_cmp import ArrayDiff


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


def test_merit_gradient(problem, dec):
    dec = np.random.randn(problem.ndec) if dec is None else dec
    grad = problem.merit_gradient(dec)
    merit_diff = utils.central_diff(problem.merit, dec)
    assert ArrayDiff(grad, merit_diff) < 1e-7


def test_merit_hessian(problem, dec):
    dec = np.random.randn(problem.ndec) if dec is None else dec
    hess = full_hessian(problem)(dec)
    grad_diff = utils.central_diff(problem.merit_gradient, dec)
    assert ArrayDiff(hess, grad_diff) < 1e-7


def test_constraint_jacobian(problem, dec):
    dec = np.random.randn(problem.ndec) if dec is None else dec
    jac = full_jac(problem)(dec)
    cons_diff = utils.central_diff(problem.constraint, dec)
    assert ArrayDiff(jac, cons_diff) < 1e-7


def test_constraint_hessian(problem, dec):
    dec = np.random.randn(problem.ndec) if dec is None else dec
    jac_diff = utils.central_diff(full_jac(problem), dec)
    hess = full_cons_hessian(problem)
    for i in range(problem.ncons):
        mult = np.zeros(problem.ncons)
        mult[i] = 1
        H = hess(dec,mult)
        assert ArrayDiff(H, jac_diff[:,:,i]) < 1e-7, f'{i}-th constraint'


@pytest.fixture(params=[])
def problem(request):
    """Optimization problem."""
    raise NotImplementedError


@pytest.fixture(params=range(4), ids=lambda i: f'seed{i}')
def seed(request):
    """Random number generator seed."""
    np.random.seed(request.param)
    return request.param


@pytest.fixture
def dec(problem, seed):
    """Random problem decision variable."""
    return np.random.randn(problem.ndec)
