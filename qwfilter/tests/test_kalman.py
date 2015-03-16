'''Kalman filtering / smoothing test module.'''


import numpy as np
import numpy.ma as ma
import numpy.testing
import pytest

from qwfilter import kalman, utils


pytest_plugins = "qwfilter.testsupport.array_cmp"
from qwfilter.testsupport.array_cmp import ArrayCmp


@pytest.fixture(params=range(3))
def seed(request):
    '''Random number generator seed.'''
    np.random.seed(request.param)
    return request.param


@pytest.fixture(params=range(1, 5))
def n(request):
    '''Length of test vector or matrix.'''
    return request.param


@pytest.fixture
def vec(seed, n):
    '''Random vector.'''
    return np.random.randn(n)


@pytest.fixture
def cov(seed, n):
    '''Random n by n positive definite symmetric matrix.'''
    M = np.random.randn(n, n + 1)
    return M.dot(M.T)


@pytest.fixture
def mat(seed, n):
    '''Random n by n matrix.'''
    A = np.random.randn(n, n)
    return A


@pytest.fixture(params=['cholesky', 'svd'])
def ut_sqrt(request):
    '''Unscented transform square root option.'''
    return request.param


@pytest.fixture
def ut_sqrt_func(ut_sqrt):
    '''Function corresponding to the square root option.'''
    if ut_sqrt == 'svd':
        return kalman.svd_sqrt
    elif ut_sqrt == 'cholesky':
        return kalman.cholesky_sqrt


@pytest.fixture(params=[0, 0.5, 1])
def ut_kappa(request):
    '''Unscented transform kappa parameter.'''
    return request.param


@pytest.fixture
def ut(ut_sqrt, ut_kappa):
    '''Standalone UnscentedTransform object.'''
    return kalman.UnscentedTransform(sqrt=ut_sqrt, kappa=ut_kappa)


def test_ut_sqrt(ut_sqrt_func, cov):
    S = ut_sqrt_func(cov)
    STS = np.dot(S.T, S)
    assert ArrayCmp(STS) == cov


def test_cholesky_sqrt_diff(cov, n):
    S = kalman.cholesky_sqrt(cov)
    def f(x):
        Q = cov.copy()
        Q[i, j] += x
        if i != j:
            Q[j, i] += x
        return kalman.cholesky_sqrt(Q)
    
    jac = kalman.cholesky_sqrt_diff(S)
    for i, j in np.ndindex(n, n):
        numerical = utils.central_diff(f, 0)
        assert ArrayCmp(jac[i, j], atol=1e-7) == numerical
        
        dQ = np.zeros((n, n))
        dQ[i, j] = 1
        dQ[j, i] = 1
        jac_ij = kalman.cholesky_sqrt_diff(S, dQ)
        assert ArrayCmp(jac[i, j]) == jac_ij


def test_sigma_points(ut, vec, cov):
    '''Test if the mean and covariance of the sigma-points is sane.'''
    [sigma, weights] = ut.gen_sigma_points(vec, cov)
    ut_mean = sigma.dot(weights)
    np.testing.assert_allclose(ut_mean, vec)

    dev = sigma - ut_mean[:, None]
    ut_cov = np.einsum('ik,jk,k', dev, dev, weights)
    np.testing.assert_allclose(ut_cov, cov)


def test_linear_ut(ut, vec, cov, mat):
    '''Test the unscented transform of a linear function.'''
    f = lambda x: mat.dot(x) + 1
    [ut_mean, ut_cov] = ut.unscented_transform(f, vec, cov)
    
    desired_mean = f(vec)
    np.testing.assert_allclose(ut_mean, desired_mean)

    desired_cov = mat.dot(cov).dot(mat.T)
    np.testing.assert_allclose(ut_cov, desired_cov)
    
    ut_crosscov = ut.transform_crosscov()
    desired_crosscov = cov.dot(mat.T)
    np.testing.assert_allclose(ut_crosscov, desired_crosscov)


def test_sigma_points_diff(ut, ut_sqrt, vec, cov, n):
    '''Test the derivative of the unscented transform sigma points.'''
    if ut_sqrt == 'svd':
        pytest.skip("`svd_sqrt_diff` not implemented yet.")
    
    def sigma(mean, cov):
        return ut.gen_sigma_points(mean, cov)[0]
    
    ds_dmean_num = utils.central_diff(lambda x: sigma(x, cov), vec)
    
    sigma(vec, cov)
    ds_dmean = ut.sigma_points_diff(np.identity(n), np.zeros((n, n, n)))
    np.testing.assert_allclose(ds_dmean_num, ds_dmean)


def test_transform_diff_wrt_q(ut, ut_sqrt, n, vec, cov):
    '''Test the derivatives of unscented transform.'''
    if ut_sqrt == 'svd':
        pytest.skip("`svd_sqrt_diff` not implemented yet.")
    
    def f(x, q):
        return np.cumsum(q)[:, None] * x
        
    def ut_mean(q):
        return ut.unscented_transform(lambda x: f(x, q), vec, cov)[0]
    
    def ut_cov(q):
        return ut.unscented_transform(lambda x: f(x, q), vec, cov)[1]
    
    q0 = np.arange(n) + 1
    num_mean_diff = utils.central_diff(ut_mean, q0)
    num_cov_diff = utils.central_diff(ut_cov, q0)

    def f_diff(x, dx=0):
        assert np.all(dx == 0)
        i, j = np.tril_indices(n)
        ret = np.zeros_like(dx)
        ret[i, :, j] = x[i]
        return ret
    
    mean_diff = np.zeros((n, n))
    cov_diff = np.zeros((n, n, n))
    ut.unscented_transform(lambda x: f(x, q0), vec, cov)
    mean_diff, cov_diff = ut.transform_diff(f_diff, mean_diff, cov_diff)
    
    np.testing.assert_allclose(mean_diff, num_mean_diff, atol=1e-8)
    np.testing.assert_allclose(cov_diff, num_cov_diff, atol=1e-8)


class EulerDiscretizedAtmosphericReentry:
    nx = 5
    nu = 0
    nw = 2
    ny = 2
    
    wcov = np.diag([2.4064e-5, 2.4064e-5])
    vcov = np.diag([0.017, 0.001]) ** 2
    
    def f(self, k, x, u=None, w=None):
        [x1, x2, x3, x4, x5] = x
        [w1, w2] = w if w is not None else [0.0, 0.0]
        
        beta0 = -0.59783
        H0 = 13.406
        Gm0 = 3.9860e5
        R0 = 6374
        
        R = np.hypot(x1, x2)
        V = np.hypot(x3, x4)
        beta = beta0 * np.exp(x5)
        D = beta * np.exp((R0 - R) / H0) * V
        G = -Gm0 / R ** 3
        
        dx1 = x3
        dx2 = x4
        dx3 = D * x3 + G * x1
        dx4 = D * x4 + G * x2
        dx5 = np.zeros_like(x5)
        drift = np.array([dx1, dx2, dx3, dx4, dx5])
        
        perturb = np.zeros((5,) + np.shape(w1))
        perturb[2] = w1
        perturb[3] = w2
        
        T = 0.05
        return x + drift * T + perturb * np.sqrt(T)
    
    def h(self, t, x, u=None):
        [x1, x2, x3, x4, x5] = np.asarray(x)
        
        xr = 6374
        yr = 0
        
        rr = np.hypot(x1 - xr, x2 - yr)
        theta = np.arctan2(x2 - yr, x1 - xr)
        
        return np.array([rr, theta])


def sim():
    model = EulerDiscretizedAtmosphericReentry()
    
    N = 4001
    x = np.zeros((N, model.nx))
    x[0] = np.random.multivariate_normal(
        [6500.4, 349.14, -1.8093, -6.7967, 0.6932], 
        np.diag([1e-6, 1e-6, 1e-6, 1e-6, 0])
    )
    w = np.random.multivariate_normal([0, 0], model.w_cov, N)
    
    for k in range(N-1):
        x[k+1] = model.f(k, x[k], [], w[k])

    y = ma.zeros((N, 2))
    v = np.random.multivariate_normal([0, 0], model.w_cov, (N + 1) // 2)
    y[::2] = model.h(None, x[::2].T).T + v
    y[1::2] = ma.masked
    
    return [x, y, model]


def run_filter():
    [x_sim, y, model] = sim()
    
    x0_mean = [6500.4, 349.14, -1.8093, -6.7967, 0]
    x0_cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1])
    filter = kalman.DTUnscentedKalmanFilter(
        model, x0_mean, x0_cov, sqrt='cholesky', kappa=0.5
    )
    filter.filter(y)

    return [filter, x_sim, y, model]


def test_dummy():
    a = np.asarray([1,3])
    assert ArrayCmp(a) == [2, 3]

