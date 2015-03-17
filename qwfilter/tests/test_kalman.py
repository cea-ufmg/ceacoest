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
def nx(request):
    '''Number of states to test with.'''
    return request.param


@pytest.fixture(params=range(1, 3))
def np_(request):
    '''Number of parameters to test with.'''
    return request.param


@pytest.fixture
def x(seed, nx):
    '''Random state vector.'''
    return np.random.randn(nx)


@pytest.fixture
def p(seed, np_):
    '''Random parameter vector.'''
    return np.random.randn(np_)


@pytest.fixture
def cov(seed, nx):
    '''Random x covariance matrix.'''
    M = np.random.randn(nx, nx + 1)
    return M.dot(M.T)


@pytest.fixture
def A(seed, nx):
    '''Random state transition matrix.'''
    A = np.random.randn(nx, nx)
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
def ut(ut_sqrt, ut_kappa, nx):
    '''Standalone UnscentedTransform object.'''
    return kalman.UnscentedTransform(nx, sqrt=ut_sqrt, kappa=ut_kappa)


class NonlinearFunction:
    def __init__(self, nx, np_):
        self.nx = nx
        self.np = np_

    def __call__(self, x, p):
        f = np.zeros_like(x)
        for i, j, k in np.ndindex(self.nx, self.nx, self.np):
            if i <= j:
                f[..., j] += p[k] * x[i] ** (k + j)
        return f
    
    def d_dp(self, x, p):
        df_dp = np.zeros((self.np,) + np.shape(x))
        for i, j, k in np.ndindex(self.nx, self.nx, self.np):
            if i <= j:
                df_dp[k, ..., j] +=  x[i] ** (k + j)
        return df_dp
    
    def d_dx(self, x, p):
        df_dx = np.zeros((self.nx,) + np.shape(x))
        for i, j, k in np.ndindex(self.nx, self.nx, self.np):
            if i <= j and k + j != 0:
                df_dx[i, ..., j] += p[k] * (k + j) * x[i] ** (k + j - 1)
        return df_dx


@pytest.fixture
def nlfunction(nx, np_):
    return NonlinearFunction(nx, np_)


def test_nlfunction_dx(nlfunction, x, p):
    numerical = utils.central_diff(lambda x: nlfunction(x, p), x)
    analytical = nlfunction.d_dx(x, p)
    assert ArrayCmp(analytical) == numerical


def test_nlfunction_dp(nlfunction, x, p):
    numerical = utils.central_diff(lambda p: nlfunction(x, p), p)
    analytical = nlfunction.d_dp(x, p)
    assert ArrayCmp(analytical) == numerical


def test_ut_sqrt(ut_sqrt_func, cov):
    S = ut_sqrt_func(cov)
    STS = np.dot(S.T, S)
    assert ArrayCmp(STS) == cov


def test_cholesky_sqrt_diff(cov, nx):
    S = kalman.cholesky_sqrt(cov)
    def f(x):
        Q = cov.copy()
        Q[i, j] += x
        if i != j:
            Q[j, i] += x
        return kalman.cholesky_sqrt(Q)
    
    jac = kalman.cholesky_sqrt_diff(S)
    for i, j in np.ndindex(nx, nx):
        numerical = utils.central_diff(f, 0)
        assert ArrayCmp(jac[i, j], atol=1e-7) == numerical
        
        dQ = np.zeros((nx, nx))
        dQ[i, j] = 1
        dQ[j, i] = 1
        jac_ij = kalman.cholesky_sqrt_diff(S, dQ)
        assert ArrayCmp(jac[i, j]) == jac_ij


def test_sigma_points(ut, x, cov):
    '''Test if the mean and covariance of the sigma-points is sane.'''
    sigma = ut.gen_sigma_points(x, cov)
    ut_mean = np.dot(ut.weights, sigma)
    assert ArrayCmp(ut_mean) == x
    
    dev = sigma - ut_mean
    ut_cov = np.einsum('ki,kj,k', dev, dev, ut.weights)
    assert ArrayCmp(ut_cov) == cov


def test_affine_ut(ut, x, cov, A, nx):
    '''Test the unscented transform of an affine function.'''
    f = lambda x: np.dot(x, A.T) + np.arange(nx)
    [ut_mean, ut_cov] = ut.unscented_transform(f, x, cov)
    
    desired_mean = f(x)
    assert ArrayCmp(ut_mean) == desired_mean

    desired_cov = A.dot(cov).dot(A.T)
    assert ArrayCmp(ut_cov) == desired_cov
    
    ut_crosscov = ut.transform_crosscov()
    desired_crosscov = np.dot(cov, A.T)
    assert ArrayCmp(ut_crosscov) == desired_crosscov


def test_sigma_points_diff(ut, ut_sqrt, x, cov, nx):
    '''Test the derivative of the unscented transform sigma points.'''
    if ut_sqrt == 'svd':
        pytest.skip("`svd_sqrt_diff` not implemented yet.")
    
    def sigma(mean, cov):
        return ut.gen_sigma_points(mean, cov)[0]
    
    ds_dmean_num = utils.central_diff(lambda x: sigma(x, cov), x)
    
    sigma(x, cov)
    ds_dmean = ut.sigma_points_diff(np.identity(nx), np.zeros((nx, nx, nx)))
    assert ArrayCmp(ds_dmean_num) == ds_dmean


#######################################################################
############# TODO: update below to use nlfunction, x instead of vec 
############# and nx instead of n.
#######################################################################
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

