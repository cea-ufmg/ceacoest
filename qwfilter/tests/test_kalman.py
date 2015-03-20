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
def nq(request):
    '''Number of parameters to test with.'''
    return request.param


@pytest.fixture
def x(seed, nx):
    '''Random state vector.'''
    return np.random.randn(nx)


@pytest.fixture
def q(seed, nq):
    '''Random parameter vector.'''
    return np.random.randn(nq)


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
    '''Parametrized nonlinear function.'''
    def __init__(self, nx, nq):
        self.nx = nx
        self.nq = nq

    def __call__(self, x, q):
        f = np.zeros_like(x)
        for i, j, k in np.ndindex(self.nx, self.nx, self.nq):
            if i <= j:
                f[..., j] += q[..., k] * x[..., i] ** (k + j)
        return f
    
    def d_dq(self, x, q):
        df_dq = np.zeros((self.nq,) + np.shape(x))
        for i, j, k in np.ndindex(self.nx, self.nx, self.nq):
            if i <= j:
                df_dq[k, ..., j] +=  x[..., i] ** (k + j)
        return df_dq
    
    def d_dx(self, x, q):
        df_dx = np.zeros((self.nx,) + np.shape(x))
        for i, j, k in np.ndindex(self.nx, self.nx, self.nq):
            if i <= j and k + j != 0:
                df_dx[i, ..., j] += q[..., k] * (k + j) * x[..., i]**(k + j - 1)
        return df_dx


@pytest.fixture
def nlfunction(nx, nq):
    '''Parametrized nonlinear function.'''
    return NonlinearFunction(nx, nq)


def test_nlfunction_dx(nlfunction, x, q):
    '''Assert that nlfunction's derivative with respect to x is correct.'''
    numerical = utils.central_diff(lambda x: nlfunction(x, q), x)
    analytical = nlfunction.d_dx(x, q)
    assert ArrayCmp(analytical) == numerical


def test_nlfunction_dq(nlfunction, x, q):
    '''Assert that nlfunction's derivative with respect to p is correct.'''
    numerical = utils.central_diff(lambda q: nlfunction(x, q), q)
    analytical = nlfunction.d_dq(x, q)
    assert ArrayCmp(analytical) == numerical


def test_ut_sqrt(ut_sqrt_func, cov):
    '''Test if the ut_sqrt functions satisfy their definition.'''
    S = ut_sqrt_func(cov)
    STS = np.dot(S.T, S)
    assert ArrayCmp(STS) == cov


def test_cholesky_sqrt_diff(cov, nx):
    '''Check the derivative of the Cholesky decomposition.'''
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


def test_sigma_points_diff_wrt_mean(ut, ut_sqrt, x, cov, nx):
    '''Test the derivative of the unscented transform sigma points.'''
    if ut_sqrt == 'svd':
        pytest.skip("`svd_sqrt_diff` not implemented yet.")
    
    def sigma(mean, cov):
        return ut.gen_sigma_points(mean, cov)
    
    ds_dmean_num = utils.central_diff(lambda x: sigma(x, cov), x)
    
    sigma(x, cov)
    ds_dmean = ut.sigma_points_diff(np.identity(nx), np.zeros((nx, nx, nx)))
    assert ArrayCmp(ds_dmean_num) == ds_dmean


def test_sigma_points_diff_wrt_cov(ut, ut_sqrt, x, cov, nx):
    '''Test the derivative of the unscented transform sigma points.'''
    if ut_sqrt == 'svd':
        pytest.skip("`svd_sqrt_diff` not implemented yet.")
    
    def sigma(cij):
        Q = cov.copy()
        Q[i[k], j[k]] = cij
        Q[j[k], i[k]] = cij
        return ut.gen_sigma_points(x, Q)

    ut.gen_sigma_points(x, cov)
    i, j = np.tril_indices(nx)
    ntril = len(i)
        
    cov_diff = np.zeros((ntril, nx, nx))
    cov_diff[np.arange(ntril), i, j] = 1
    cov_diff[np.arange(ntril), j, i] = 1
    dsigma_dcov = ut.sigma_points_diff(np.zeros((ntril, nx)), cov_diff)
    
    for k in range(ntril):
        dsigma_dcij_num = utils.central_diff(lambda cij: sigma(cij), 
                                             cov[i[k], j[k]])
        assert ArrayCmp(dsigma_dcov[k]) == dsigma_dcij_num


def test_transform_diff_wrt_q(ut, ut_sqrt, nlfunction, x, q, cov, nx, nq):
    '''Test the derivatives of unscented transform.'''
    if ut_sqrt == 'svd':
        pytest.skip("`svd_sqrt_diff` not implemented yet.")
    
    pytest.skip()
    
    def ut_mean(q):
        return ut.unscented_transform(lambda x: nlfunction(x, q), x, cov)[0]
    
    def ut_cov(q):
        return ut.unscented_transform(lambda x: nlfunction(x, q), x, cov)[1]
    
    num_mean_diff = utils.central_diff(ut_mean, q)
    num_cov_diff = utils.central_diff(ut_cov, q)
    
    def f_diff(x, dx):
        return nlfunction.d_dq(x, q)
    
    in_mean_diff = np.zeros((nq, nx))
    in_cov_diff = np.zeros((nq, nx, nx))
    ut.unscented_transform(lambda x: nlfunction(x, q), x, cov)
    mean_diff, cov_diff = ut.transform_diff(f_diff, in_mean_diff, in_cov_diff)
    
    assert ArrayCmp(mean_diff) == num_mean_diff
    assert ArrayCmp(cov_diff) == num_cov_diff


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
