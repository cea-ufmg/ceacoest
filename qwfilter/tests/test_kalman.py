'''Kalman filtering / smoothing test module.'''


import numpy as np
import numpy.testing
import pytest

from .. import kalman


@pytest.fixture(params=range(3))
def seed(request):
    '''Random number generator seed.'''
    np.random.seed(request.param)
    return request.param


@pytest.fixture
def vec_4(seed):
    '''Random length 4 vector.'''
    return np.random.randn(4)


@pytest.fixture
def cov_4(seed):
    '''Random 4x4 positive definite symmetric matrix.'''
    A = np.random.randn(4, 10)
    return A.dot(A.T)


@pytest.fixture
def mat_4(seed):
    '''Random 4x4 matrix.'''
    A = np.random.randn(4, 4)
    return A


@pytest.fixture(params=['cholesky', 'svd'])
def ut_sqrt(request):
    '''Unscented transform square root option.'''
    return request.param


@pytest.fixture(params=[0, 0.5, 1])
def ut_kappa(request):
    '''Unscented transform kappa parameter.'''
    return request.param


@pytest.fixture
def ut(ut_sqrt, ut_kappa):
    '''Standalone UnscentedTransform object.'''
    return kalman.UnscentedTransform(sqrt=ut_sqrt, kappa=ut_kappa)


def test_sigma_points(ut, vec_4, cov_4):
    '''Test if the mean and covariance of the sigma-points is sane.'''
    [sigma, weights] = ut.gen_sigma_points(vec_4, cov_4)
    ut_mean = sigma.dot(weights)
    np.testing.assert_allclose(ut_mean, vec_4)

    dev = sigma - ut_mean[:, None]
    ut_cov = np.einsum('ik,jk,k', dev, dev, weights)
    np.testing.assert_allclose(ut_cov, cov_4)


def test_linear_ut(ut, vec_4, cov_4, mat_4):
    '''Test the unscented transform of a linear function.'''
    f = lambda x: mat_4.dot(x) + 1
    [ut_mean, ut_cov] = ut.unscented_transform(f, vec_4, cov_4)
    
    desired_mean = f(vec_4)
    np.testing.assert_allclose(ut_mean, desired_mean)

    desired_cov = mat_4.dot(cov_4).dot(mat_4.T)
    np.testing.assert_allclose(ut_cov, desired_cov)
    
    ut_xcov = ut.transform_xcov()
    desired_xcov = cov_4.dot(mat_4.T)
    np.testing.assert_allclose(ut_xcov, desired_xcov)


class EulerDiscretizedAtmosphericReentry:
    nx = 5
    nu = 0
    nw = 2
    ny = 2
    
    w_cov = np.diag([2.4064e-5, 2.4064e-5])
    v_cov = np.diag([0.017, 0.001]) ** 2
    
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
    x[0] = [6500.4, 349.14, -1.8093, -6.7967, 0.6932]
    w = np.random.multivariate_normal([0, 0], model.w_cov, N)
    
    for k in range(N-1):
        x[k+1] = model.f(k, x[k], [], w[k])
    
    v = np.random.multivariate_normal([0, 0], model.w_cov, N)
    y = model.h(None, x.T).T + v
    
    return [x, y, model]


if __name__ == '__main__':    
    [x_sim, y, model] = sim()
    
    x0_mean = [6500.4, 349.14, -1.8093, -6.7967, 0]
    x0_cov = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1])
    filter = kalman.DTUnscentedKalmanFilter(model, x0_mean, x0_cov)
    filter.filter(y)

