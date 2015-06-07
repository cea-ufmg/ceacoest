"""Kalman filtering / smoothing test module.

TODO
----
 * Test filter vectorization.

"""


import numpy as np
import numpy.ma as ma
import numpy.testing
import pytest
import sympy
import sym2num

from qwfilter import kalman, utils


pytest_plugins = "qwfilter.testsupport.array_cmp"
from qwfilter.testsupport.array_cmp import ArrayCmp, ArrayDiff


@pytest.fixture(params=range(4))
def seed(request):
    """Random number generator seed."""
    np.random.seed(request.param)
    return request.param


@pytest.fixture(params=range(1, 5), scope='module')
def nx(request):
    """Number of states to test with."""
    return request.param


@pytest.fixture(params=range(1, 3), scope='module')
def nq(request):
    """Number of parameters to test with."""
    return request.param


@pytest.fixture
def x(seed, nx):
    """Random state vector."""
    return np.random.randn(nx)


@pytest.fixture
def q(seed, nq):
    """Random parameter vector."""
    return np.random.randn(nq)


@pytest.fixture
def cov(seed, nx):
    """Random x covariance matrix."""
    M = np.random.randn(nx, nx + 1) / nx
    return M.dot(M.T)


@pytest.fixture
def A(seed, nx):
    """Random state transition matrix."""
    A = np.random.randn(nx, nx)
    return A


@pytest.fixture(params=[0, 0.5, 1])
def ut_kappa(request):
    """Unscented transform kappa parameter."""
    return request.param


@pytest.fixture(params=['cholesky', 'svd'])
def ut_sqrt(request):
    """Unscented transform square root option."""
    return request.param


@pytest.fixture
def ut(ut_sqrt, ut_kappa, nx):
    """Standalone UnscentedTransform object."""
    options = {'sqrt': ut_sqrt, 'kappa': ut_kappa}
    UTClass = kalman.choose_ut_transform_class(options)
    return UTClass(nx, **options)


@pytest.fixture
def ut_work(x, cov, ut):
    """Unscented transform work data."""
    return ut.Work(x, cov)


@pytest.fixture(scope='module')
def parametrized_mean_cov(nx, x, cov):
    """Parametrized mean-covariance pair."""    
    i, j = np.tril_indices(nx)
    
    class ParametrizedMeanCovPair:
        nq = len(i) + nx
        q = np.zeros(nq)
        
        @staticmethod
        def x(q):
            return x + q[:nx]
        
        @staticmethod
        def Px(q):
            Px = cov.copy()
            Px[i, j] += q[nx:]
            Px[j, i] += q[nx:] * (i != j)
            return Px

        @classmethod
        def ut_work(cls, q):
            return kalman.UnscentedTransformWork(cls.x(q), cls.Px(q))
        
        dx_dq = np.zeros((nq, nx))
        dx_dq[np.arange(nx), np.arange(nx)] = 1

        dPx_dq = np.zeros((nq, nx, nx))
        dPx_dq[np.arange(nx, nq), i, j] = 1
        dPx_dq[np.arange(nx, nq), j, i] = 1
    
    return ParametrizedMeanCovPair
        

@pytest.fixture(scope='module')
def model(nx, nq):
    '''Discrete-time test model.'''
    
    class SymbolicModel(sym2num.SymbolicModel):
        var_names = {'k', 'x', 'q'}
        '''Name of model variables.'''
        
        function_names = {'f', 'h', 'Q', 'R'}
        '''Name of the model functions.'''
        
        derivatives = [('df_dx', 'f', 'x'), ('df_dq', 'f', 'q'),
                       ('d2f_dx2', 'df_dx',  'x'), 
                       ('d2f_dq_dx', 'df_dx', 'q'),
                       ('d2f_dq2', 'df_dq',  'q'),
                       ('dQ_dx', 'Q', 'x'), ('dQ_dq', 'Q', 'q'),
                       ('d2Q_dx2', 'dQ_dx',  'x'), 
                       ('d2Q_dq_dx', 'dQ_dx', 'q'),
                       ('d2Q_dq2', 'dQ_dq',  'q'),
                       ('dh_dx', 'h', 'x'), ('dh_dq', 'h', 'q'),
                       ('dR_dq', 'R', 'q'),]
        '''List of the model function derivatives to calculate / generate.'''
        
        k = 'k'
        '''Discretized sample index.'''
        
        x = ['x%d' % i for i in range(nx)]
        '''State vector.'''
    
        q = ['q%d' % i for i in range(nq)]
        '''State vector.'''
        
        def f(self, k, x, q):
            '''Drift function.'''
            ret = np.zeros(nx, dtype=object)
            for i, j in np.ndindex(nx, nq):
                if i >= j:
                    ret[i] = ret[i] + sympy.sin(i + j + x[i] + q[j])
            return ret
        
        def h(self, k, x, q):
            '''Measurement function.'''
            return [x[0] * q[0], x[-1] * x[0]]

        def Q(self, k, x, q):
            '''Measurement function.'''
            ret = np.eye(nx, dtype=object)
            ret[0, -1] = x[0]**2  * q[0]**2 * 1e-3
            ret[-1, 0] = ret[0, -1]
            ret[0, 0] = 1 + x[-1] ** 2 * q[-1] ** 2
            return ret
        
        def R(self, q):
            '''Measurement function.'''
            return [[q[-1]**2 + 0.2, 0.01*q[0]], [0.01*q[0], 1]]

    ModelClass = sym2num.class_obj(
        SymbolicModel(), sym2num.ScipyPrinter(),
        name='GeneratedModel', meta=sym2num.ParametrizedModel.meta
    )
    ModelClass.nx = nx
    ModelClass.nq = nq
    ModelClass.ny = 2
    defaults = dict(k=0, q=np.zeros(nq))
    return ModelClass(defaults)


def test_ut_sqrt(ut, ut_work, cov):
    """Test if the ut_sqrt functions satisfy their definition."""
    S = ut.sqrt(ut_work, cov)
    STS = np.dot(S.T, S)
    assert ArrayDiff(STS, cov) < 1e-8


def test_ut_sqrt_diff(ut, ut_work, parametrized_mean_cov, cov):
    """Check the derivative of the unscented transform square root."""
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("Square-root derivative not implemented yet.")
    
    def S(q):
        return ut.sqrt(ut_work, parametrized_mean_cov.Px(q))
    numerical = utils.central_diff(S, parametrized_mean_cov.q)
    
    ut.sqrt(ut_work, cov)
    analytical = ut.sqrt_diff(ut_work, parametrized_mean_cov.dPx_dq)
    assert ArrayDiff(numerical, analytical) < 1e-8


def test_sigma_points(ut, ut_work, x, cov):
    """Test if the mean and covariance of the sigma-points is sane."""
    sigma = ut.sigma_points(ut_work)
    ut_mean = np.dot(ut.weights, sigma)
    assert ArrayDiff(ut_mean, x) < 1e-8
    
    idev = ut_work.idev
    ut_cov = np.einsum('ki,kj,k', idev, idev, ut.weights)
    assert ArrayDiff(ut_cov, cov) < 1e-8


def test_affine_ut(ut, ut_work, x, cov, A, nx):
    """Test the unscented transform of an affine function."""
    f = lambda x: np.dot(x, A.T) + np.arange(nx)
    [ut_mean, ut_cov] = ut.transform(ut_work, f)
    
    desired_mean = f(x)
    assert ArrayDiff(ut_mean, desired_mean) < 1e-8

    desired_cov = A.dot(cov).dot(A.T)
    assert ArrayDiff(ut_cov, desired_cov) < 1e-8
    
    ut_crosscov = ut.crosscov(ut_work)
    desired_crosscov = np.dot(cov, A.T)
    assert ArrayDiff(ut_crosscov, desired_crosscov) < 1e-8


def test_sigma_points_diff(ut, ut_work, parametrized_mean_cov):
    """Test the derivative of the unscented transform sigma points."""
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("Square-root derivative not implemented yet.")
    
    def sigma(q):
        return ut.sigma_points(parametrized_mean_cov.ut_work(q))
    numerical = utils.central_diff(sigma, parametrized_mean_cov.q)
    numerical = np.rollaxis(numerical, -2)
    
    ut.sigma_points(ut_work)
    dx_dq = parametrized_mean_cov.dx_dq
    dPx_dq = parametrized_mean_cov.dPx_dq
    analytical = ut.sigma_points_diff(ut_work, dx_dq, dPx_dq)
    assert ArrayDiff(numerical, analytical) < 1e-8


def test_ut_diff_x(ut, ut_work, parametrized_mean_cov, model):
    """Test the derivative of the unscented transform sigma points."""
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("Square-root derivative not implemented yet.")
    if model.nq > 1:
        pytest.skip("Test does not use depend on model parameters.")
    
    def f(x):
        return model.f(x=x)
    def df_dx(x):
        return model.df_dx(x=x)
    def df_dq(x):
        return np.zeros((parametrized_mean_cov.nq, model.nx))

    def ut_x(q):
        return ut.transform(parametrized_mean_cov.ut_work(q), f)[0]
    def ut_Px(q):
        return ut.transform(parametrized_mean_cov.ut_work(q), f)[1]
    numerical_x = utils.central_diff(ut_x, parametrized_mean_cov.q)
    numerical_Px = utils.central_diff(ut_Px, parametrized_mean_cov.q)
    
    ut.transform(ut_work, f)
    dx_dq = parametrized_mean_cov.dx_dq
    dPx_dq = parametrized_mean_cov.dPx_dq
    analytical_x, analytical_Px = ut.transform_diff(
        ut_work, df_dq, df_dx, dx_dq, dPx_dq
    )
    assert ArrayDiff(numerical_x, analytical_x) < 1e-8
    assert ArrayDiff(numerical_Px, analytical_Px) < 1e-8


def test_ut_diff_q(ut, ut_work, model, q, nx, nq):
    """Test the derivative of the unscented transform sigma points."""
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("Square-root derivative not implemented yet.")
    
    def ut_x(q):
        return ut.transform(ut_work, lambda x: model.f(x=x, q=q))[0]
    def ut_Px(q):
        return ut.transform(ut_work, lambda x: model.f(x=x, q=q))[1]
    numerical_x = utils.central_diff(ut_x, q)
    numerical_Px = utils.central_diff(ut_Px, q)
    
    def df_dq(x):
        return model.df_dq(x=x, q=q)
    def df_dx(x):
        return model.df_dx(x=x, q=q)
    ut.transform(ut_work, lambda x: model.f(x=x, q=q))
    dx_dq = np.zeros((nq, nx))
    dPx_dq = np.zeros((nq, nx, nx))
    analytical_x, analytical_Px = ut.transform_diff(
        ut_work, df_dq, df_dx, dx_dq, dPx_dq
    )
    assert ArrayDiff(numerical_x, analytical_x) < 1e-8
    assert ArrayDiff(numerical_Px, analytical_Px) < 1e-8

