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
from qwfilter.testsupport.array_cmp import ArrayDiff


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


@pytest.fixture(params=[(), (3,), (2, 3, 1, 1)], scope='module')
def base_shape(request):
    """Base shape of broadcasting."""
    return request.param


@pytest.fixture
def x(seed, nx, base_shape):
    """Random state vector."""
    x_shape = base_shape + (nx,)
    return np.random.randn(*x_shape)


@pytest.fixture
def q(seed, nq):
    """Random parameter vector."""
    return np.random.randn(nq)


@pytest.fixture
def y(seed):
    """Random measurement vector."""
    return np.random.randn(2)


@pytest.fixture
def cov(seed, nx, base_shape):
    """Random x covariance matrix."""
    M_shape = base_shape + (nx, nx + 1)
    M = np.random.randn(*M_shape) / nx
    return np.einsum('...ik,...jk', M, M)


@pytest.fixture
def A(seed, nx):
    """Random state transition matrix."""
    A = np.random.randn(nx, nx)
    return A


@pytest.fixture(params=[0, 0.5, 1], scope='module')
def ut_kappa(request):
    """Unscented transform kappa parameter."""
    return request.param


@pytest.fixture(params=['cholesky', 'svd'], scope='module')
def ut_sqrt(request):
    """Unscented transform square root option."""
    return request.param


@pytest.fixture
def ut(ut_sqrt, ut_kappa, nx):
    """Standalone UnscentedTransform object."""
    options = {'sqrt': ut_sqrt, 'kappa': ut_kappa}
    UTClass = kalman.choose_ut_transform_class(options)
    return UTClass(nx, **options)


@pytest.fixture(scope='module')
def model_class(nx, nq):
    '''Discrete-time test model.'''
    
    class SymbolicModel(sym2num.SymbolicModel):
        var_names = {'k', 'x', 'q', 'Px'}
        '''Name of model variables.'''
        
        function_names = {'f', 'h', 'Q', 'R', 'v', 'Pv'}
        '''Name of the model functions.'''
        
        derivatives = [('df_dx', 'f', 'x'), ('df_dq', 'f', 'q'),
                       ('d2f_dx2', 'df_dx',  'x'), 
                       ('d2f_dx_dq', 'df_dx', 'q'),
                       ('d2f_dq2', 'df_dq',  'q'),
                       ('dQ_dx', 'Q', 'x'), ('dQ_dq', 'Q', 'q'),
                       ('d2Q_dx2', 'dQ_dx',  'x'), 
                       ('d2Q_dx_dq', 'dQ_dx', 'q'),
                       ('d2Q_dq2', 'dQ_dq',  'q'),
                       ('dh_dx', 'h', 'x'), ('dh_dq', 'h', 'q'),
                       ('d2h_dx2', 'dh_dx',  'x'), 
                       ('d2h_dx_dq', 'dh_dx', 'q'),
                       ('d2h_dq2', 'dh_dq',  'q'),
                       ('dR_dq', 'R', 'q'), ('d2R_dq2', 'dR_dq', 'q'),
                       ('dv_dq', 'v', 'q'), ('d2v_dq2', 'dv_dq', 'q'),
                       ('dPv_dq', 'Pv', 'q'), ('d2Pv_dq2', 'dPv_dq', 'q')]
        '''List of the model function derivatives to calculate / generate.'''
        
        k = 'k'
        '''Discretized sample index.'''
        
        x = ['x%d' % i for i in range(nx)]
        '''State vector.'''
    
        q = ['q%d' % i for i in range(nq)]
        '''State vector.'''

        Px = [['Px%d_%d' % (i, j) for j in range(nx)] for i in range(nx)]
        '''State covariance matrix.'''

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
        
        def R(self, q, k):
            '''Measurement function.'''
            return [[q[-1]**2 + 0.2, 0.01*q[0]], [0.01*q[0], 1]]

        def v(self, q, x):
            '''Parameter dependent state vector.'''
            return [q[i % nq] for i in range(nx)]
        
        def Pv(self, q, Px):
            '''Parameter dependent state covariance.'''
            Pv = Px.copy()
            for i, j in np.ndindex(nx, nx):
                if i == j:
                    Pv[i, j] += 1e-2 * q[(i + j) % nq]**2
                else:
                    Pv[i, j] += q[(i + j) % nq] * 1e-3
            return Pv
    
    ModelClass = sym2num.class_obj(
        SymbolicModel(), sym2num.ScipyPrinter(),
        name='GeneratedModel', meta=sym2num.ParametrizedModel.meta
    )
    ModelClass.nx = nx
    ModelClass.nq = nq
    ModelClass.ny = 2
    return ModelClass


@pytest.fixture
def model(model_class, x, q, cov):
    defaults = dict(k=0, x=x, q=q, Px=cov)
    return model_class(defaults)


@pytest.fixture
def parametrized_ukf(model, ut_kappa, ut_sqrt):
    def factory(q):
        mq = model.parametrize(q=q)
        ukf = kalman.DTUnscentedKalmanFilter(
            mq, mq.v(), mq.Pv(), kappa=ut_kappa, sqrt=ut_sqrt
        )
        ukf.dx_dq = model.dv_dq()
        ukf.dPx_dq = model.dPv_dq()
        return ukf
    return factory


def test_ut_sqrt(ut, cov):
    """Test if the ut_sqrt functions satisfy their definition."""
    S = ut.sqrt(cov)
    STS = np.einsum('...ki,...kj', S, S)
    assert ArrayDiff(STS, cov) < 1e-8


def test_ut_sqrt_diff(ut, model, q):
    """Check the derivative of the unscented transform square root."""
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("Square-root derivative not implemented yet.")
    
    def S(q):
        return ut.sqrt(model.Pv(q))
    numerical = utils.central_diff(S, q)
    
    ut.sqrt(model.Pv(q))
    analytical = ut.sqrt_diff(model.dPv_dq(q))
    analytical = np.rollaxis(analytical, -3)
    assert ArrayDiff(numerical, analytical) < 1e-8


def test_ut_sqrt_diff2(ut, model, q):
    """Check the derivative of the unscented transform square root."""
    if not hasattr(ut, 'sqrt_diff2'):
        pytest.skip("Square-root derivative not implemented yet.")
    
    def dS_dq(q):
        ut.sqrt(model.Pv(q))
        return ut.sqrt_diff(model.dPv_dq(q))
    numerical = utils.central_diff(dS_dq, q)
    
    ut.sqrt(model.Pv(q))
    ut.sqrt_diff(model.dPv_dq(q))
    analytical = ut.sqrt_diff2(model.d2Pv_dq2(q))
    analytical = np.rollaxis(analytical, -4)
    assert ArrayDiff(numerical, analytical) < 1e-8


def test_affine_ut(ut, x, cov, A, nx):
    """Test the unscented transform of an affine function."""
    f = lambda x: np.einsum('ij,...j', A, x) + np.arange(nx)
    [ut_mean, ut_cov] = ut.transform(x, cov, f)
    
    desired_mean = f(x)
    assert ArrayDiff(ut_mean, desired_mean) < 1e-8
    
    desired_cov = np.einsum('...ij,ki,lj', cov, A, A)
    #desired_cov = A.dot(cov).dot(A.T)
    assert ArrayDiff(ut_cov, desired_cov) < 1e-8
    
    ut_crosscov = ut.crosscov()
    desired_crosscov = np.dot(cov, A.T)
    assert ArrayDiff(ut_crosscov, desired_crosscov) < 1e-8


def test_sigma_points(ut, x, cov):
    """Test if the mean and covariance of the sigma-points is sane."""
    sigma = ut.sigma_points(x, cov)
    ut_mean = np.einsum('k,k...', ut.weights, sigma)
    assert ArrayDiff(ut_mean, x) < 1e-8
    
    ut_cov = np.einsum('k...i,k...j,k', ut.idev, ut.idev, ut.weights)
    assert ArrayDiff(ut_cov, cov) < 1e-8


def test_sigma_points_diff(ut, model, q):
    """Test the derivative of the unscented transform sigma points."""
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("Square-root derivative not implemented yet.")
    
    def sigma(q):
        return ut.sigma_points(model.v(q), model.Pv(q))
    numerical = utils.central_diff(sigma, q)
    numerical = np.rollaxis(numerical, -2, -1)
    
    ut.sigma_points(model.v(), model.Pv())
    analytical = ut.sigma_points_diff(model.dv_dq(), model.dPv_dq())
    analytical = np.rollaxis(analytical, -2)
    assert ArrayDiff(numerical, analytical) < 5e-8


def test_sigma_points_diff2(ut, model, q):
    """Test the derivative of the unscented transform sigma points."""
    if not hasattr(ut, 'sqrt_diff2'):
        pytest.skip("Square-root derivative not implemented yet.")
    
    def dsigma_dq(q):
        ut.sigma_points(model.v(q), model.Pv(q))
        return ut.sigma_points_diff(model.dv_dq(q), model.dPv_dq(q))
    numerical = utils.central_diff(dsigma_dq, q)
    numerical = np.rollaxis(numerical, -3)
    
    ut.sigma_points(model.v(), model.Pv())
    ut.sigma_points_diff(model.dv_dq(), model.dPv_dq())
    analytical = ut.sigma_points_diff2(model.d2v_dq2(), model.d2Pv_dq2())
    assert ArrayDiff(numerical, analytical) < 1e-8


def test_ut_diff(ut, model, x, q):
    """Test the derivative of the unscented transform sigma points."""
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("Square-root derivative not implemented yet.")
    
    def df_dx(x):
        return model.df_dx(x=x)
    def df_dq(x):
        return model.df_dq(x=x)
    
    def transform(q):
        mq = model.parametrize(q=q)
        out = ut.transform(mq.v(), mq.Pv(), lambda x: mq.f(x=x))
        return out + (ut.crosscov(),)
    numerical_x = utils.central_diff(lambda q: transform(q)[0], q)
    numerical_Px = utils.central_diff(lambda q: transform(q)[1], q)
    numerical_Pio = utils.central_diff(lambda q: transform(q)[2], q)
    
    ut.transform(model.v(), model.Pv(), lambda x: model.f(x=x))
    analytical_x, analytical_Px = ut.transform_diff(
        df_dq, df_dx, model.dv_dq(), model.dPv_dq()
    )
    analytical_Pio = ut.crosscov_diff()
    assert ArrayDiff(numerical_x, analytical_x) < 1e-8
    assert ArrayDiff(numerical_Px, analytical_Px) < 1e-8
    assert ArrayDiff(numerical_Pio, analytical_Pio) < 1e-8


def test_ut_diff2(ut, model, x, q):
    """Test the derivative of the unscented transform sigma points."""
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("Square-root derivative not implemented yet.")
    
    def df_dx(x):
        return model.df_dx(x=x)
    def df_dq(x):
        return model.df_dq(x=x)
    def d2f_dq2(x):
        return model.d2f_dq2(x=x)
    def d2f_dx2(x):
        return model.d2f_dx2(x=x)
    def d2f_dx_dq(x):
        return model.d2f_dx_dq(x=x)
    
    def transform(q):
        mq = model.parametrize(q=q)
        ut.transform(mq.v(), mq.Pv(), lambda x: mq.f(x=x))
        ut.crosscov()
        diff_out = ut.transform_diff(
            lambda x: mq.df_dq(x=x), lambda x: mq.df_dx(x=x), 
            mq.dv_dq(), mq.dPv_dq()
        )
        return diff_out + (ut.crosscov_diff(),)
    numerical_x = utils.central_diff(lambda q: transform(q)[0], q)
    numerical_Px = utils.central_diff(lambda q: transform(q)[1], q)
    numerical_Pio = utils.central_diff(lambda q: transform(q)[2], q)
    
    ut.transform(model.v(), model.Pv(), lambda x: model.f(x=x))
    ut.transform_diff(df_dq, df_dx, model.dv_dq(), model.dPv_dq())
    analytical_x, analytical_Px = ut.transform_diff2(
        d2f_dq2, d2f_dx2, d2f_dx_dq, model.d2v_dq2(), model.d2Pv_dq2()
    )
    analytical_Pio = ut.crosscov_diff2()
    assert ArrayDiff(numerical_x, analytical_x) < 1e-8
    assert ArrayDiff(numerical_Px, analytical_Px) < 5e-8
    assert ArrayDiff(numerical_Pio, analytical_Pio) < 5e-8


def test_ut_pred_diff(parametrized_ukf, ut, model, q):
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("UT square-root derivative not implemented yet.")

    def pred(q):
        ukf = parametrized_ukf(q)
        ukf.predict()
        return ukf
    numerical_x = utils.central_diff(lambda q: pred(q).x, q)
    numerical_Px = utils.central_diff(lambda q: pred(q).Px, q)

    ukf = parametrized_ukf(q)
    ukf.predict()
    ukf.prediction_diff()
    analytical_x = ukf.dx_dq
    analytical_Px = ukf.dPx_dq
    assert ArrayDiff(numerical_x, analytical_x) < 1e-8
    assert ArrayDiff(numerical_Px, analytical_Px) < 5e-8


def test_ut_pred_diff2(parametrized_ukf, ut, model, q):
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("UT square-root derivative not implemented yet.")

    def pred(q):
        ukf = parametrized_ukf(q)
        ukf.predict()
        ukf.prediction_diff()
        return ukf
    numerical_x = utils.central_diff(lambda q: pred(q).dx_dq, q)
    numerical_Px = utils.central_diff(lambda q: pred(q).dPx_dq, q)
    
    ukf = parametrized_ukf(q)
    ukf.predict()
    ukf.prediction_diff()
    ukf.prediction_diff2()
    analytical_x = ukf.d2x_dq2
    analytical_Px = ukf.d2Px_dq2
    assert ArrayDiff(numerical_x, analytical_x) < 1e-8
    assert ArrayDiff(numerical_Px, analytical_Px) < 1e-8


def test_ut_corr_diff(parametrized_ukf, ut, model, q, y):
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("UT square-root derivative not implemented yet.")
        
    def corr(q):
        ukf = parametrized_ukf(q)
        ukf.correct(y)
        ukf.update_likelihood()
        return ukf
    numerical_L = utils.central_diff(lambda q: corr(q).L, q)
    numerical_x = utils.central_diff(lambda q: corr(q).x, q)
    numerical_Px = utils.central_diff(lambda q: corr(q).Px, q)
    
    ukf = parametrized_ukf(q)
    ukf.correct(y)
    ukf.update_likelihood()
    ukf.correction_diff()
    ukf.likelihood_diff()
    analytical_L = ukf.dL_dq
    analytical_x = ukf.dx_dq
    analytical_Px = ukf.dPx_dq
    assert ArrayDiff(numerical_L, analytical_L) < 5e-8
    assert ArrayDiff(numerical_x, analytical_x) < 5e-8
    assert ArrayDiff(numerical_Px, analytical_Px) < 1e-7

def test_ut_corr_diff2(parametrized_ukf, ut, model, q, y):
    if not hasattr(ut, 'sqrt_diff'):
        pytest.skip("UT square-root derivative not implemented yet.")
        
    def corr(q):
        ukf = parametrized_ukf(q)
        ukf.correct(y)
        ukf.correction_diff()
        ukf.update_likelihood()
        ukf.likelihood_diff()
        return ukf
    numerical_L = utils.central_diff(lambda q: corr(q).dL_dq, q)
    numerical_x = utils.central_diff(lambda q: corr(q).dx_dq, q)
    numerical_Px = utils.central_diff(lambda q: corr(q).dPx_dq, q)
    
    ukf = parametrized_ukf(q)
    ukf.correct(y)
    ukf.update_likelihood()
    ukf.correction_diff()
    ukf.correction_diff2()
    ukf.likelihood_diff()
    ukf.likelihood_diff2()
    analytical_L = ukf.d2L_dq2
    analytical_x = ukf.d2x_dq2
    analytical_Px = ukf.d2Px_dq2
    assert ArrayDiff(numerical_L, analytical_L) < 1e-7
    assert ArrayDiff(numerical_x, analytical_x) < 1e-7
    assert ArrayDiff(numerical_Px, analytical_Px) < 1e-7
