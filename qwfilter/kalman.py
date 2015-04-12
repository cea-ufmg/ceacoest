'''Kalman filtering / smoothing module.

TODO
----
 * Add derivative of SVD square root.
 * Vectorize cholesky_sqrt_diff, sigma_points_diff and transform_diff.
 * Make docstrings for all constructors.

Improvement ideas
-----------------
 * Make the unscented transform object a member variable of the predictor
   instead of superclass, so that it is not shared among prediction and
   correction. Weights can also be generated in the constructor.
 * Since the transition noise is independent from the states, the unscented
   transform for the prediction can be devided in two simpler ones.

'''


import abc
import collections
import re

import numpy as np
import numpy.ma as ma
import numpy.linalg
import scipy.linalg


class DTKalmanFilterBase(metaclass=abc.ABCMeta):
    '''Discrete-time Kalman filter/smoother abstract base class.'''
    
    def __init__(self, model, mean, cov, **options):
        # Save check and initalize basic filter data
        self.model = model
        '''The underlying system model.'''
        
        self.mean = np.asarray(mean, dtype=float)
        '''The latest filtered state mean.'''
        assert self.mean.shape[-1] == model.nx
        
        self.cov = np.asarray(cov, float)
        '''The latest filtered state covariance.'''
        assert self.cov.shape == self.mean.shape + (model.nx,)

        self.base_shape = self.mean.shape[:-1]
        '''Shape of scalar element for filter vectorization.'''
                
        self.t = 0
        '''The latest filtered time sample.'''
        
        # Process the keyword options
        self.initialize_options(options)
    
    def initialize_options(self, options):
        '''Initialize and validate filter options, given as a dictionary.
        
        Options
        -------
        save_likelihood :
            Whether to save the measurement likelihood. True by default.
        calculate_pred_crosscov :
            Whether to calculate the prediction cross-covariance matrices. 
            False by default.
        calculate_gradients :
            Whether to calculate the filter gradients with respect to the
            parameter vector q. False by default.
        initial_mean_grad : (np, ..., nx) array_like
            Gradient of mean of inital states. Zero by default.
        initial_cov_grad : (np, ..., nx, nx) array_like
            Gradient of covariance of inital states. Zero by default.
        save_history : int or 'filter'
            Wether to save the filter history or not. If it is an int then
            it specifies the history size. Otherwise the size is defined by
            the filter function. Zero by default.
        t0 : float
            Initial time, zero by default.
        
        '''
        # Get the base options
        self.t = options.get('t0', 0)
        self.save_history = options.get('save_history', 0)
        self.save_likelihood = options.get('save_likelihood', True)
        self.calculate_gradients = options.get('calculate_gradients', False)
        self.calculate_pred_crosscov = options.get('calculate_pred_crosscov',
                                                   False)
        # Initialize the variables associtated to the options
        if self.save_likelihood:
            self.loglikelihood = 0
        if self.calculate_gradients:
            nq = self.model.nq
            default_mean_grad = np.zeros((nq,) + self.mean.shape)
            default_cov_grad = np.zeros((nq,) + self.cov.shape)
            self.mean_grad = options.get('initial_mean_grad', default_mean_grad)
            self.cov_grad = options.get('initial_cov_grad', default_cov_grad)
            self.loglikelihood_grad = np.zeros((nq,) + self.mean.shape[:-1])
        if self.save_history == 'filter':
            self.initialize_history(0)
        else:
            self.initialize_history(self.save_history)
    
    def initialize_history(self, size):
        self.history_size = size
        self.history_ind = 0
        if self.history_size:
            # Allocate the history vectors
            self.pred_mean = np.zeros((self.history_size,) + self.mean.shape)
            self.pred_cov = np.zeros((self.history_size,) + self.cov.shape)
            self.corr_mean = np.zeros((self.history_size,) + self.mean.shape)
            self.corr_cov = np.zeros((self.history_size,) + self.cov.shape)
            if self.calculate_pred_crosscov:
                self.pred_crosscov = np.zeros_like(self.corr_cov)
            # Initialize the history variables
            self.pred_mean[0] = self.mean
            self.corr_mean[0] = self.mean
            self.pred_cov[0] = self.cov
            self.corr_cov[0] = self.cov
    
    def _save_prediction(self, mean, cov):
        '''Save the prection data.'''
        self.mean = mean
        self.cov = cov
        if self.save_history and self.history_ind + 1 < self.history_size:
            k = self.history_ind = self.history_ind + 1
            self.pred_mean[k] = mean
            self.pred_cov[k] = cov
            self.corr_mean[k] = mean
            self.corr_cov[k] = cov
    
    def _save_correction(self, mean, cov):
        '''Save the correction data.'''
        self.mean = mean
        self.cov = cov
        if self.save_history and self.history_ind < self.history_size:
            k = self.history_ind
            self.corr_mean[k] = mean
            self.corr_cov[k] = cov
    
    def _save_prediction_crosscov(self, crosscov):
        '''Save the prediction cross-covariance.'''
        if self.save_history and self.history_ind < self.history_size:
            self.pred_crosscov[self.history_ind] = crosscov
    
    @abc.abstractmethod
    def predict(self, t):
        '''Predict the state distribution at a next time sample.'''
        raise NotImplementedError("Pure abstract method.")
    
    @abc.abstractmethod
    def correct(self, y):
        '''Correct the state distribution, given the measurement vector.'''
        raise NotImplementedError("Pure abstract method.")
    
    def filter(self, t, y):
        t = np.asarray(t)
        y = np.asanyarray(y)
        
        if self.t > t[0]:
            raise ValueError("Given times start before initial filter time.")
        
        if self.save_history == 'filter':
            self.initialize_history(len(t))
        
        if self.t == t[0]:
            self.correct(y[0])
            y = y[1:]
            t = t[1:]
        
        for tk, yk in zip(t, y):
            self.predict(tk)
            self.correct(yk)

    
def svd_sqrt(mat):
    '''SVD-based square root of a symmetric positive-semidefinite matrix.
    
    Used for unscented transform.
    
    Example
    -------
    Generate a random positive-semidefinite symmetric matrix.
    >>> np.random.seed(0)
    >>> A = np.random.randn(4, 10)
    >>> Q = np.dot(A, A.T)
    
    The square root should satisfy S'S = Q
    >>> S = svd_sqrt(Q)
    >>> STS = np.dot(S.T, S)
    >>> np.testing.assert_allclose(Q, STS)
    
    '''
    [U, s, Vh] = numpy.linalg.svd(mat)
    return np.swapaxes(U * np.sqrt(s), -1, -2)


def cholesky_sqrt(mat):
    '''Upper triangular Cholesky decomposition for unscented transform.'''
    lower_chol = numpy.linalg.cholesky(mat)
    return np.swapaxes(lower_chol, -1, -2)


def cholesky_sqrt_diff(S, dQ=None):
    '''Derivatives of lower triangular Cholesky decomposition.
    
    Parameters
    ----------
    S : (n, n) array_like
        The upper triangular Cholesky decomposition of a matrix `Q`, i.e.,
        `S.T * S.T == Q`.
    dQ : (..., n, n) array_like or None
        The derivatives of `Q` with respect to some parameters. Must be
        symmetric with respect to the last two axes, i.e., 
        `dQ[...,i,j] == dQ[...,j,i]`. If `dQ` is `None` then the derivatives
        are taken with respect to `Q`, i.e., `dQ[i,j,i,j] = 1` and
        `dQ[i,j,j,i] = 1`.
    
    Returns
    -------
    dS : (..., n, n) array_like
        The derivative of `S` with respect to some parameters or with respect
        to `Q` if `dQ` is `None`.
    
    '''
    S = np.asarray(S)
    
    n = S.shape[-1]
    k = np.arange(n)
    i, j = np.tril_indices(n)
    ix, jx, kx = np.ix_(i, j, k)
    
    A = np.zeros((n, n, n, n))
    A[ix, jx, ix, kx] = S[kx, jx]
    A[ix, jx, jx, kx] += S[kx, ix]
    A_tril = A[i, j][..., i, j]
    A_tril_inv = scipy.linalg.inv(A_tril)
    
    if dQ is None:
        nnz = len(i)
        dQ_tril = np.zeros((n, n, nnz))
        dQ_tril[i, j, np.arange(nnz)] = 1
        dQ_tril[j, i, np.arange(nnz)] = 1
    else:
        dQ_tril = dQ[..., i, j]
    
    D_tril = np.einsum('ab,...b->...a', A_tril_inv, dQ_tril)
    D = np.zeros(dQ_tril.shape[:-1] + (n, n))
    D[..., j, i] = D_tril
    return D


cholesky_sqrt.diff = cholesky_sqrt_diff


class UnscentedTransform:
    
    def __init__(self, nin, **options):
        '''Unscented transform object constructor.
        
        Parameters
        ----------
        nin : int
            Number of inputs
        
        Options
        -------
        kappa :
            Weight of the center sigma point. Zero by default.
        sqrt : str or callable
            Matrix "square root" used to generate sigma points. If equal to
            'svd' or 'cholesky' then svd_sqrt or cholesky_sqrt are used,
            respectively. Otherwise, if it is a callable the object is used.
            Equal to 'cholesky' by default.
        
        '''
        self.nin = nin
        '''Number of inputs.'''
        
        self.kappa = options.get('kappa', 0.0)
        '''Weight of the center sigma point.'''
        assert self.nin + self.kappa != 0
        
        sqrt_opt = options.get('sqrt', 'cholesky')
        if sqrt_opt == 'cholesky':
            sqrt = cholesky_sqrt
        elif sqrt_opt == 'svd':
            sqrt = svd_sqrt
        elif isinstance(sqrt_opt, collections.Callable):
            sqrt = sqrt_opt
        else:
            raise ValueError("Invalid value for `sqrt` option.")
        self.sqrt = sqrt
        '''Unscented transform square root method.'''
        
        self.nsigma = 2 * nin + (self.kappa != 0)
        '''Number of sigma points.'''
        
        self.weights = np.repeat(0.5 / (nin + self.kappa), self.nsigma)
        '''Transform weights.'''
        if self.kappa != 0:
            self.weights[-1] = self.kappa / (nin + self.kappa)
    
    def gen_sigma_points(self, mean, cov):
        '''Generate sigma-points and their deviations.'''
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        kappa = self.kappa
        nin = self.nin
        
        cov_sqrt = self.sqrt((nin + kappa) * cov)
        self.in_dev = np.zeros((self.nsigma,) + mean.shape)
        self.in_dev[:nin] = cov_sqrt
        self.in_dev[nin:(2 * nin)] = -cov_sqrt
        self.in_sigma = self.in_dev + mean
        return self.in_sigma
    
    def transform(self, f, mean, cov):
        in_sigma = self.gen_sigma_points(mean, cov)
        weights = self.weights
        
        out_sigma = f(in_sigma)
        out_mean = np.einsum('k,k...', weights, out_sigma)
        out_dev = out_sigma - out_mean
        out_cov = np.einsum('k...i,k...j,k->...ij', out_dev, out_dev, weights)
        
        self.out_dev = out_dev
        return (out_mean, out_cov)
    
    def crosscov(self):
        weights = self.weights
        try:
            in_dev = self.in_dev
            out_dev = self.out_dev
        except AttributeError:
            msg = "Transform must be done before requesting crosscov."
            raise RuntimeError(msg)
        
        return np.einsum('k...i,k...j,k->...ij', in_dev, out_dev, weights)
    
    def sigma_points_diff(self, mean_diff, cov_diff):
        '''Derivative of sigma-points.'''
        try:
            in_dev = self.in_dev
        except AttributeError:
            msg = "Transform must be done before requesting derivatives."
            raise RuntimeError(msg)
        
        nin = self.nin
        nq = len(mean_diff)
        kappa = self.kappa
        
        cov_sqrt = in_dev[:nin]
        cov_sqrt_diff = self.sqrt.diff(cov_sqrt, (nin + kappa) * cov_diff)
        dev_diff = np.zeros((self.nsigma,) + mean_diff.shape)
        dev_diff[:nin] = np.rollaxis(cov_sqrt_diff, -2)
        dev_diff[nin:(2 * nin)] = -dev_diff[:nin]
        in_sigma_diff = dev_diff + mean_diff
        return in_sigma_diff
    
    def transform_diff(self, f_diff, mean_diff, cov_diff, crosscov=False):
        weights = self.weights
        try:
            in_dev = self.in_dev
            out_dev = self.out_dev
            in_sigma = self.in_sigma
        except AttributeError:
            msg = "Transform must be done before requesting derivatives."
            raise RuntimeError(msg)
        
        in_sigma_diff = self.sigma_points_diff(mean_diff, cov_diff)
        out_sigma_diff = f_diff(in_sigma, in_sigma_diff)
        out_mean_diff = np.einsum('k,k...', weights, out_sigma_diff)
        out_dev_diff = out_sigma_diff - out_mean_diff
        out_cov_diff = np.einsum('k...i,k...j,k->...ij',
                                 out_dev_diff, out_dev, weights)
        out_cov_diff += np.einsum('k...i,k...j,k->...ij',
                                  out_dev, out_dev_diff, weights)
        if crosscov:
            in_dev_diff = in_sigma_diff - mean_diff ##### Inefficient
            crosscov_diff = np.einsum('k...qi,k...j,k->...qij',
                                      in_dev_diff, out_dev, weights)
            crosscov_diff += np.einsum('k...i,k...qj,k->...qij',
                                       in_dev, out_dev_diff, weights)
            return (out_mean_diff, out_cov_diff, crosscov_diff)
        else:
            return (out_mean_diff, out_cov_diff)


class DTUnscentedPredictor(DTKalmanFilterBase):

    def __init__(self, model, mean, cov, **options):
        super().__init__(model, mean, cov, **options)

        ut_options = options.copy()
        for key, val in options.items():
            match = re.match('pred_ut_(?P<subkey>\w+)', key)
            if match:
                ut_options[match.group('subkey')] = val
        
        self.__ut = UnscentedTransform(model.nx, **ut_options)
    
    def predict(self, t):
        '''Predict the state distribution at the next time index.'''
        td = np.array([self.t, t])
        def trans(x):
            return self.model.fd(td, x)
        
        pred_mean, pred_cov = self.__ut.transform(trans, self.mean, self.cov)
        pred_cov += self.model.Qd(td, self.mean)
        self._save_prediction(pred_mean, pred_cov)
        
        if self.calculate_pred_crosscov:
            pred_crosscov = self.crosscov()
            self._save_prediction_crosscov(pred_crosscov)
        
        if self.calculate_gradients:
            self._calculate_prediction_grad(t)
                
        self.t = t
    
    def _calculate_prediction_grad(self, t):
        td = np.array((self.t, t))
        x = self.mean
        dx_dq = self.mean_grad

        dQd_dq = self.model.dQd_dq(td, x)
        dQd_dx = self.model.dQd_dx(td, x)
        DQd_Dq = dQd_dq + np.einsum('...ij,...jkl', dx_dq, dQd_dx)
        
        def Dfd_Dq(x, dx_dq):
            dfd_dq = self.model.dfd_dq(td, x)
            dfd_dx = self.model.dfd_dx(td, x)
            return dfd_dq + np.einsum('...qx,...xf->...qf', dx_dq, dfd_dx)
        ut_grads = self.__ut.transform_diff(Dfd_Dq, dx_dq, self.cov_grad)
        
        self.mean_grad = ut_grads[0]
        self.cov_grad = ut_grads[1] + DQd_Dq


class DTUnscentedCorrector(DTKalmanFilterBase):

    def __init__(self, model, mean, cov, **options):
        super().__init__(model, mean, cov, **options)
        
        ut_options = options.copy()
        for key, val in options.items():
            match = re.match('corr_ut_(?P<subkey>\w+)', key)
            if match:
                ut_options[match.group('subkey')] = val
        
        self.__ut = UnscentedTransform(model.nx, **ut_options)
    
    def correct(self, y):
        '''Correct the state distribution, given the measurement vector.'''
        assert np.shape(y) == (self.model.ny,), "No vectorization accepted in y"
        
        mask = ma.getmaskarray(y)
        if np.all(mask):
            return
        
        # Remove inactive outputs
        active = ~mask
        y = ma.compressed(y)
        R = self.model.R()[np.ix_(active, active)]
        def meas_fun(x):
            return self.model.h(self.t, x)[..., active]
        
        # Perform unscented transform
        hmean, hcov = self.__ut.transform(meas_fun, self.mean, self.cov)
        hcrosscov = self.__ut.crosscov()
        
        # Factorize covariance
        ycov = hcov + R
        ycov_chol = numpy.linalg.cholesky(ycov)
        ycov_chol_inv = numpy.linalg.inv(ycov_chol)
        ycov_inv = np.einsum('...ki,...kj', ycov_chol_inv, ycov_chol_inv)
        
        # Perform correction
        err = y - hmean
        gain = np.einsum('...ik,...kj', hcrosscov, ycov_inv)
        corr_mean = self.mean + np.einsum('...ij,...j', gain, err)
        corr_cov = self.cov - np.einsum('...ik,...jl,...lk', gain, gain, ycov)
        self._save_correction(corr_mean, corr_cov)
        
        if self.save_likelihood:
            ycov_chol_diag = np.einsum('...kk->...k', ycov_chol)
            self.loglikelihood += (
                -0.5 * np.einsum('...i,...ij,...j', err, ycov_inv, err) -
                np.log(ycov_chol_diag).sum(-1)
            )

        if self.calculate_gradients:
            self._calculate_correction_grad(
                active, err, gain, hcrosscov, ycov, ycov_inv, ycov_chol
            )
    
    def _calculate_correction_grad(self, active, e, K, Pxh, Py, PyI, PyC):
        t = self.t
        x = self.mean
        dx_dq = self.mean_grad
        dPx_dq = self.cov_grad
        dR_dq = self.model.dR_dq()[(...,) + np.ix_(active, active)]
        
        def Dh_Dq_fun(x, dx_dq):
            dh_dq = self.model.dh_dq(t, x)[..., active]
            dh_dx = self.model.dh_dx(t, x)[..., active]
            return dh_dq + np.einsum('...qx,...xh->...qh', dx_dq, dh_dx)
        ut_grads = self.__ut.transform_diff(Dh_Dq_fun, dx_dq, dPx_dq, True)
        Dh_Dq, dPh_dq, dPxh_dq = ut_grads
        
        de_dq = -Dh_Dq
        dPy_dq = dPh_dq + dR_dq
        dPyI_dq = -np.einsum('...ij,...ajk,...kl', PyI, dPy_dq, PyI)
        dK_dq = np.einsum('...ik,...akj', Pxh, dPyI_dq)
        dK_dq += np.einsum('...aik,...kj', dPxh_dq, PyI)
        
        self.mean_grad += np.einsum('...aij,...j', dK_dq, e)
        self.mean_grad += np.einsum('...ij,...aj', K, de_dq)
        self.cov_grad -= np.einsum('...aik,...jl,...lk', dK_dq, K, Py)
        self.cov_grad -= np.einsum('...ik,...ajl,...lk', K, dK_dq, Py)
        self.cov_grad -= np.einsum('...ik,...jl,...alk', K, K, dPy_dq)

        if self.save_likelihood:
            dPyC_dq = cholesky_sqrt_diff(PyC, dPy_dq)
            diag_PyC = np.einsum('...kk->...k', PyC)
            diag_dPyC_dq = np.einsum('...kk->...k', dPyC_dq)
            self.loglikelihood_grad -= np.sum(diag_dPyC_dq / diag_PyC, axis=-1)
            self.loglikelihood_grad -= 0.5 * (
                np.einsum('...ai,...ij,...j', de_dq, PyI, e) +
                np.einsum('...i,...aij,...j', e, dPyI_dq, e) +
                np.einsum('...i,...ij,...aj', e, PyI, de_dq)
            )


class DTUnscentedKalmanFilter(DTUnscentedPredictor, DTUnscentedCorrector):
    pass

