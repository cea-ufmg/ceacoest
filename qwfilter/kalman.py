'''Kalman filtering / smoothing module.

TODO
----
 * Add derivative of SVD square root.
 * Vectorize cholesky_sqrt_diff.
 * Make docstrings for all constructors.

Improvement ideas
-----------------
 * Make the unscented transform object a member variable of the predictor
   instead of superclass, so that it is not shared among prediction and
   correction. Weights can also be generated in the constructor.

'''


import abc
import collections

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
        
        self.k = 0
        '''The latest working sample number.'''
        
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
        
        '''
        # Initialize any other inherited options
        try:
            super().initialize_options(options)
        except AttributeError:
            pass

        # Get the base options
        self.save_likelihood = options.get('save_likelihood', True)
        self.calculate_gradients = options.get('calculate_gradients', False)
        self.calculate_pred_crosscov = options.get('calculate_pred_crosscov',
                                                   False)
        
        # Initialize the variables associtated to the options
        if self.save_likelihood:
            self.loglikelihood = 0
        
        if self.calculate_gradients:
            default_mean_grad = np.zeros((self.model.np,) + self.mean.shape)
            default_cov_grad = np.zeros((self.model.np,) + self.cov.shape)
            self.mean_grad = options.get('initial_mean_grad', default_mean_grad)
            self.cov_grad = options.get('initial_cov_grad', default_cov_grad)
    
    def _save_prediction(self, mean, cov):
        '''Save the prection data and update time index.'''
        self.mean = mean
        self.cov = cov
    
    def _save_correction(self, mean, cov):
        '''Save the correction data.'''
        self.mean = mean
        self.cov = cov
    
    def _save_prediction_crosscov(self, crosscov):
        '''Save the prediction cross-covariance.'''
        pass
    
    @abc.abstractmethod
    def predict(self, u=[]):
        '''Predict the state distribution at the next time index.'''
        raise NotImplementedError("Pure abstract method.")
    
    @abc.abstractmethod
    def correct(self, y, u=[]):
        '''Correct the state distribution, given the measurement vector.'''
        raise NotImplementedError("Pure abstract method.")
    
    def filter(self, y, u=None):
        if u is None:
            u = np.empty((len(y), 0))
        
        for k, yk in enumerate(y[:-1]):
            uk = u[k]
            self.correct(yk, uk)
            self.predict(uk)
        
        self.correct(y[-1], u[-1])


class FilterArrayHistory(DTKalmanFilterBase):
    '''Class to save filter history in numpy arrays.'''
    
    def initalize_options(self, options):
        '''Initialize and validate options, given as a dictionary.
        
        Options
        -------
        history_size :
            Number of elements to allocate for history, zero by default.
        
        '''
        # Initialize any other inherited options
        try:
            super().initialize_options(options)
        except AttributeError:
            pass
        
        self.initialize_history(options.get('history_size', 0))
    
    def initialize_history(self, size):
        self.history_size = size
        mean_shape = (size,) + self.mean.shape
        cov_shape = (size,) + self.cov.shape
        
        self.pred_mean = np.zeros(mean_shape)
        self.pred_cov = np.zeros(cov_shape)
        self.corr_mean = np.zeros(mean_shape)
        self.corr_cov = np.zeros(cov_shape)
        
        if self.calculate_pred_crosscov:
            self.pred_crosscov = np.zeros(cov_shape)
    
    def _save_prediction(self, mean, cov):
        super()._save_prediction(mean, cov)
        
        if self.k < self.history_size:
            self.pred_mean[self.k] = mean
            self.pred_cov[self.k] = cov
            self.corr_mean[self.k] = mean
            self.corr_cov[self.k] = cov
    
    def _save_correction(self, mean, cov):
        super()._save_correction(mean, cov)
        
        if self.k < self.history_size:
            self.corr_mean[self.k] = mean
            self.corr_cov[self.k] = cov

    def _save_prediction_crosscov(self, crosscov):
        '''Save the prediction cross-covariance.'''
        super()._save_prediction_crosscov(mean, cov)
        
        if self.k < self.history_size:
            self.pred_crosscov[self.k] = crosscov

    
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
    return np.rollaxis(U * np.sqrt(s), -1)


def cholesky_sqrt(mat):
    '''Upper triangular Cholesky decomposition for unscented transform.'''
    lower_chol = numpy.linalg.cholesky(mat)
    return np.rollaxis(lower_chol, -1)


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
    
    def __init__(self, nin, nout=None, **options):
        self.nin = nin
        '''Number of inputs.'''
        
        self.nout = nin if nout is None else nout
        '''Number of outputs.'''
        
        self.initialize_options(options)
        
        self.nsigma = 2 * nin + (self.kappa != 0)
        '''Number of sigma points.'''
        
        self.weights = np.repeat(0.5 / (nin + self.kappa), self.nsigma)
        '''Transform weights.'''
        if self.kappa != 0:
            self.weights[-1] = self.kappa / (nin + self.kappa)
    
    def initialize_options(self, options):
        '''Initialize and validate transform options, given as a dictionary.
        
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
        self.kappa = options.get('kappa', 0.0)
        assert self.nin + self.kappa != 0
        
        sqrt_opt = options.get('sqrt', 'cholesky')
        if sqrt_opt == 'cholesky':
            self.sqrt = cholesky_sqrt
        elif sqrt_opt == 'svd':
            self.sqrt = svd_sqrt
        elif isinstance(sqrt_opt, collections.Callable):
            self.sqrt = sqrt_opt
        else:
            raise ValueError("Invalid value for `sqrt` option.")
    
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
        return in_sigma
    
    def unscented_transform(self, f, mean, cov):
        in_sigma = self.gen_sigma_points(mean, cov)
        weights = self.weights
        
        out_sigma = f(in_sigma)
        out_mean = np.tensordot(out_sigma, weights, axes=(0, 0))
        out_dev = out_sigma - out_mean
        out_cov = np.einsum('k...i,k...j->...ij', out_dev, out_dev, weights)
        
        self.out_dev = out_dev
        return (out_mean, out_cov)
    
    def transform_crosscov(self):
        weights = self.weights
        try:
            in_dev = self.in_dev
            out_dev = self.output_dev
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
        kappa = self.kappa
        
        cov_sqrt = in_dev[nin:]
        cov_sqrt_diff = self.sqrt.diff(cov_sqrt, (nin + kappa) * cov_diff)
        dev_diff = np.zeros((self.nsigma,) + self.mean_diff.shape)
        dev_diff[:nin] = cov_sqrt_diff
        dev_diff[nin:(2 * nin)] = -cov_sqrt_diff
        
        in_sigma_diff = dev_diff + mean_diff
        return in_sigma_diff
    
    def transform_diff(self, f_diff, mean_diff, cov_diff):
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
        out_mean_diff = np.tensordot(out_sigma_diff, weights, axes=(0, 0))
        out_dev_diff = out_sigma_diff - out_mean_diff
        out_cov_diff = np.einsum('k...i,k...j,k->...ij', 
                                 out_dev_diff, out_dev, weights)
        out_cov_diff += np.einsum('k...i,k...j,k->...ij', 
                                  out_dev, out_dev_diff, weights)
        return (out_mean_diff, out_cov_diff)


class DTUnscentedPredictor(DTKalmanFilterBase, UnscentedTransform):
    
    def predict(self, u=[]):
        '''Predict the state distribution at the next time index.'''
        
        aug_mean = np.concatenate([self.mean, np.zeros(self.model.nw)])
        aug_cov = scipy.linalg.block_diag(self.cov, self.model.wcov)
        def faug(xaug):
            x = xaug[:self.model.nx]
            w = xaug[self.model.nx:]
            return self.model.f(self.k, x, u, w)
        
        pred_mean, pred_cov = self.unscented_transform(faug, aug_mean, aug_cov)
        self._save_prediction(pred_mean, pred_cov)
        
        if self.calculate_gradients:
            self._calculate_prediction_grad(u, aug_mean, aug_cov)
        
        if self.calculate_pred_crosscov:
            aug_crosscov = self.transform_crosscov()
            pred_crosscov = aug_crosscov[:self.model.nx]
            self._save_prediction_crosscov(pred_crosscov)
        
        self.k += 1
    
    def _calculate_prediction_grad(self, u=[]):
        nx = self.model.nx
        nw = self.model.nw
        np_ = self.model.np #Trailing underscore to differentiate from numpy
        
        aug_mean_grad = np.concatenate([self.mean_grad, np.zeros(nw, np_)])
        aug_cov_grad = np.zeros((nx + nw, nx + nw, np_))
        aug_cov_grad[:nx, :nx] = self.cov_grad
        aug_cov_grad[nx:, nx:] = self.model.dwcov_dq
        
        def faug_grad(xaug, dxaug_dq):
            x = xaug[:nx]
            w = xaug[nx:]
            dx_dq = dxaug_dq[:nx]
            dw_dq = dxaug_dq[nx:]
            
            df_dq = self.model.df_dq(self.k, x, u, w)
            df_dx = self.model.df_dx(self.k, x, u, w)
            df_dw = self.model.df_dw(self.k, x, u, w)
            return (df_dq + np.einsum('...x,x...', df_dx, dx_dq) +
                    np.einsum('...w,w...', df_dw, dw_dq))
        
        grads = self.transform_diff(faug_grad, aug_mean_grad, aug_cov_grad)
        self.mean_grad, self.cov_grad = grads


class DTUnscentedCorrector(DTKalmanFilterBase, UnscentedTransform):
    
    def correct(self, y, u=[]):
        '''Correct the state distribution, given the measurement vector.'''
        mask = ma.getmaskarray(y)
        if np.all(mask):
            return
        
        #Remove inactive outputs
        active = ~mask
        y = ma.compressed(y)
        vcov = self.model.vcov[np.ix_(active, active)]
        def meas_fun(x):
            return self.model.h(self.k, x, u)[active]
        
        #Perform unscented transform
        hmean, hcov = self.unscented_transform(meas_fun, self.mean, self.cov)
        hcrosscov = self.transform_crosscov()
        
        #Factorize covariance
        ycov = hcov + vcov
        ycov_chol = scipy.linalg.cholesky(ycov, lower=True)
        ycov_chol_inv = scipy.linalg.inv(ycov_chol)
        ycov_inv = ycov_chol_inv.T.dot(ycov_chol_inv)
        
        #Perform correction
        err = y - hmean
        gain = np.dot(hcrosscov, ycov_inv)
        pred_mean = self.mean + gain.dot(err)
        pred_cov = self.cov - gain.dot(ycov).dot(gain.T)
        self._save_correction(pred_mean, pred_cov)
        
        if self.save_likelihood:
            self.loglikelihood += (-0.5 * err.dot(ycov_inv).dot(err) +
                                   -np.log(np.diag(ycov_chol)).sum())


class DTUnscentedKalmanFilter(DTUnscentedPredictor, DTUnscentedCorrector):
    pass

