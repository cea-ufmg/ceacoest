'''Kalman filtering / smoothing module.

TODO
----
 * Add derivative of SVD square root.

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
import scipy.linalg


class DTKalmanFilterBase(metaclass=abc.ABCMeta):
    '''Discrete-time Kalman filter/smoother abstract base class.'''
    
    def __init__(self, model, mean, cov, **options):
        self.model = model
        '''The underlying system model.'''
        
        self.mean = np.asarray(mean, dtype=float)
        '''The latest filtered state mean.'''
        assert (self.model.nx,) == self.mean.shape
        
        self.cov = np.asarray(cov, dtype=float)
        '''The latest filtered state covariance.'''
        assert (self.model.nx, self.model.nx) == self.cov.shape
        
        self.initialize_options(options)
        
        self.k = 0
        '''The latest working sample number.'''
        
        if self.save_history:
            self.pred_mean = [self.mean]
            '''Predicted state mean history.'''
            
            self.pred_cov = [self.cov]
            '''Predicted state covariance history.'''
            
            self.corr_mean = [self.mean]
            '''Corrected state mean history.'''
            
            self.corr_cov = [self.cov]
            '''Corrected state covariance history.'''
        
        if self.save_pred_crosscov:
            self.pred_crosscov = []
            '''State prediction cross-covariance history.'''
        
        if self.save_likelihood:
            self.loglikelihood = 0
            '''Measurement log-likelihood.'''
        
        if self.calculate_gradients:
            self.cov_grad = np.zeros((self.model.nx, self.model.nx))
    
    def initialize_options(self, options):
        '''Initialize and validate filter options, given as a dictionary.
        
        Options
        -------
        save_history :
            Whether to save the filtering history or just the latest values.
            True by default.
        save_likelihood :
            Whether to save the measurement likelihood. True by default.
        save_pred_crosscov :
            Whether to save the prediction cross-covariance matrices. True by
            default, automatically false if save_history is false.
        calculate_gradients :
            Whether to calculate the filter gradients with respect to the
            parameter vector q. False by default.
        initial_mean_grad : (nx,) array_like
            Gradient of mean of inital states. Zero by default.
        
        '''
        #Initialize any other inherited options
        try:
            super().set_options(options)
        except AttributeError:
            pass
        
        self.save_history = options.get('save_history', True)
        self.save_likelihood = options.get('save_likelihood', True)
        self.save_pred_crosscov = (self.save_history and 
                                   options.get('save_pred_crosscov', True))
        self.calculate_gradients = options.get('calculate_gradients', False)
        
        if self.calculate_gradients:
            try:
                self.mean_grad = options['initial_mean_grad']
            except KeyError:
                self.mean_grad = np.zeros((self.model.nx, self.model.nq))
    
    def _save_prediction(self, mean, cov):
        '''Save the prection data and update time index.'''
        self.mean = mean
        self.cov = cov
        self.k += 1
        
        if self.save_history:
            self.pred_mean.append(mean)
            self.pred_cov.append(cov)
            self.corr_mean.append(mean)
            self.corr_cov.append(cov)
    
    def _save_correction(self, mean, cov):
        '''Save the correction data.'''
        self.mean = mean
        self.cov = cov
        
        if self.save_history:
            self.corr_mean[-1] = mean
            self.corr_cov[-1] = cov
    
    @abc.abstractmethod
    def predict(self, u=[]):
        '''Predict the state distribution at the next time index.'''
        raise NotImplementedError("Pure abstract method.")
    
    @abc.abstractmethod
    def correct(self, y, u=[]):
        '''Correct the state distribution, given the measurement vector.'''
        raise NotImplementedError("Pure abstract method.")
    
    def filter(self, y, u=None):
        for k, yk in enumerate(y[:-1]):
            uk = [] if u is None else u[k]
            self.correct(yk, uk)
            self.predict(uk)
        
        self.correct(y[-1], [] if u is None else u[-1])


def svd_sqrt(mat):
    '''SVD-based square root of a symmetric positive-semidefinite matrix.

    Used for unscented transform.
    
    Example
    -------
    Generate a random positive-semidefinite symmetric matrix.
    >>> A = np.random.randn(4, 4)
    >>> Q = np.dot(A, A.T)
    
    The suare root should satisfy SS' = Q
    >>> S = svd_sqrt(Q)
    >>> SST = np.dot(S, S.T)
    >>> np.testing.assert_allclose(Q, SST)
    
    '''
    [U, s, Vh] = scipy.linalg.svd(mat)
    return U * np.sqrt(s)


def cholesky_sqrt(mat):
    '''Lower triangular Cholesky decomposition for unscented transform.'''
    return scipy.linalg.cholesky(mat, lower=True)


def cholesky_sqrt_diff(S, dQ=None):
    '''Derivatives of lower triangular Cholesky decomposition.
    
    Parameters
    ----------
    S : (n, n) array_like
        The lower triangular Cholesky decomposition of a matrix `Q`, i.e.,
        `S * S.T == Q`.
    dQ : (n, n, ...) array_like or None
        The derivatives of `Q` with respect to some parameters. Must be
        symmetric with respect to the first two axes, i.e., 
        `dQ[i,j,...] == dQ[j,i,...]`. If `dQ` is `None` then the derivatives
        are taken with respect to `Q`, i.e., `dQ[i,j,i,j] = 1` and
        `dQ[i,j,j,i] = 1`.
    
    Returns
    -------
    dS : (n, n, ...) array_like
        The derivative of `S` with respect to some parameters or with respect
        to `Q` if `dQ` is `None`.
    
    '''
    n = len(S)
    k = np.arange(n)
    i, j = np.tril_indices_from(S)
    ix, jx, kx = np.ix_(i, j, k)
    
    A = np.zeros((n, n, n, n))
    A[ix, jx, ix, kx] = S[jx, kx]
    A[ix, jx, jx, kx] += S[ix, kx]
    A_tril = A[i, j][..., i, j]
    A_tril_inv = scipy.linalg.inv(A_tril)
    
    if dQ is None:
        nnz = len(i)
        dQ_tril = np.zeros((nnz, n, n))
        dQ_tril[np.arange(nnz), i, j] = 1
        dQ_tril[np.arange(nnz), j, i] = 1
    else:
        dQ_tril = dQ[i, j]
    
    D_tril = np.einsum('ab,b...->a...', A_tril_inv, dQ_tril)
    D = np.zeros((n, n) + dQ_tril.shape[1:])
    D[i, j] = D_tril
    
    return D


cholesky_sqrt.diff = cholesky_sqrt_diff


class UnscentedTransform:
    
    def __init__(self, **options):
        self.initialize_options(options)
    
    def initialize_options(self, options):
        '''Initialize and validate filter options, given as a dictionary.
        
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
        #Initialize any other inherited options
        try:
            super().set_options(options)
        except AttributeError:
            pass
        
        self.kappa = options.get('kappa', 0.0)
        
        sqrt_opt = options.get('sqrt', 'cholesky')
        if sqrt_opt == 'cholesky':
            self.sqrt = cholesky_sqrt
        elif sqrt_opt == 'svd':
            self.sqrt = svd_sqrt
        elif isinstance(sqrt_opt, collections.Callable):
            self.sqrt = sqrt_opt
        else:
            raise ValueError("Invalid value for 'sqrt' option.")
    
    def gen_sigma_points(self, mean, cov):
        '''Generate sigma-points deviations and weights.'''
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        n = len(mean)
        kappa = self.kappa
        assert n + kappa != 0
        
        cov_sqrt = self.sqrt((n + kappa) * cov)
        dev = np.hstack((-cov_sqrt, cov_sqrt))
        weights = np.repeat(0.5 / (n + kappa), 2 * n)
        
        if kappa != 0:
            dev = np.hstack((np.zeros((n, 1)), dev))
            weights = np.hstack([kappa / (n + kappa), weights])
        
        self.input_dev = dev
        self.weights = weights
        self.input_sigma = dev + mean[:, None]
        
        return (self.input_sigma, weights)
    
    def unscented_transform(self, f, mean, cov):
        [input_sigma, weights] = self.gen_sigma_points(mean, cov)
        
        output_sigma = f(input_sigma)
        output_mean = np.dot(output_sigma, weights)
        output_dev = output_sigma - output_mean[:, None]
        output_cov = np.einsum('ik,jk,k', output_dev, output_dev, weights)
        
        self.output_dev = output_dev
        return (output_mean, output_cov)
    
    def transform_crosscov(self):
        try:
            input_dev = self.input_dev
            output_dev = self.output_dev
            weights = self.weights
        except AttributeError:
            msg = "Transform must be done before requesting crosscov."
            raise RuntimeError(msg)
        
        return np.einsum('ik,jk,k', input_dev, output_dev, weights)

    def sigma_points_diff(self, mean_diff, cov_diff):
        '''Derivative of sigma-points.'''
        try:
            input_dev = self.input_dev
        except AttributeError:
            msg = "Transform must be done before requesting derivatives."
            raise RuntimeError(msg)
        
        n = len(input_dev)
        kappa = self.kappa
        
        cov_sqrt = input_dev[:, -n:]
        cov_sqrt_diff = self.sqrt.diff(cov_sqrt, (n + kappa) * cov_diff)
        dev_diff = np.hstack((-cov_sqrt_diff, cov_sqrt_diff))
        
        if kappa != 0:
            center_point_diff = np.zeros((n, 1) + mean_diff.shape[1:])
            dev_diff = np.hstack((center_point_diff, dev_diff))
        
        input_sigma_diff = dev_diff + mean_diff[:, None]
        return input_sigma_diff
    
    def transform_diff(self, f_diff, mean_diff, cov_diff):
        try:
            input_dev = self.input_dev
            output_dev = self.output_dev
            input_sigma = self.input_sigma
            weights = self.weights
        except AttributeError:
            msg = "Transform must be done before requesting derivatives."
            raise RuntimeError(msg)
        
        in_sigma_diff = self.sigma_points_diff(mean_diff, cov_diff)
        out_sigma_diff = f_diff(input_sigma, in_sigma_diff)
        
        out_mean_diff = np.einsum('xs...,s->x...', out_sigma_diff, weights)
        out_dev_diff = out_sigma_diff - out_mean_diff[:, None]
        out_cov_diff = np.einsum('ik...,jk,k->ij...', out_dev_diff, 
                                 output_dev, weights)
        out_cov_diff += np.einsum('ik,jk...,k->ij...', output_dev,
                                  out_dev_diff, weights)
        
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
        
        if self.calculate_gradients:
            self._calculate_prediction_grad(u, aug_mean, aug_cov)
        
        if self.save_pred_crosscov:
            aug_crosscov = self.transform_crosscov()
            pred_crosscov = aug_crosscov[:self.model.nx]
            self.pred_crosscov.append(pred_crosscov)
        
        self._save_prediction(pred_mean, pred_cov)
    
    def _calculate_prediction_grad(self, u=[]):
        nx = self.model.nx
        nw = self.model.nw
        nq = self.model.nq
        
        aug_mean_grad = np.concatenate([self.mean_grad, np.zeros(nw, nq)])
        aug_cov_grad = np.zeros((nx + nw, nx + nw, nq))
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

