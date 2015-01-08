'''Kalman filtering / smoothing module.'''


import abc
import collections

import numpy as np
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
        
        self.options = options
        '''Filter options.'''
        
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
    
        if self.save_pred_xcov:
            self.pred_xcov = []
            '''State prediction cross-covariance history.'''
        
        if self.save_likelihood:
            self.loglikelihood = 0
            '''Measurement log-likelihood.'''
 
    @property
    def save_history(self):
        '''Whether to save the filtering history or just the latest values.'''
        return self.options.get('save_history', True)

    @property
    def save_likelihood(self):
        '''Whether to save the measurement likelihood.'''
        return self.options.get('save_likelihood', True)

    @property
    def save_pred_xcov(self):
        '''Whether to save the prediction cross-covariance matrices.'''
        return self.save_history and self.options.get('save_pred_xcov', True)
    
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


class UnscentedTransform:
    
    def __init__(self, **options):
        self.options = options
    
    @property
    def kappa(self):
        return self.options.get('kappa', 0.0)
    
    @property
    def sqrt(self):
        return self.options.get('sqrt', 'svd')
    
    def _gen_sigma_points(self, mean, cov):
        '''Generate sigma-points deviations and weights.'''
        mean = np.asarray(mean)
        cov = np.asarray(cov)
        n = len(mean)
        kappa = self.kappa
        sqrt = self.sqrt
        assert n + kappa != 0
        
        if sqrt == 'svd':
            [U, s, Vh] = scipy.linalg.svd((n + kappa) * cov)
            cov_sqrt = U * np.sqrt(s)
        elif sqrt == 'cholesky':
            cov_sqrt = scipy.linalg.cholesky((n + kappa) * cov, lower=True)
        elif isinstance(sqrt, collections.Callable):
            cov_sqrt = sqrt((n + kappa) * cov)
        else:
            raise ValueError("Unknown option for 'sqrt' argument.")
        
        dev = np.hstack((cov_sqrt, -cov_sqrt))
        weights = np.repeat(0.5 / (n + kappa), 2 * n)
        
        if kappa != 0:
            dev = np.hstack((np.zeros_like(mean), dev))
            weights = np.hstack([kappa / (n + kappa), weights])
        
        self.input_dev = dev
        self.weights = weights
        self.input_sigma = dev + mean[:, None]
        
        return (self.input_sigma, weights)
    
    def unscented_transform(self, f, mean, cov):
        [input_sigma, weights] = self._gen_sigma_points(mean, cov)
        
        output_sigma = f(input_sigma)
        output_mean = np.dot(output_sigma, weights)
        output_dev = output_sigma - output_mean[:, None]
        output_cov = np.einsum('ik,jk,k', output_dev, output_dev, weights)
        
        self.output_dev = output_dev
        return (output_mean, output_cov)
    
    def transform_xcov(self):
        try:
            input_dev = self.input_dev
            output_dev = self.output_dev
            weights = self.weights
        except AttributeError:
            raise RuntimeError(
                "Unscented transform must be done before requesting xcov."
            )
        
        return np.einsum('ik,jk,k', input_dev, output_dev, weights)


class DTUnscentedPredictor(DTKalmanFilterBase, UnscentedTransform):
    
    def predict(self, u=[]):
        '''Predict the state distribution at the next time index.'''

        aug_mean = np.hstack([self.mean, np.zeros(self.model.nw)])
        aug_cov = scipy.linalg.block_diag(self.cov, self.model.w_cov)
        def aug_f(aug_x):
            x = aug_x[:self.model.nx]
            w = aug_x[self.model.nx:]
            return self.model.f(self.k, x, u, w)
        
        pred_mean, pred_cov = self.unscented_transform(aug_f, aug_mean, aug_cov)
        self._save_prediction(pred_mean, pred_cov)
        
        if self.save_pred_xcov:
            aug_xcov = self.transform_xcov()
            pred_xcov = aug_xcov[:self.model.nx]
            self.pred_xcov.append(pred_xcov)


class DTUnscentedCorrector(DTKalmanFilterBase, UnscentedTransform):
    
    def correct(self, y, u=[]):
        '''Correct the state distribution, given the measurement vector.'''
        
        h_mean, h_cov = self.unscented_transform(
            lambda x: self.model.h(self.k, x, u), self.mean, self.cov
        )
        h_xcov = self.transform_xcov()
        
        y_cov = h_cov + self.model.v_cov
        y_cov_chol = scipy.linalg.cholesky(y_cov, lower=True)
        y_cov_chol_inv = scipy.linalg.inv(y_cov_chol)
        y_cov_inv = y_cov_chol_inv.T.dot(y_cov_chol_inv)
        
        err = y - h_mean
        gain = np.dot(h_xcov, y_cov_inv)
        pred_mean = self.mean + gain.dot(err)
        pred_cov = self.cov - gain.dot(y_cov).dot(gain.T)
        self._save_correction(pred_mean, pred_cov)
        
        if self.save_likelihood:
            self.loglikelihood += (-0.5 * err.dot(y_cov_inv).dot(err) +
                                   -np.log(np.diag(y_cov_chol)).sum())


class DTUnscentedKalmanFilter(DTUnscentedPredictor, DTUnscentedCorrector):
    pass

