'''Kalman filtering / smoothing module.'''


import abc
import collections

import numpy as np
import scipy.linalg


class DTKalmanFilterBase(metaclass=abc.ABCMeta):
    
    def __init__(self, model, mean, cov, save_history=True,
                 save_pred_xcov=True, save_likelihood=True, **options):
        self.model = model
        '''The underlying system model.'''

        self.mean = np.asarray(mean, dtype=float)
        '''The latest filtered state mean.'''

        self.cov = np.asarray(cov, dtype=float)
        '''The latest filtered state covariance.'''
        
        self.save_history = save_history
        '''Whether to save the filtering history or just the latest values.'''
        
        self.save_pred_xcov = save_pred_xcov
        '''Whether to save the prediction cross-covariance matrices.'''
        
        self.save_likelihood = save_likelihood
        '''Whether to save the measurement likelihood.'''

        self.options = options
        '''Filter options.'''
        
        self.k = 0
        '''The latest working sample number.'''
        
        if save_history:
            self.pred_mean = [mean]
            '''Predicted state mean history.'''
            
            self.pred_cov = [cov]
            '''Predicted state covariance history.'''
            
            self.corr_mean = [mean]
            '''Corrected state mean history.'''
            
            self.corr_cov = [cov]
            '''Corrected state covariance history.'''
    
            if save_pred_xcov:
                self.pred_xcov = []
                '''State prediction cross-covariance history.'''
        
        if save_likelihood:
            self.loglikelihood = 0
            '''Measurement log-likelihood.'''
    
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


def sigma_points(mean, cov, kappa=0, sqrt='svd'):
    '''Generate sigma-points deviations and weights for unscented transform.'''
    n = len(mean)
    assert n + kappa != 0
    
    if sqrt == 'svd':
        [U, s, Vh] = scipy.linalg.svd((n + kappa) * cov)
        dev = (U * np.sqrt(s)).T
    elif sqrt == 'cholesky':
        dev = scipy.linalg.cholesky((n + kappa) * cov, lower=False)
    elif isinstance(sqrt, collections.Callable):
        dev = sqrt((n + kappa) * cov)
    else:
        raise ValueError("Unknown option for 'sqrt' argument.")
    
    dev = np.vstack([dev, -dev])
    weights = np.repeat(0.5 / (n + kappa), 2 * n)
    
    if kappa != 0:
        dev = np.vstack([np.zeros_like(mean), dev])
        weights = np.hstack([kappa / (n + kappa), weights])
    
    return (dev, weights)


def unscented_transform(f, mean, cov, return_xcov=False, kappa=0, sqrt='svd'):
    [input_dev, weights] = sigma_points(mean, cov, kappa, sqrt)
    
    output_sigma = np.array([f(mean + dev) for dev in input_dev])
    output_mean = np.dot(output_sigma, weights)
    output_dev = output_sigma - output_mean
    output_cov = np.einsum('ki,kj,k', output_dev, output_dev, weights)
    
    if return_xcov:
        xcov = np.einsum('ki,kj,k', input_dev, output_dev, weights)
        return (output_mean, output_cov, xcov)
    else:
        return (output_mean, output_cov)


class DTUnscentedBase(DTKalmanFilterBase):
    
    @property
    def kappa(self):
        return self.options.get('kappa', 0.0)

    @property
    def sqrt(self):
        return self.options.get('sqrt', 'svd')


class DTUnscentedPredictor(DTUnscentedBase):
    
    def predict(self, u=[]):
        '''Predict the state distribution at the next time index.'''

        aug_mean = np.hstack([self.mean, np.zeros(self.model.nw)])
        aug_cov = scipy.linalg.block_diag([self.cov, self.model.w_cov])
        def aug_f(aug_x):
            x = aug_x[:self.model.nx]
            w = aug_x[self.model.nx:]
            return self.model.f(self.k, x, u, w)
        
        pred_mean, pred_cov, *xcov_out = unscented_transform(
            aug_f, aug_mean, aug_cov, return_xcov=self.save_pred_xcov,
            kappa=self.kappa, sqrt=self.sqrt
        )
        self._save_prediction(pred_mean, pred_cov)
        
        if self.save_history and self.save_pred_xcov:
            pred_xcov = xcov_out[0][:self.model.nx, :self.model.nx]
            self.pred_xcov.append(pred_xcov)


class DTUnscentedCorrector(DTUnscentedBase):
    
    def correct(self, y, u=[]):
        '''Correct the state distribution, given the measurement vector.'''
        
        h_mean, h_cov, h_xcov = unscented_transform(
            lambda x: self.model.h(k, x, u), self.mean, self.cov, 
            return_xcov=True, kappa=self.kappa, sqrt=self.sqrt
        )
        
        y_cov = h_cov + self.model.v_cov
        y_cov_chol = scipy.linalg.cholesky(ycov, lower=True)
        y_cov_chol_inv = np.linalg.inv(y_cov_chol)
        y_cov_inv = ycov_chol_inv.T.dot(y_cov_chol_inv)
        
        err = y - h_mean
        gain = np.dot(h_xcov, y_cov_inv)
        pred_mean = self.mean + gain.dot(err)
        pred_cov = self.cov - gain.dot(y_cov).dot(gain.T)
        self._save_correction(pred_mean, pred_cov)
        
        if self.save_likelihood:
            self.loglikelihood += (-0.5*err.dot(y_cov_inv).dot(err) +
                                   -np.log(np.diag(y_cov_chol)).sum())


class DTUnscentedKalmanFilter(DTUnscentedPredictor, DTUnscentedCorrector):
    pass

