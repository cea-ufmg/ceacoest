'''Kalman filtering / smoothing module.'''


import abc

import numpy as np


class DiscreteTimeKalmanFilterBase(metaclass=abc.ABCMeta):
    
    def __init__(self, model, mean, cov, save_history=True, 
                 save_pred_xcov=True, save_likelihood=True):
        self.model = model
        '''The underlying system model.'''
        
        self.save_history = save_history
        '''Whether to save the filtering history or just the latest values.'''
        
        self.save_pred_xcov = save_pred_xcov
        '''Whether to save the prediction cross-covariance matrices.'''
        
        self.save_likelihood = save_likelihood
        '''Whether to save the measurement likelihood.'''
        
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
    
    def _save_prediction(self, mean, cov, xcov=None):
        '''Save the prection data and update time index.'''
        self.mean = mean
        self.cov = cov
        self.k += 1
        
        if self.save_history:
            self.pred_mean.append(mean)
            self.pred_cov.append(cov)
            self.corr_mean.append(mean)
            self.corr_cov.append(cov)
            
            if self.save_pred_xcov:
                self.pred_xcov.append(xcov)

    def _save_correction(self, mean, cov, loglikelihood=0):
        '''Save the correction data and update time index.'''
        self.mean = mean
        self.cov = cov
        
        if self.save_history:
            self.corr_mean[-1] = mean
            self.corr_cov[-1] = cov
            
        if self.save_likelihood:
            self.loglikelihood += loglikelihood
    
    @abc.abstractmethod
    def predict(self, u=[]):
        '''Predict the state distribution at the next time index.'''
        raise NotImplementedError("Pure abstract method.")
    
    @abc.abstractmethod
    def correct(self, y):
        '''Correct the state distribution, given the measurement vector.'''
        raise NotImplementedError("Pure abstract method.")
    
    def filter(self, y, u=None):
        for k, yk in y[:-1]:
            self.correct(yk)
            
            uk = [] if u is None else u[k]
            self.predict(uk)
        
        self.correct(y[-1])


