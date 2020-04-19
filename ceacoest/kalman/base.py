"""Abstract base classes for Kalman filtering / smoothing."""

import abc
import collections
import re

import attrdict
import numpy as np
import numpy.ma as ma
import numpy.linalg

from .. import utils


class DTFilter(metaclass=abc.ABCMeta):
    """Discrete-time Kalman filter/smoother abstract base class."""
    
    def __init__(self, model, x=None, Px=None, **options):
        """Create a discrete-time Kalman filter.
        
        Parameters
        ----------
        model :
            The underlying system model.
        
        """
        self.model = model
        """The underlying system model."""
        
        self.x = model.x0() if x is None else np.asarray(x)
        """State vector mean."""
        
        self.Px = model.Px0() if Px is None else np.asarray(Px)
        """State vector covariance."""
        
        self.k = options.get('k', 0)
        """Time index."""
        
        self.L = options.get('L', 0.0)
        """Measurement log-likelihood."""
        
        nq = getattr(model, 'nq', 0)
        nx = model.nx
        base_shape = self.x.shape[:-1]
        self.base_shape = base_shape
        """Base shape of broadcasting."""

        self.dL_dq = options.get('dL_dq', np.zeros(base_shape + (nq,)))
        """Measurement log-likelihood derivative."""

        self.d2L_dq2 = options.get('dL_dq', np.zeros(base_shape + (nq, nq)))
        """Measurement log-likelihood derivative."""
        
        self.dx_dq = self._get_initial('dx_dq', options, (nq, nx))
        """State vector derivative."""
        
        self.dPx_dq = self._get_initial('dPx_dq', options, (nq, nx, nx))
        """State vector covariance derivative."""

        self.d2x_dq2 = self._get_initial('d2x_dq2', options, (nq, nq, nx))
        """State vector derivative."""
        
        self.d2Px_dq2 = self._get_initial('d2Px_dq2', options, (nq, nq, nx, nx))
        """State vector covariance derivative."""
    
    def _get_initial(self, key, options, shape):
        try:
            return np.asarray(options[key])
        except KeyError:
            try:
                attr = key.replace('x', 'x0')
                return getattr(self.model, attr)()
            except AttributeError:
                return np.zeros(self.base_shape + shape)
    
    @abc.abstractmethod
    def predict(self):
        """Predict the state distribution at a next time sample."""
        raise NotImplementedError("Pure abstract method.")
    
    @abc.abstractmethod
    def correct(self, y):
        """Correct the state distribution, given the measurement vector."""
        raise NotImplementedError("Pure abstract method.")

    def smoother_correction(self, xpred, Pxpred, Pxf, xsmooth, Pxsmooth):
        PxIpred = np.linalg.inv(Pxpred)
        K = np.einsum('...ij,...jk', Pxf, PxIpred)
        e = xsmooth - xpred
        x_inc = np.einsum('...ij,...j', K, e)
        Px_inc = np.einsum('...ij,...jk,...lk', K, Pxsmooth - Pxpred, K)
        return x_inc, Px_inc
    
    def filter(self, y):
        y = np.asanyarray(y)
        N = len(y)
        x = np.zeros((N,) + self.x.shape)
        Px = np.zeros((N,) + self.x.shape + (self.model.nx,))
        
        for k in range(N):
            x[k], Px[k] = self.correct(y[k])
            if k < N - 1:
                self.predict()
        
        return x, Px

    def smooth(self, y):
        y = np.asanyarray(y)
        N = len(y)
        x = np.zeros((N,) + np.shape(self.x))
        xpred = np.zeros_like(x)
        Px = np.zeros((N,) + np.shape(self.x) + (self.model.nx,))
        Pxpred = np.zeros_like(Px)
        Pxfpred = np.zeros((N - 1,) + np.shape(self.x) + (self.model.nx,))
        
        xpred[0] = self.x
        Pxpred[0] = self.Px
        for k in range(N):
            x[k], Px[k] = self.correct(y[k])
            if k < N - 1:
                xpred[k+1], Pxpred[k+1] = self.predict()
                Pxfpred[k] = self.prediction_crosscov()
        
        for k in reversed(range(1, N)):
            x_inc, Px_inc = self.smoother_correction(
                xpred[k], Pxpred[k], Pxfpred[k-1], x[k], Px[k]
            )
            x[k - 1] += x_inc
            Px[k - 1] += Px_inc
        
        return x, Px
    
    def pem_merit(self, y):
        y = np.asanyarray(y)
        N = len(y)
        
        for k in range(N):
            self.correct(y[k])
            self.update_likelihood()
            if k < N - 1:
                self.predict()
        
        return self.L
    
    def pem_gradient(self, y):
        y = np.asanyarray(y)
        N = len(y)
        
        for k in range(N):
            self.correct(y[k])
            self.correction_diff()
            self.update_likelihood()
            self.likelihood_diff()
            if k < N - 1:
                self.predict()
                self.prediction_diff()
        
        return self.dL_dq

    def pem_hessian(self, y):
        y = np.asanyarray(y)
        N = len(y)
        
        for k in range(N):
            self.correct(y[k])
            self.correction_diff()
            self.correction_diff2()
            self.update_likelihood()
            self.likelihood_diff()
            self.likelihood_diff2()
            if k < N - 1:
                self.predict()
                self.prediction_diff()
                self.prediction_diff2()
        
        return self.d2L_dq2
