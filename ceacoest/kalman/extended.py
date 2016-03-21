"""Extended Kalman filtering / smoothing module.

TODO
----
* Make docstrings for all constructors.

Improvement ideas
-----------------

"""


import abc
import collections
import re

import attrdict
import numpy as np
import numpy.ma as ma
import numpy.linalg

from .. import utils
from . import base


class DTPredictor(base.DTKalmanFilter):
        
    def predict(self):
        """Predict the state distribution at the next time index."""
        f = self.model.f(self.k, self.x)
        df_dx = self.model.df_dx(self.k, self.x)
        Pf = np.einsum('...ai,...ij,...bj', df_dx, self.Px, df_dx)
        Q = self.model.Q(self.k, self.x)
        
        self.prev_x = self.x
        self.prev_Px = self.Px
        self.k += 1
        self.x = f
        self.Px = Pf + Q
        self.df_dx = df_dx
        return self.x, self.Px
    
    def prediction_diff(self):
        """Calculate the derivatives of the prediction."""
        k = self.k - 1 
        x = self.prev_x
        dx_dq = self.dx_dq
        dPx_dq = self.dPx_dq
        
        df_dq = self.model.df_dq(k, x)
        d2f_dx_dq = self.model.d2f_dx_dq(k, x)
        d2f_dx2 = self.model.d2f_dx2(k, x)
        dQ_dq = self.model.dQ_dq(k, x)
        dQ_dx = self.model.dQ_dx(k, x)
        
        Df_Dq = df_dq + np.einsum('...ib,...ai', df_dx, dx_dq)
        dPf_dq = np.einsum('...aci,...ij,...bj', d2f_dx_dq, dPx_dq, df_dx)
        dPf_dq += np.einsum(
            '...ak,...kci,...ij,...bj', dx_dq, d2f_dx2, dPx_dq, df_dx
        )
        dPf_dq += np.swapaxes(dPf_dq, -1, -2)
        dPf_dq += np.einsum('...ci,...aij,...bj', df_dx, dPx_dq, df_dx)
        DQ_Dq = dQ_dq + np.einsum('...ij,...jkl', self.dx_dq, dQ_dx)
        
        self.dQ_dx = dQ_dx
        self.prev_dx_dq = self.dx_dq
        self.dx_dq = Df_Dq
        self.dPx_dq = dPf_dq + DQ_Dq
    
    def prediction_diff2(self):
        """Calculate the second derivatives of the prediction."""
        k = self.k - 1
        x = self.prev_x
        dx_dq = self.prev_dx_dq
        
        def d2f_dq2_fun(x):
            return self.model.d2f_dq2(k, x)
        def d2f_dx2_fun(x):
            return self.model.d2f_dx2(k, x)
        def d2f_dx_dq_fun(x):
            return self.model.d2f_dx_dq(k, x)
        D2f_Dq2, D2Pf_Dq2 = self.__ut.transform_diff2(
            d2f_dq2_fun, d2f_dx2_fun, d2f_dx_dq_fun, self.d2x_dq2, self.d2Px_dq2
        )
        d2Q_dx_dq = self.model.d2Q_dx_dq(k, x)
        D2Q_Dq2 = np.einsum('...bi,...aikl', dx_dq, d2Q_dx_dq)
        D2Q_Dq2 += np.swapaxes(D2Q_Dq2, -3, -4)
        D2Q_Dq2 += self.model.d2Q_dq2(k, x)
        D2Q_Dq2 += np.einsum('...abi,...ikl', self.d2x_dq2, self.dQ_dx)
        D2Q_Dq2 += np.einsum('...bi,...jikl,...aj',
                             dx_dq, self.model.d2Q_dx2(k, x), dx_dq)
        self.d2x_dq2 = D2f_Dq2
        self.d2Px_dq2 = D2Pf_Dq2 + D2Q_Dq2

    def prediction_crosscov(self):
        return self.__ut.crosscov()


class DTCorrector(base.DTKalmanFilter):
    
    def __init__(self, model, x=None, Px=None, **options):
        # Initialize base
        super().__init__(model, x, Px, **options)
        
        # Get transform options
        ut_options = options.copy()
        ut_options.update(utils.extract_subkeys(options, 'corr_ut_'))
        
        # Create the transform object
        UTClass = choose_ut_transform_class(ut_options)
        self.__ut = UTClass(model.nx, **ut_options)
    
    def correct(self, y):
        """Correct the state distribution, given the measurement vector."""
        # Get the y-mask
        mask = ma.getmaskarray(y)
        self.active = active = ~mask
        if np.all(mask):
            return self.x, self.Px
        
        # Remove inactive outputs
        y = ma.compressed(y)
        R = self.model.R()[np.ix_(active, active)]
        def h_fun(x):
            return self.model.h(self.k, x)[..., active]
        
        # Perform unscented transform
        h, Ph = self.__ut.transform(self.x, self.Px, h_fun)
        Pxh = self.__ut.crosscov()
        
        # Factorize covariance
        self.__chol = DifferentiableCholesky()
        Py = Ph + R
        PyC = self.__chol(Py)
        PyCI = np.linalg.inv(PyC)
        PyI = np.einsum('...ik,...jk', PyCI, PyCI)
        
        # Perform correction
        e = y - h
        K = np.einsum('...ik,...kj', Pxh, PyI)
        x_corr = self.x + np.einsum('...ij,...j', K, e)
        Px_corr = self.Px - np.einsum('...ik,...jl,...kl', K, K, Py)
        
        # Save and return the correction data
        self.prev_x = self.x
        self.prev_Px = self.Px
        self.e = e
        self.x = x_corr
        self.Px = Px_corr
        self.Pxh = Pxh
        self.Py = Py
        self.PyI = PyI
        self.PyC = PyC
        self.PyCI = PyCI
        self.K = K
        return x_corr, Px_corr

    def correction_diff(self):
        """Calculate the derivatives of the correction."""
        if not np.any(self.active):
            return
        
        # Get the model and transform derivatives
        def dh_dq_fun(x):
            return self.model.dh_dq(self.k, x)[..., self.active]
        def dh_dx_fun(x):
            return self.model.dh_dx(self.k, x)[..., self.active]
        Dh_Dq, DPh_Dq = self.__ut.transform_diff(
            dh_dq_fun, dh_dx_fun, self.dx_dq, self.dPx_dq
        )
        dPxh_dq = self.__ut.crosscov_diff()
        dR_dq = self.model.dR_dq()[(...,) + np.ix_(self.active, self.active)]

        # Calculate the correction derivatives
        de_dq = -Dh_Dq
        dPy_dq = DPh_Dq + dR_dq
        dPyI_dq = -np.einsum('...ij,...ajk,...kl', self.PyI, dPy_dq, self.PyI)
        dK_dq = np.einsum('...ik,...akj', self.Pxh, dPyI_dq)
        dK_dq += np.einsum('...aik,...kj', dPxh_dq, self.PyI)

        self.de_dq = de_dq
        self.dK_dq = dK_dq
        self.dPy_dq = dPy_dq
        self.dPyI_dq = dPyI_dq
        self.dPxh_dq = dPxh_dq
        self.prev_dx_dq = self.dx_dq.copy()
        self.prev_dPx_dq = self.dPx_dq.copy()
        self.dx_dq += np.einsum('...aij,...j', dK_dq, self.e)
        self.dx_dq += np.einsum('...ij,...aj', self.K, de_dq)
        self.dPx_dq -= np.einsum('...ik,...jl,...alk', self.K, self.K, dPy_dq)
        dK_dq__K__Py = np.einsum('...aik,...jl,...lk', dK_dq, self.K, self.Py)
        self.dPx_dq -= dK_dq__K__Py + np.swapaxes(dK_dq__K__Py, -1, -2)
    
    def correction_diff2(self):
        """Calculate the second derivatives of the correction."""
        if not np.any(self.active):
            return

        # Get some saved data
        K = self.K
        PyI = self.PyI
        dK_dq = self.dK_dq
        dPy_dq = self.dPy_dq
        dPyI_dq = self.dPyI_dq
        
        # Get the model and transform derivatives
        def d2h_dq2_fun(x):
            return self.model.d2h_dq2(self.k, x)[..., self.active]
        def d2h_dx2_fun(x):
            return self.model.d2h_dx2(self.k, x)[..., self.active]
        def d2h_dx_dq_fun(x):
            return self.model.d2h_dx_dq(self.k, x)[..., self.active]
        D2h_Dq2, D2Ph_Dq2 = self.__ut.transform_diff2(
            d2h_dq2_fun, d2h_dx2_fun, d2h_dx_dq_fun, self.d2x_dq2, self.d2Px_dq2
        )
        d2Pxh_dq2 = self.__ut.crosscov_diff2()
        d2R_dq2 = self.model.d2R_dq2()[(...,) + np.ix_(self.active,self.active)]
        
        # Calculate the correction derivatives
        d2e_dq2 = -D2h_Dq2
        d2Py_dq2 = D2Ph_Dq2 + d2R_dq2
        d2PyI_dq2 = -np.einsum('...aij,...bjk,...kl', dPyI_dq, dPy_dq, PyI)
        d2PyI_dq2 -= np.einsum('...ij,...abjk,...kl', PyI, d2Py_dq2, PyI)
        d2PyI_dq2 -= np.einsum('...ij,...bjk,...akl', PyI, dPy_dq, dPyI_dq)
        d2K_dq2 = np.einsum('...aik,...bkj', self.dPxh_dq, dPyI_dq)
        d2K_dq2 += np.einsum('...ik,...abkj', self.Pxh, d2PyI_dq2)
        d2K_dq2 += np.einsum('...abik,...kj', d2Pxh_dq2, PyI)
        d2K_dq2 += np.einsum('...bik,...akj', self.dPxh_dq, dPyI_dq)

        self.d2x_dq2 += np.einsum('...abij,...j', d2K_dq2, self.e)
        self.d2x_dq2 += np.einsum('...bij,...aj', dK_dq, self.de_dq)
        self.d2x_dq2 += np.einsum('...aij,...bj', dK_dq, self.de_dq)
        self.d2x_dq2 += np.einsum('...ij,...abj', K, d2e_dq2)
        self.d2Px_dq2 -= np.einsum('...abik,...jl,...lk', d2K_dq2, K, self.Py)
        self.d2Px_dq2 -= np.einsum('...bik,...ajl,...lk', dK_dq, dK_dq, self.Py)
        self.d2Px_dq2 -= np.einsum('...aik,...bjl,...lk', dK_dq, dK_dq, self.Py)
        dK_dq__K__dPy_dq = np.einsum('...bik,...jl,...alk', dK_dq, K, dPy_dq)
        dK_dq__K__dPy_dq += np.swapaxes(dK_dq__K__dPy_dq, -2, -1)
        dK_dq__K__dPy_dq += np.swapaxes(dK_dq__K__dPy_dq, -3, -4)
        self.d2Px_dq2 -= dK_dq__K__dPy_dq
        self.d2Px_dq2 -= np.einsum('...ik,...abjl,...lk', K, d2K_dq2, self.Py)
        self.d2Px_dq2 -= np.einsum('...ik,...jl,...ablk', K, K, d2Py_dq2)
        self.d2e_dq2 = d2e_dq2
        self.d2Py_dq2 = d2Py_dq2
        self.d2PyI_dq2 = d2PyI_dq2
    
    def update_likelihood(self):
        """Update measurement log-likelihood."""
        if not np.any(self.active):
            return
        
        self.PyCD = np.einsum('...kk->...k', self.PyC)
        self.L -= 0.5 * np.einsum('...i,...ij,...j', self.e, self.PyI, self.e) 
        self.L -= np.log(self.PyCD).sum(-1)
    
    def likelihood_diff(self):
        """Calculate measurement log-likelihood derivatives."""
        if not np.any(self.active):
            return
        
        # Get the work variables
        e = self.e
        PyI = self.PyI
        de_dq = self.de_dq
        dPyI_dq = self.dPyI_dq
        
        # Calculate the likelihood derivatives
        dPyC_dq = self.__chol.diff(self.dPy_dq)
        self.dPyCD_dq = np.einsum('...kk->...k', dPyC_dq)
        self.dL_dq -= np.sum(self.dPyCD_dq / self.PyCD[..., None, :], axis=-1)
        self.dL_dq -= np.einsum('...ai,...ij,...j', de_dq, PyI, e)
        self.dL_dq -= 0.5 * np.einsum('...i,...aij,...j', e, dPyI_dq, e)
    
    def likelihood_diff2(self):
        """Calculate measurement log-likelihood derivatives."""
        if not np.any(self.active):
            return
        
        # Get the work variables
        e = self.e
        PyI = self.PyI
        PyCD = self.PyCD
        de_dq = self.de_dq
        dPyI_dq = self.dPyI_dq
        dPyCD_dq = self.dPyCD_dq
        d2e_dq2 = self.d2e_dq2
        d2PyI_dq2 = self.d2PyI_dq2
        
        # Calculate the likelihood derivatives
        d2PyC_dq2 = self.__chol.diff2(self.d2Py_dq2)
        d2PyCD_dq2 = np.einsum('...kk->...k', d2PyC_dq2)
        self.d2L_dq2 -= np.sum(d2PyCD_dq2 / PyCD[..., None, None, :], axis=-1)
        self.d2L_dq2 += np.einsum('...ak,...bk', dPyCD_dq,
                                  dPyCD_dq / PyCD[..., None, :]**2)
        self.d2L_dq2 -= np.einsum('...ai,...ij,...bj', de_dq, PyI, de_dq)
        self.d2L_dq2 -= np.einsum('...abi,...ij,...j', d2e_dq2, PyI, e)
        self.d2L_dq2 -= 0.5 * np.einsum('...i,...abij,...j', e, d2PyI_dq2, e)
        de_dq__dPyI_dq__e = np.einsum('...ai,...bij,...j', de_dq, dPyI_dq, e)
        self.d2L_dq2 -= de_dq__dPyI_dq__e
        self.d2L_dq2 -= np.swapaxes(de_dq__dPyI_dq__e, -1, -2)


class DTKalmanFilter(DTPredictor, DTCorrector):
    pass

