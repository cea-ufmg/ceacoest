"""Kalman filtering / smoothing module.

TODO
----
* Add derivative of SVD square root.
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

from . import utils


class DTKalmanFilterBase(metaclass=abc.ABCMeta):
    """Discrete-time Kalman filter/smoother abstract base class."""
    
    def __init__(self, model, x, Px, **options):
        """Create a discrete-time Kalman filter.
        
        Parameters
        ----------
        model :
            The underlying system model.
        
        """
        self.model = model
        """The underlying system model."""
        
        self.x = np.asarray(x)
        """State vector mean."""
        
        self.Px = np.asarray(Px)
        """State vector covariance."""
        
        self.k = options.get('k', 0)
        """Time index."""
        
        self.L = options.get('L', 0.0)
        """Measurement log-likelihood."""
        
        nq = model.nq
        nx = model.nx
        base_shape = self.x.shape[:-1]
        self.base_shape = base_shape
        """Base shape of broadcasting."""
        
        self.dL_dq = options.get('dL_dq', np.zeros(base_shape + (nq,)))
        """Measurement log-likelihood derivative."""

        self.d2L_dq2 = options.get('dL_dq', np.zeros(base_shape + (nq, nq)))
        """Measurement log-likelihood derivative."""

        self.dx_dq = options.get('dx_dq', np.zeros(base_shape + (nq, nx)))
        """State vector derivative."""
        
        self.dPx_dq = options.get('dPx_dq', np.zeros(base_shape + (nq, nx, nx)))
        """State vector covariance derivative."""

        self.d2x_dq2 = options.get('dx_dq', np.zeros(base_shape + (nq, nq, nx)))
        """State vector derivative."""
        
        self.d2Px_dq2 = options.get('dPx_dq',np.zeros(base_shape+(nq,nq,nx,nx)))
        """State vector covariance derivative."""
    
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


class UnscentedTransformBase(metaclass=abc.ABCMeta):
    """Unscented transform base class."""
    
    def __init__(self, ni, **options):
        """Unscented transform object constructor.
        
        Parameters
        ----------
        ni : int
            Number of inputs
        
        Options
        -------
        kappa :
            Weight of the center sigma point. Zero by default.
        
        """
        self.ni = ni
        """Number of inputs."""
        
        self.kappa = options.get('kappa', 0.0)
        """Weight parameter of the center sigma point."""
        assert self.ni + self.kappa != 0
        
        self.nsigma = 2 * ni + (self.kappa != 0)
        """Number of sigma points."""
        
        weights = np.repeat(0.5 / (ni + self.kappa), self.nsigma)
        if self.kappa != 0:
            weights[-1] = self.kappa / (ni + self.kappa)
        self.weights = weights
        """Transform weights."""
    
    @abc.abstractmethod
    def sqrt(self, Q):
        """Unscented transform square root method."""
        raise NotImplementedError("Pure abstract method.")
    
    def sigma_points(self, i, Pi):
        """Generate sigma-points and their deviations.
        
        The sigma points are the first dimension of the returned array.
        """
        ni = self.ni
        S = np.rollaxis(self.sqrt((ni + self.kappa) * Pi), -2)
        idev = np.zeros((self.nsigma,) + np.shape(i))
        idev[:ni] = S
        idev[ni:(2 * ni)] = -S
        isigma = idev + i
        
        self.isigma = isigma
        self.idev = idev
        return isigma
    
    def sigma_points_diff(self, di_dq, dPi_dq):
        """Derivative of sigma-points."""
        ni = self.ni
        
        dS_dq = self.sqrt_diff((ni + self.kappa) * dPi_dq)
        didev_dq = np.zeros((self.nsigma,) + di_dq.shape)
        didev_dq[:ni] = np.rollaxis(dS_dq, -2)
        didev_dq[ni:(2 * ni)] = -didev_dq[:ni]
        disigma_dq = didev_dq + di_dq
        
        self.didev_dq = didev_dq
        self.disigma_dq = disigma_dq
        return disigma_dq

    def sigma_points_diff2(self, d2i_dq2, d2Pi_dq2):
        """Derivative of sigma-points."""
        ni = self.ni
        
        d2S_dq2 = self.sqrt_diff2((ni + self.kappa) * d2Pi_dq2)
        d2idev_dq2 = np.zeros((self.nsigma,) + d2i_dq2.shape)
        d2idev_dq2[:ni] = np.rollaxis(d2S_dq2, -2)
        d2idev_dq2[ni:(2 * ni)] = -d2idev_dq2[:ni]
        d2isigma_dq2 = d2idev_dq2 + d2i_dq2
        self.d2idev_dq2 = d2idev_dq2
        return d2isigma_dq2
    
    def transform(self, i, Pi, f):
        isigma = self.sigma_points(i, Pi)
        weights = self.weights
        
        osigma = f(isigma)
        o = np.einsum('k,k...', weights, osigma)
        odev = osigma - o
        Po = np.einsum('k...i,k...j,k', odev, odev, weights)
        
        self.osigma = osigma
        self.odev = odev
        self.o = o
        self.Po = Po
        return (o, Po)
    
    def transform_diff(self, df_dq, df_di, di_dq, dPi_dq):
        weights = self.weights
        isigma = self.isigma
        odev = self.odev
        self.df_di = df_di(isigma)
        
        disigma_dq = self.sigma_points_diff(di_dq, dPi_dq)
        Dosigma_Dq = df_dq(isigma)
        Dosigma_Dq += np.einsum('k...if,k...ai->k...af', self.df_di, disigma_dq)
        
        do_dq = np.einsum('k,k...', weights, Dosigma_Dq)
        dodev_dq = Dosigma_Dq - do_dq
        dPo_dq = np.einsum('k...lj,k...i,k->...lij', dodev_dq, odev, weights)
        dPo_dq += np.swapaxes(dPo_dq, -1, -2)
        
        self.disigma_dq = disigma_dq
        self.dodev_dq = dodev_dq
        self.do_dq = do_dq
        self.dPo_dq = dPo_dq
        return (do_dq, dPo_dq)
    
    def transform_diff2(self, d2f_dq2, d2f_di2, d2f_di_dq, d2i_dq2, d2Pi_dq2):
        weights = self.weights
        odev = self.odev
        isigma = self.isigma
        disigma_dq = self.disigma_dq
        dodev_dq = self.dodev_dq
        
        
        d2isigma_dq2 = self.sigma_points_diff2(d2i_dq2, d2Pi_dq2)
        D2osigma_Dq2 = np.einsum('k...aif,k...bi->k...baf',
                                 d2f_di_dq(isigma), disigma_dq)
        D2osigma_Dq2 += np.swapaxes(D2osigma_Dq2, -2, -3)
        D2osigma_Dq2 += d2f_dq2(isigma)
        D2osigma_Dq2 += np.einsum('k...ijf,k...ai,k...bj->k...baf',
                                  d2f_di2(isigma), disigma_dq, disigma_dq)
        D2osigma_Dq2 += np.einsum('k...if,k...bai->k...baf', 
                                  self.df_di, d2isigma_dq2)
        
        d2o_dq2 = np.einsum('k,k...', weights, D2osigma_Dq2)
        d2odev_dq2 = D2osigma_Dq2 - d2o_dq2
        d2Po_dq2 = np.einsum('k...abi,k...j,k', d2odev_dq2, odev, weights)
        d2Po_dq2 += np.einsum('k...bi,k...aj,k', dodev_dq, dodev_dq, weights)
        d2Po_dq2 += np.swapaxes(d2Po_dq2, -1, -2)
        self.d2odev_dq2 = d2odev_dq2
        return (d2o_dq2, d2Po_dq2)
    
    def crosscov(self):
        return np.einsum('k...i,k...j,k', self.idev, self.odev, self.weights)
    
    def crosscov_diff(self):
        dPio_dq = np.einsum('k...ai,k...j,k', 
                            self.didev_dq, self.odev, self.weights)
        dPio_dq += np.einsum('k...i,k...aj,k',
                             self.idev, self.dodev_dq, self.weights)
        return dPio_dq

    def crosscov_diff2(self):
        d2Pio_dq2 = np.einsum('k...bi,k...aj,k',
                              self.didev_dq, self.dodev_dq, self.weights)
        d2Pio_dq2 += np.swapaxes(d2Pio_dq2, -3, -4)
        d2Pio_dq2 += np.einsum('k...abi,k...j,k',
                               self.d2idev_dq2, self.odev, self.weights)
        d2Pio_dq2 += np.einsum('k...i,k...abj,k',
                               self.idev, self.d2odev_dq2, self.weights)
        return d2Pio_dq2


class SVDUnscentedTransform(UnscentedTransformBase):
    """Unscented transform using singular value decomposition."""
    
    def sqrt(self, Q):
        """Unscented transform square root method."""
        [U, s, VT] = np.linalg.svd(Q)
        self.S = np.einsum('...ij,...j->...ji', U, np.sqrt(s))
        return self.S


class DifferentiableCholesky:
    
    diff_data_initialized = 0

    def initialize_diff_data(self, ni):
        if self.diff_data_initialized == ni:
            return
        
        self.ni_range = np.arange(ni)
        self.ni_tril_indices = np.tril_indices(ni)
        self.diff_data_initialized = ni
 
    def __call__(self, Q):
        """Perform the upper triangular Cholesky decomposition."""
        self.S = np.swapaxes(np.linalg.cholesky(Q), -1, -2)
        return self.S
    
    def diff(self, dQ_dq):
        """Derivatives of upper triangular Cholesky decomposition.
        
        Parameters
        ----------
        dQ_dq : (..., nq, ni, ni) array_like
            The derivatives of `Q` with respect to some parameter vector.
             Must be symmetric with respect to the last two axes, i.e., 
            `dQ_dq[..., i, j] == dQ_dq[..., j, i]` for all `i, j` pairs.
        
        Returns
        -------
        dS_dq : (..., nq, ni, ni) array
            The derivative of the Cholesky decomposition of `Q` with respect
            to the parameter vector.
        
        """
        nq = np.size(dQ_dq, -3)
        ni = np.size(self.S, -1)
        self.initialize_diff_data(ni)
        
        k = self.ni_range
        i, j = self.ni_tril_indices
        ix, jx, kx = np.ix_(i, j, k)
        
        A = np.zeros(self.S.shape + (ni, ni))
        A[..., ix, jx, ix, kx] = self.S[..., kx, jx]
        A[..., ix, jx, jx, kx] += self.S[..., kx, ix]
        AL = A[..., i, j, :, :][..., i, j]
        ALI = np.linalg.inv(AL)
        
        dQL_dq = dQ_dq[..., i, j]
        dSL_dq = np.einsum('...ij,...bj', ALI, dQL_dq)
        dS_dq = np.zeros_like(dQ_dq)
        dS_dq[..., j, i] = dSL_dq
        
        self.dQL_dq = dQL_dq
        self.dS_dq = dS_dq
        self.AL = AL
        self.ALI = ALI
        return dS_dq
    
    def diff2(self, d2Q_dq2):
        """Second derivatives of upper triangular Cholesky decomposition."""
        nq = np.size(d2Q_dq2, -3)
        ni = np.size(self.S, -1)
        self.initialize_diff_data(ni)
        
        k = self.ni_range
        i, j = self.ni_tril_indices
        ix, jx, kx = np.ix_(i, j, k)
        
        dA_dq = np.zeros(self.dS_dq.shape + (ni, ni))
        dA_dq[..., ix, jx, ix, kx] = self.dS_dq[..., kx, jx]
        dA_dq[..., ix, jx, jx, kx] += self.dS_dq[..., kx, ix]
        dAL_dq = dA_dq[..., i, j, :, :][..., i, j]
        dALI_dq = -np.einsum('...ij,...ajk,...kl', self.ALI, dAL_dq, self.ALI)

        d2QL_dq2 = d2Q_dq2[..., i, j]
        d2SL_dq2 = np.einsum('...ij,...abj', self.ALI, d2QL_dq2)
        d2SL_dq2 += np.einsum('...aij,...bj', dALI_dq, self.dQL_dq)
        d2S_dq2 = np.zeros(self.S.shape[:-2] + (nq, nq, ni, ni))
        d2S_dq2[..., j, i] = d2SL_dq2
        return d2S_dq2


class CholeskyUnscentedTransform(UnscentedTransformBase):
    """Unscented transform using Cholesky decomposition."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cholesky = DifferentiableCholesky()
    
    def sqrt(self, Q):
        """Unscented transform square root method."""
        return self.cholesky(Q)
    
    def sqrt_diff(self, dQ_dq):
        """Derivatives of unscented transform Cholesky decomposition."""
        return self.cholesky.diff(dQ_dq)

    def sqrt_diff2(self, d2Q_dq2):
        """Second derivatives of unscented transform Cholesky decomposition."""
        return self.cholesky.diff2(d2Q_dq2)


def choose_ut_transform_class(options):
    """Choose an unscented transform class from an options dict."""
    sqrt = options.get('sqrt', 'cholesky')
    if sqrt == 'cholesky':
        return CholeskyUnscentedTransform
    elif sqrt == 'svd':
        return SVDUnscentedTransform
    else:
        raise ValueError("Invalid value for `sqrt` option.")


class DTUnscentedPredictor(DTKalmanFilterBase):
    
    def __init__(self, model, x, Px, **options):
        # Initialize base
        super().__init__(model, x, Px, **options)
        
        # Get transform options
        ut_options = options.copy()
        ut_options.update(utils.extract_subkeys(options, 'pred_ut_'))
        
        # Create the transform object
        UTClass = choose_ut_transform_class(ut_options)
        self.__ut = UTClass(model.nx, **ut_options)
    
    def predict(self):
        """Predict the state distribution at the next time index."""
        def f_fun(x):
            return self.model.f(self.k, x)
        
        f, Pf = self.__ut.transform(self.x, self.Px, f_fun)
        Q = self.model.Q(self.k, self.x)
        
        self.prev_x = self.x
        self.prev_Px = self.Px
        self.k += 1
        self.x = f
        self.Px = Pf + Q
        return self.x, self.Px
    
    def prediction_diff(self):
        """Calculate the derivatives of the prediction."""
        k = self.k - 1 
        x = self.prev_x
        
        def df_dq_fun(x):
            return self.model.df_dq(k, x)
        def df_dx_fun(x):
            return self.model.df_dx(k, x)
        Df_Dq, DPf_Dq = self.__ut.transform_diff(
            df_dq_fun, df_dx_fun, self.dx_dq, self.dPx_dq
        )
        dQ_dq = self.model.dQ_dq(k, x)
        dQ_dx = self.model.dQ_dx(k, x)
        DQ_Dq = dQ_dq + np.einsum('...ij,...jkl', self.dx_dq, dQ_dx)
        
        self.dQ_dx = dQ_dx
        self.prev_dx_dq = self.dx_dq
        self.dx_dq = Df_Dq
        self.dPx_dq = DPf_Dq + DQ_Dq

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


class DTUnscentedCorrector(DTKalmanFilterBase):
    
    def __init__(self, model, x, Px, **options):
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


class DTUnscentedKalmanFilter(DTUnscentedPredictor, DTUnscentedCorrector):
    pass

