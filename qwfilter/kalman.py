"""Kalman filtering / smoothing module.

TODO
----
* Add derivative of SVD square root.
* Vectorize the UT functions.
* Make docstrings for all constructors.
* Implement filter Hessian.

Improvement ideas
-----------------
* Allow gradients and Hessian to be calculated offline, saving processing time
  at the cost of memory.

"""


import abc
import collections
import re

import attrdict
import numpy as np
import numpy.ma as ma
import numpy.linalg
import scipy.linalg

from . import utils


class DTKalmanFilterWork:
    """Kalman filter work data."""
    
    def __init__(self, x, Px, k=0, **variables):
        self.x = np.asarray(x)
        """Current state vector mean."""
        
        self.Px = np.asarray(Px)
        """Current state vector covariance."""
        
        self.k = k
        """Current time step."""
        
        for name, value in variables.items():
            setattr(self, name, value)


class DTKalmanFilterBase(metaclass=abc.ABCMeta):
    """Discrete-time Kalman filter/smoother abstract base class.

    Due to the various use cases of Kalman filters (e.g. online filtering,
    offline filtering, smoothing, prediction error method parameter estimation,
    importance sampling for particle filters, etc) this class and subclasses
    retains only information on the procedures. The data should be managed by
    other classes such as those holding the filter state and filter history.
    """
    
    Work = DTKalmanFilterWork
    """Work data class."""
    
    def __init__(self, model, **options):
        """Create a discrete-time Kalman filter.
        
        Parameters
        ----------
        model :
            The underlying system model.
        
        """
        self.model = model
        """The underlying system model."""
    
    @abc.abstractmethod
    def predict(self, work):
        """Predict the state distribution at a next time sample."""
        raise NotImplementedError("Pure abstract method.")
    
    @abc.abstractmethod
    def correct(self, work, y):
        """Correct the state distribution, given the measurement vector."""
        raise NotImplementedError("Pure abstract method.")
    
    def filter(self, work, y):
        y = np.asanyarray(y)
        N = len(y)
        x = np.zeros((N,) + np.shape(work.x))
        Px = np.zeros((N,) + np.shape(work.x) + (self.model.nx,))
        
        for k in range(N):
            x[k], Px[k] = self.correct(work, y[k])
            if k < N - 1:
                self.predict(work)
        
        return x, Px

    def pem_merit(self, work, y):
        y = np.asanyarray(y)
        N = len(y)
        
        if not hasattr(work, 'L'):
            work.L = 0.0

        for k in range(N):
            self.correct(work, y[k])
            self.update_likelihood(work)
            if k < N - 1:
                self.predict(work)
        
        return work.L

    def pem_gradient(self, work, y):
        y = np.asanyarray(y)
        N = len(y)
        nx  = self.model.nx
        nq = self.model.nq
        
        if not hasattr(work, 'L'):
            work.L = 0.0
        if not hasattr(work, 'dL_dq'):
            work.dL_dq = np.zeros(nq)
        if not hasattr(work, 'dx_dq'):
            work.dx_dq = np.zeros((nq, nx))
        if not hasattr(work, 'dPx_dq'):
            work.dPx_dq = np.zeros((nq, nx, nx))
        
        for k in range(N):
            self.correct(work, y[k])
            self.correction_diff(work)
            self.update_likelihood(work)
            self.likelihood_diff(work)
            if k < N - 1:
                self.predict(work)
                self.prediction_diff(work)
        
        return work.dL_dq


def cholesky_sqrt_diff2(S, d2Q, work):
    """Second derivatives of lower triangular Cholesky decomposition."""
    S = np.asarray(S)
    dQ = work['dQ']
    dS = work['dS']
        
    n = S.shape[-1]
    m = dQ.shape[0]
    (i, j, k) = work['i j k']
    (ix, jx, kx) = work['ix jx kx']
    dQ_tril = work['dQ_tril']
    A_tril = work['A_tril']
    A_tril_inv = work['A_tril_inv']
    
    dA = np.zeros((m, n, n, n, n))
    dA[:, ix, jx, ix, kx] = dS[:, kx, jx]
    dA[:, ix, jx, jx, kx] += dS[:, kx, ix]
    dA_tril = dA[:, i, j][..., i, j]
    dA_tril_inv = -np.einsum('ij,ajk,kl', A_tril_inv, dA_tril, A_tril_inv)

    d2Q_tril = d2Q[..., i, j]
    d2S_tril = np.einsum('aij,...j->a...i', dA_tril_inv, dQ_tril)
    d2S_tril += np.einsum('ij,...j->...i', A_tril_inv, d2Q_tril)
    d2S = np.zeros(d2Q_tril.shape[:-1] + (n, n))
    d2S[..., j, i] = d2S_tril
    return d2S


class UnscentedTransformWork:
    """Unscented transform work data."""
    def __init__(self, i, Pi):
        self.i = np.asarray(i)
        self.Pi = np.asarray(Pi)


class UnscentedTransformBase(metaclass=abc.ABCMeta):
    """Unscented transform base class."""
    
    Work = UnscentedTransformWork
    """Work data class."""
    
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
    def sqrt(self, work, Q):
        """Unscented transform square root method."""
        raise NotImplementedError("Pure abstract method.")
    
    def sigma_points(self, work):
        """Generate sigma-points and their deviations.
        
        The sigma points are the lines of the returned matrix.
        """
        ni = self.ni
        S = self.sqrt(work, (ni + self.kappa) * work.Pi)
        idev = np.zeros((self.nsigma, ni))
        idev[:ni] = S
        idev[ni:(2 * ni)] = -S
        isigma = idev + work.i
        
        work.isigma = isigma
        work.idev = idev
        return isigma
    
    def sigma_points_diff(self, work, di_dq, dPi_dq):
        """Derivative of sigma-points."""
        ni = self.ni
        
        dS_dq = self.sqrt_diff(work, (ni + self.kappa) * dPi_dq)
        didev_dq = np.zeros((self.nsigma,) + di_dq.shape)
        didev_dq[:ni] = np.rollaxis(dS_dq, -2)
        didev_dq[ni:(2 * ni)] = -didev_dq[:ni]
        disigma_dq = didev_dq + di_dq
        
        work.didev_dq = didev_dq
        work.disigma_dq = disigma_dq
        return disigma_dq
    
    def transform(self, work, f):
        isigma = self.sigma_points(work)
        weights = self.weights
        
        osigma = f(isigma)
        o = np.einsum('k,ki', weights, osigma)
        odev = osigma - o
        Po = np.einsum('ki,kj,k', odev, odev, weights)
        
        work.osigma = osigma
        work.odev = odev
        work.o = o
        work.Po = Po
        return (o, Po)
    
    def transform_diff(self, work, df_dq, df_dx, di_dq, dPi_dq):
        weights = self.weights
        isigma = work.isigma
        
        disigma_dq = self.sigma_points_diff(work, di_dq, dPi_dq)
        Dosigma_Dq = np.einsum('ijk,ilj->ilk', df_dx(isigma), disigma_dq)
        Dosigma_Dq += df_dq(isigma)
        
        do_dq = np.einsum('k,k...', weights, Dosigma_Dq)
        dodev_dq = Dosigma_Dq - do_dq
        dPo_dq = np.einsum('klj,ki,k->lij', dodev_dq, work.odev, weights)
        dPo_dq += np.swapaxes(dPo_dq, -1, -2)
        
        work.dodev_dq = dodev_dq
        work.do_dq = do_dq
        work.dPo_dq = dPo_dq
        return (do_dq, dPo_dq)
    
    def crosscov(self, work):
        return np.einsum('ki,kj,k', work.idev, work.odev, self.weights)
    
    def crosscov_diff(self, work):
        dPio_dq = np.einsum('kli,kj,k->lij', 
                            work.didev_dq, work.odev, self.weights)
        dPio_dq += np.einsum('ki,klj,k->lij', 
                             work.idev, work.dodev_dq, self.weights)
        return dPio_dq


class SVDUnscentedTransform(UnscentedTransformBase):
    """Unscented transform using singular value decomposition."""
    
    def sqrt(self, work, Q):
        """Unscented transform square root method."""
        [U, s, VT] = scipy.linalg.svd(Q)
        work.S = np.transpose(U * np.sqrt(s))
        return work.S


class DifferentiableCholesky:
    
    Work = attrdict.AttrDict
    
    @staticmethod
    def initialize_gradient_data(work, ni):
        if getattr(work, 'gradient_data_initialized', 0) == ni:
            return
        
        work.ni_range = np.arange(ni)
        work.ni_tril_indices = np.tril_indices(ni)
        work.gradient_data_initialized = ni
 
    @staticmethod
    def decompose(work, Q):
        """Perform the Cholesky decomposition."""
        work.S = scipy.linalg.cholesky(Q, lower=False)
        return work.S
    
    @classmethod
    def diff(cls, work, dQ_dq):
        """Derivatives of Cholesky decomposition.
        
        Parameters
        ----------
        work :
            Work data.
        dQ_dq : (nq, ni, ni) array_like
            The derivatives of `Q` with respect to some parameter vector.
             Must be symmetric with respect to the last two axes, i.e., 
            `dQ_dq[..., i, j] == dQ_dq[..., j, i]` for all `i, j` pairs.
        
        Returns
        -------
        dS_dq : (nq, ni, ni) array_like
            The derivative of the Cholesky decomposition of `Q` with respect
            to the parameter vector.
        
        """
        nq = len(dQ_dq)
        ni = len(work.S)
        cls.initialize_gradient_data(work, ni)
        
        k = work.ni_range
        i, j = work.ni_tril_indices
        ix, jx, kx = np.ix_(i, j, k)
        
        A = np.zeros((ni, ni, ni, ni))
        A[ix, jx, ix, kx] = work.S[kx, jx]
        A[ix, jx, jx, kx] += work.S[kx, ix]
        A_tril = A[i, j][..., i, j]
        A_tril_inv = scipy.linalg.inv(A_tril)
        
        dQ_dq_tril = dQ_dq[..., i, j]        
        dS_dq_tril = np.einsum('ab,...b->...a', A_tril_inv, dQ_dq_tril)
        dS_dq = np.zeros((nq, ni, ni))
        dS_dq[..., j, i] = dS_dq_tril
        return dS_dq


class CholeskyUnscentedTransform(UnscentedTransformBase):
    """Unscented transform using Cholesky decomposition."""
    
    def sqrt(self, work, Q):
        """Unscented transform square root method."""
        return DifferentiableCholesky.decompose(work, Q)
    
    def sqrt_diff(self, work, dQ_dq):
        """Derivatives of Unscented transform Cholesky decomposition."""
        return DifferentiableCholesky.diff(work, dQ_dq)


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
    
    def __init__(self, model, **options):
        # Initialize base
        super().__init__(model, **options)
        
        # Get transform options
        ut_options = options.copy()
        ut_options.update(utils.extract_subkeys(options, 'pred_ut_'))
        
        # Create the transform object
        UTClass = choose_ut_transform_class(ut_options)
        self.__ut = UTClass(model.nx, **ut_options)
    
    def predict(self, work):
        """Predict the state distribution at the next time index."""
        def f_fun(x):
            return self.model.f(work.k, x)
        
        work.pred_ut = self.__ut.Work(work.x, work.Px)
        f, Pf = self.__ut.transform(work.pred_ut, f_fun)
        Q = self.model.Q(work.k, work.x)
        
        work.prev_x = work.x
        work.prev_Px = work.Px
        work.k += 1
        work.x = f
        work.Px = Pf + Q
        return work.x, work.Px
    
    def prediction_diff(self, work):
        """Calculate the derivatives of the prediction."""
        k = work.k - 1 
        x = work.prev_x
        
        def df_dq_fun(x):
            return self.model.df_dq(k, x)
        def df_dx_fun(x):
            return self.model.df_dx(k, x)
        Df_Dq, DPf_Dq = self.__ut.transform_diff(
            work.pred_ut, df_dq_fun, df_dx_fun, work.dx_dq, work.dPx_dq
        )
        dQ_dq = self.model.dQ_dq(k, x)
        dQ_dx = self.model.dQ_dx(k, x)
        DQ_Dq = dQ_dq + np.einsum('ij,jkl', work.dx_dq, dQ_dx)
        
        work.prev_dx_dq = work.dx_dq.copy()
        work.prev_dPx_dq = work.dPx_dq.copy()
        work.dx_dq += Df_Dq
        work.dPx_dq += DPf_Dq + DQ_Dq
    
    def _calculate_prediction_hess(self, dQ_dq, dQ_dx, ut_work):
        k = self.k
        x = self.x
        dx_dq = self.dx_dq
        dPx_dq = self.dPx_dq
        d2x_dq2 = self.d2x_dq2
        d2Px_dq2 = self.d2Px_dq2
        
        d2Q_dq2 = self.model.d2Q_dq2(k, x)
        d2Q_dq_dx = self.model.d2Q_dq_dx(k, x)
        D2Q_Dq2 = d2Q_dq2 + np.einsum('...aijk,...bi', dQ_dq_dx, dx_dq)
        D2Q_Dq2 += np.einsum('...aij,...jkl', d2x_dq2, dQ_dx)
        D2Q_Dq2 += np.einsum('...ij,...bjkl,...ab', dx_dq, d2Q_dx2, dx_dq)
        D2Q_Dq2 += np.einsum('...ij,...ajkl', dx_dq, d2Q_dq_dx)

        def Df_Dq_fun(x, dx_dq):
            df_dq = self.model.df_dq(k, x)
            df_dx = self.model.df_dx(k, x)
            d2f_dq2 = self.model.d2f_dq2(k, x)
            d2f_dq_dx = self.model.d2f_dq_dx(k, x)
            d2f_dq2  = self.model.d2f_dq2(k, x)
            Df_Dq = d2f_dq2 + np.einsum('...akc,...bk', d2f_dq_dx, dx_dq)
            Df_Dq += np.einsum('...abi,...ij', d2x_dq2, df_dx)
            Df_Dq += np.einsum('...ai,...bij', dx_dq, d2f_dq_dx)
            Df_Dq += np.einsum('...ai,...ijk,...bj', dx_dq, d2f_dx2, dx_dq)
            return Df_Dq
        Df_Dq, DPf_Dq = self.__ut.transform_diff(Df_Dq_fun, dx_dq, dPx_dq, work)


class DTUnscentedCorrector(DTKalmanFilterBase):
    
    def __init__(self, model, **options):
        # Initialize base
        super().__init__(model, **options)
        
        # Get transform options
        ut_options = options.copy()
        ut_options.update(utils.extract_subkeys(options, 'corr_ut_'))
        
        # Create the transform object
        UTClass = choose_ut_transform_class(ut_options)
        self.__ut = UTClass(model.nx, **ut_options)
    
    def correct(self, work, y):
        """Correct the state distribution, given the measurement vector."""
        # Get the y-mask
        mask = ma.getmaskarray(y)
        work.active = active = ~mask
        if np.all(mask):
            return work.x, work.Px
        
        # Remove inactive outputs
        y = ma.compressed(y)
        R = self.model.R()[np.ix_(active, active)]
        def h_fun(x):
            return self.model.h(work.k, x)[..., active]
        
        # Perform unscented transform
        work.corr_ut = self.__ut.Work(work.x, work.Px)
        h, Ph = self.__ut.transform(work.corr_ut, h_fun)
        Pxh = self.__ut.crosscov(work.corr_ut)
        
        # Factorize covariance
        work.py_chol = DifferentiableCholesky.Work()
        Py = Ph + R
        PyC = DifferentiableCholesky.decompose(work.py_chol, Py)
        PyCI = scipy.linalg.inv(PyC)
        PyI = np.einsum('ik,jk', PyCI, PyCI)
        
        # Perform correction
        e = y - h
        K = np.einsum('ik,kj', Pxh, PyI)
        x_corr = work.x + np.einsum('ij,j', K, e)
        Px_corr = work.Px - np.einsum('ik,jl,kl', K, K, Py)
        
        # Save and return the correction data
        work.prev_x = work.x
        work.prev_Px = work.Px
        work.e = e
        work.x = x_corr
        work.Px = Px_corr
        work.Pxh = Pxh
        work.Py = Py
        work.PyI = PyI
        work.PyC = PyC
        work.PyCI = PyCI
        work.K = K
        return x_corr, Px_corr

    def correction_diff(self, work):
        """Calculate the derivatives of the correction."""
        if not np.any(work.active):
            return
        
        # Get the model and transform derivatives
        def dh_dq_fun(x):
            return self.model.dh_dq(work.k, x)[..., work.active]
        def dh_dx_fun(x):
            return self.model.dh_dx(work.k, x)[..., work.active]
        Dh_Dq, DPh_Dq = self.__ut.transform_diff(
            work.corr_ut, dh_dq_fun, dh_dx_fun, work.dx_dq, work.dPx_dq
        )
        dPxh_dq = self.__ut.crosscov_diff(work.corr_ut)
        dR_dq = self.model.dR_dq()[(...,) + np.ix_(work.active, work.active)]

        # Calculate the correction derivatives
        de_dq = -Dh_Dq
        dPy_dq = DPh_Dq + dR_dq
        dPyI_dq = -np.einsum('ij,ajk,kl', work.PyI, dPy_dq, work.PyI)
        dK_dq = np.einsum('ik,akj', work.Pxh, dPyI_dq)
        dK_dq += np.einsum('aik,kj', dPxh_dq, work.PyI)

        work.de_dq = de_dq
        work.dPy_dq = dPy_dq
        work.dPyI_dq = dPyI_dq
        work.prev_dx_dq = work.dx_dq.copy()
        work.prev_dPx_dq = work.dPx_dq.copy()
        work.dx_dq += np.einsum('...aij,...j', dK_dq, work.e)
        work.dx_dq += np.einsum('...ij,...aj', work.K, de_dq)
        work.dPx_dq -= np.einsum('...aik,...jl,...lk', dK_dq, work.K, work.Py)
        work.dPx_dq -= np.einsum('...ik,...ajl,...lk', work.K, dK_dq, work.Py)
        work.dPx_dq -= np.einsum('...ik,...jl,...alk', work.K, work.K, dPy_dq)
    
    def update_likelihood(self, work):
        """Update measurement log-likelihood."""
        if not np.any(work.active):
            return
    
        work.PyCD = np.einsum('...kk->...k', work.PyC)
        work.L -= 0.5 * np.einsum('...i,...ij,...j', work.e, work.PyI, work.e) 
        work.L -= np.log(work.PyCD).sum(-1)

    def likelihood_diff(self, work):
        """Calculate measurement log-likelihood derivatives."""
        if not np.any(work.active):
            return

        # Get the work variables
        e = work.e
        PyI = work.PyI
        de_dq = work.de_dq
        dPyI_dq = work.dPyI_dq
        
        # Calculate the likelihood derivatives
        dPyC_dq = DifferentiableCholesky.diff(work.py_chol, work.dPy_dq)
        dPyCD_dq = np.einsum('...kk->...k', dPyC_dq)
        work.dL_dq -= np.sum(dPyCD_dq / work.PyCD, axis=-1)
        work.dL_dq -= 0.5 * np.einsum('...ai,...ij,...j', de_dq, PyI, e)
        work.dL_dq -= 0.5 * np.einsum('...i,...aij,...j', e, dPyI_dq, e)
        work.dL_dq -= 0.5 * np.einsum('...i,...ij,...aj', e, PyI, de_dq)
    
    def _calculate_correction_grad(self, active, e, K, Pxh, Py, PyI, PyC):
        k = self.k
        x = self.x
        dx_dq = self.dx_dq
        dPx_dq = self.dPx_dq
        dR_dq = self.model.dR_dq()[(...,) + np.ix_(active, active)]
        
        def Dh_Dq_fun(x, dx_dq):
            dh_dq = self.model.dh_dq(k, x)[..., active]
            dh_dx = self.model.dh_dx(k, x)[..., active]
            return dh_dq + np.einsum('...qx,...xh->...qh', dx_dq, dh_dx)
        ut_grads = self.__ut.transform_diff(Dh_Dq_fun, dx_dq, dPx_dq, True)
        Dh_Dq, dPh_dq, dPxh_dq = ut_grads
        
        de_dq = -Dh_Dq
        dPy_dq = dPh_dq + dR_dq
        dPyI_dq = -np.einsum('...ij,...ajk,...kl', PyI, dPy_dq, PyI)
        dK_dq = np.einsum('...ik,...akj', Pxh, dPyI_dq)
        dK_dq += np.einsum('...aik,...kj', dPxh_dq, PyI)
        
        self.dx_dq += np.einsum('...aij,...j', dK_dq, e)
        self.dx_dq += np.einsum('...ij,...aj', K, de_dq)
        self.dPx_dq -= np.einsum('...aik,...jl,...lk', dK_dq, K, Py)
        self.dPx_dq -= np.einsum('...ik,...ajl,...lk', K, dK_dq, Py)
        self.dPx_dq -= np.einsum('...ik,...jl,...alk', K, K, dPy_dq)

        dPyC_dq = cholesky_sqrt_diff(PyC, dPy_dq)
        diag_PyC = np.einsum('...kk->...k', PyC)
        diag_dPyC_dq = np.einsum('...kk->...k', dPyC_dq)
        self.dL_dq -= np.sum(diag_dPyC_dq / diag_PyC, axis=-1)
        self.dL_dq -= 0.5 * np.einsum('...ai,...ij,...j', de_dq, PyI, e)
        self.dL_dq -= 0.5 * np.einsum('...i,...aij,...j', e, dPyI_dq, e)
        self.dL_dq -= 0.5 * np.einsum('...i,...ij,...aj', e, PyI, de_dq)


class DTUnscentedKalmanFilter(DTUnscentedPredictor, DTUnscentedCorrector):
    pass

