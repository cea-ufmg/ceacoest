'''Kalman filtering / smoothing module.

TODO
----
 * Add derivative of SVD square root.
 * Vectorize cholesky_sqrt_diff, sigma_points_diff and transform_diff.
 * Make docstrings for all constructors.
 * Implement filter Hessian.

Improvement ideas
-----------------
 * Allow gradients and Hessian to be calculated offline, saving processing time
   at the cost of memory.

'''


import abc
import collections
import re

import numpy as np
import numpy.ma as ma
import numpy.linalg
import scipy.linalg

from . import utils


class DTKalmanFilterBase(metaclass=abc.ABCMeta):
    '''Discrete-time Kalman filter/smoother abstract base class.'''
    
    def __init__(self, model, x, Px, **options):
        '''Create a discrete-time Kalman filter.
        
        Required parameters
        -------------------
        model :
            The underlying system model.
        x : (..., nx) array_like
            The initial state mean
        Px : (..., nx, nx) array_like
            The initial state covariance.
        
        Options
        -------
        pem : logical or 'save' or 'grad' or 'hess'
            Whether and how the prediction error method is being used.
            If 'save' the internal filter variables are saved for gradient
            and Hessian calculation. If 'grad' or 'hess' then the filter
            gradients or Hessian are calculated online. False by default.
        save_history : int or 'filter'
            Wether to save the filter history or not. If it is an int then
            it specifies the history size. Otherwise the size is defined by
            the filter function. Equal to 'filter' by default.
        save_pred_crosscov :
            Whether to save the prediction cross-covariance matrices.
            False by default.
        k0 : float
            Initial sample index time, zero by default.
        
        '''
        # Save and initialize basic filter data
        self.model = model
        '''The underlying system model.'''
        
        self.x = np.asarray(x, dtype=float)
        '''The working filtered state mean.'''
        
        self.Px = np.asarray(Px, float)
        '''The working filtered state covariance.'''

        nx = model.nx
        nq = getattr(model, 'nq', 0)
        base_shape = self.x.shape[:-1]
        
        self.base_shape = base_shape
        '''Shape of scalar element for filter vectorization.'''
                
        self.pem = options.get('pem', False)
        '''Whether and how the prediction error method is being used.'''
        
        self.save_history = options.get('save_history', 'filter')
        '''Whether to save the filter history.'''
        
        self.save_pred_crosscov = options.get('save_pred_crosscov', False)
        '''Whether to save the prediction cross-covariance.'''

        self.k = options.get('k0', 0)
        '''The working filter sample index.'''
        
        self.history_size = 0
        '''The filter history size.'''

        self.L = 0
        '''The log-likelihood of the measurements.'''
        
        self.dL_dq = np.zeros(base_shape + (nq,)) if self.pem else None
        '''The log-likelihood gradient.'''
        
        self.dx_dq = np.zeros(base_shape + (nq, nx)) if self.pem else None
        '''The working mean gradient.'''
        
        self.dPx_dq = np.zeros(base_shape + (nq, nx, nx)) if self.pem else None
        '''The working covariance gradient.'''
        
        # Check argument shapes
        assert self.x.shape[-1:] == (model.nx,)
        assert self.Px.shape == self.base_shape + (model.nx, model.nx)
        
        # Initialize the filter history
        if self.save_history != 'filter':
            try:
                size = int(self.save_history)
                self.initialize_history(size)
            except (ValueError, TypeError):
                raise TypeError("save_history must be 'filter' or int-like.")
    
    def initialize_history(self, size):
        # Allocate the history arrays, if needed
        if size != self.history_size:
            nx = self.model.nx
            nq = self.model.nq
            base_shape = self.base_shape
            mean_shape = (size,) + base_shape + (nx,)
            cov_shape = (size,) + base_shape + (nx, nx)
            self.x_pred = np.zeros(mean_shape)
            self.x_corr = np.zeros(mean_shape)
            self.Px_pred = np.zeros(cov_shape)
            self.Px_corr = np.zeros(cov_shape)
            if self.save_pred_crosscov:
                self.Pxf = np.zeros((size - 1,) + base_shape + (nx, nx))
        
        # Initialize the history variables
        self.x_pred[0] = self.x
        self.x_corr[0] = self.x
        self.Px_pred[0] = self.Px
        self.Px_corr[0] = self.Px
        self.history_size = size
    
    def _save_prediction(self, x, Px):
        '''Save the prection data and increment the time and history index.'''
        self.x = x
        self.Px = Px
        self.k += 1
        k = self.k
        if k < self.history_size:
            self.x_pred[k] = x
            self.x_corr[k] = x
            self.Px_pred[k] = Px
            self.Px_corr[k] = Px
    
    def _save_correction(self, x, Px):
        '''Save the correction data.'''
        self.x = x
        self.Px = Px
        k = self.k
        if k < self.history_size:
            self.x_corr[k] = x
            self.Px_corr[k] = Px
    
    def _save_prediction_crosscov(self, Pxf):
        '''Save the prediction cross-covariance.'''
        if self.k < self.history_size:
            self.Pxf[self.k] = Pxf
    
    @abc.abstractmethod
    def predict(self):
        '''Predict the state distribution at a next time sample.'''
        raise NotImplementedError("Pure abstract method.")
    
    @abc.abstractmethod
    def correct(self, y):
        '''Correct the state distribution, given the measurement vector.'''
        raise NotImplementedError("Pure abstract method.")
    
    def filter(self, y):
        y = np.asanyarray(y)
        N = len(y)
        
        if self.save_history == 'filter':
            self.initialize_history(N)
        
        self.correct(y[0])
        for k in range(1, N):
            self.predict()
            self.correct(y[k])
        
    
def svd_sqrt(mat):
    '''SVD-based "square root" of a symmetric positive-semidefinite matrix.
    
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


def ldl_sqrt(mat):
    '''LDL-based "square root" of a symmetric positive-semidefinite matrix.
    
    Used for unscented transform.
    
    Example
    -------
    Generate a random positive-semidefinite symmetric matrix.
    >>> np.random.seed(0)
    >>> A = np.random.randn(4, 10)
    >>> Q = np.dot(A, A.T)
    
    The square root should satisfy S'S = Q
    >>> S = ldl_sqrt(Q)
    >>> STS = np.dot(S.T, S)
    >>> np.testing.assert_allclose(Q, STS)
    
    '''
    L, D = numpy.linalg.ldl(mat)
    sqrt_D = np.sqrt(np.einsum('...ii->...i', D))    
    return np.einsum('...ij,...j->...ji', L, sqrt_D)


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
    A_tril_inv = scipy.linalg.pinv(A_tril)
    
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


ldl_sqrt.diff = cholesky_sqrt_diff

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
            'svd', 'ldl' or 'cholesky' then `svd_sqrt`, `ldl_sqrt` or 
            `cholesky_sqrt` are used, respectively. Otherwise, if it is a
            callable, the object is used. Equal to 'cholesky' by default.
        
        '''
        self.nin = nin
        '''Number of inputs.'''
        
        self.kappa = options.get('kappa', 0.0)
        '''Weight of the center sigma point.'''
        assert self.nin + self.kappa != 0
        
        sqrt_opt = options.get('sqrt', 'cholesky')
        if sqrt_opt == 'cholesky':
            sqrt = cholesky_sqrt
        elif sqrt_opt == 'ldl':
            sqrt = ldl_sqrt
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
        in_dev_diff = np.zeros((self.nsigma,) + mean_diff.shape)
        in_dev_diff[:nin] = np.rollaxis(cov_sqrt_diff, -2)
        in_dev_diff[nin:(2 * nin)] = -in_dev_diff[:nin]
        in_sigma_diff = in_dev_diff + mean_diff
        self.in_dev_diff = in_dev_diff
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
            crosscov_diff = np.einsum('k...qi,k...j,k->...qij',
                                      self.in_dev_diff, out_dev, weights)
            crosscov_diff += np.einsum('k...i,k...qj,k->...qij',
                                       in_dev, out_dev_diff, weights)
            return (out_mean_diff, out_cov_diff, crosscov_diff)
        else:
            return (out_mean_diff, out_cov_diff)


class DTUnscentedPredictor(DTKalmanFilterBase):
    
    def __init__(self, model, mean, cov, **options):
        super().__init__(model, mean, cov, **options)
        ut_options = options.copy()
        ut_options.update(utils.extract_subkeys(options, 'pred_ut_'))
        self.__ut = UnscentedTransform(model.nx, **ut_options)
    
    def predict(self):
        '''Predict the state distribution at the next time index.'''
        k = self.k
        def f_fun(x):
            return self.model.f(k, x)
        f, Pf = self.__ut.transform(f_fun, self.x, self.Px)
        Q = self.model.Q(k, self.x)
        Px = Pf + Q
        
        # Save the prediction cross-covariance for smoothing
        if self.save_pred_crosscov:
            Pxf = self.__ut.crosscov()
            self._save_prediction_crosscov(Pxf)
        
        # Calculate the precition gradient for the PEM
        if self.pem == 'grad' or self.pem == 'hess':
            self._calculate_prediction_grad()
        
        # Save and update the time and indices
        self._save_prediction(f, Px)
    
    def _calculate_prediction_grad(self):
        k = self.k
        x = self.x
        dx_dq = self.dx_dq
        dPx_dq = self.dPx_dq
        
        dQ_dq = self.model.dQ_dq(k, x)
        dQ_dx = self.model.dQ_dx(k, x)
        DQ_Dq = dQ_dq + np.einsum('...ij,...jkl', dx_dq, dQ_dx)
        
        def Df_Dq_fun(x, dx_dq):
            df_dq = self.model.df_dq(k, x)
            df_dx = self.model.df_dx(k, x)
            return df_dq + np.einsum('...qx,...xf->...qf', dx_dq, df_dx)
        Df_Dq, DPf_Dq = self.__ut.transform_diff(Df_Dq_fun, dx_dq, dPx_dq)
        
        self.dx_dq = Df_Dq
        self.dPx_dq = DPf_Dq + DQ_Dq


class DTUnscentedCorrector(DTKalmanFilterBase):

    def __init__(self, model, mean, cov, **options):
        super().__init__(model, mean, cov, **options)
        ut_options = options.copy()
        ut_options.update(utils.extract_subkeys(options, 'pred_ut_'))
        self.__ut = UnscentedTransform(model.nx, **ut_options)
    
    def initialize_history(self, size):
        size_changed = size != self.history_size
        super().initialize_history(size)

        # The extra variables are only needed for the PEM.
        if self.pem != 'save':
            return
        
        # Allocate the history arrays, if needed        
        if size_changed:
            nx = self.model.nx
            ny = self.model.ny
            nq = self.model.nq
            base_shape = self.base_shape
            self.y_active = np.zeros((size, ny), dtype=bool)
            self.e = np.zeros((size,) + base_shape + (ny,))
            self.K = np.zeros((size,) + base_shape + (nx, ny))
            self.Pxh = np.zeros((size,) + base_shape + (nx, ny))
            self.Py = np.zeros((size,) + base_shape + (ny, ny))
            self.PyI = np.zeros((size,) + base_shape + (ny, ny))
            self.PyC = np.zeros((size,) + base_shape + (ny, ny))

    def _save_correction_pem(self, active, e, K, Pxh, Py, PyI, PyC):
        k = self.k
        cov_ind = (k, ...) + np.ix_(active, active)
        self.y_active[k] = active
        self.e[k, ..., active] = e
        self.K[k, ..., active] = K
        self.Pxh[k, ..., active] = Pxh
        self.Py[cov_ind] = Py
        self.PyI[cov_ind] = PyI
        self.PyC[cov_ind] = PyC
    
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
        def h_fun(x):
            return self.model.h(self.k, x)[..., active]
        
        # Perform unscented transform
        h, Ph = self.__ut.transform(h_fun, self.x, self.Px)
        Pxh = self.__ut.crosscov()
        
        # Factorize covariance
        Py = Ph + R
        PyC = numpy.linalg.cholesky(Py)
        PyCI = numpy.linalg.inv(PyC)
        PyI = np.einsum('...ki,...kj', PyCI, PyCI)
        
        # Perform correction
        e = y - h
        K = np.einsum('...ik,...kj', Pxh, PyI)
        x_corr = self.x + np.einsum('...ij,...j', K, e)
        Px_corr = self.Px - np.einsum('...ik,...jl,...lk', K, K, Py)
    
        # Update log-likelihood and save PEM data
        if self.pem:
            PyCD = np.einsum('...kk->...k', PyC)
            self.L -= 0.5 * np.einsum('...i,...ij,...j', e, PyI, e) 
            self.L -= np.log(PyCD).sum(-1)
        if self.pem == 'save':
            self._save_correction_pem(active, e, K, Pxh, Py, PyI, PyC)
        elif self.pem == 'grad' or self.pem == 'hess':
            self._calculate_correction_grad(active, e, K, Pxh, Py, PyI, PyC)

        # Save the correction data
        self._save_correction(x_corr, Px_corr)
    
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

