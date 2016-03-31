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


class DTPredictor(base.DTFilter):
        
    def predict(self):
        """Predict the state distribution at the next time index."""
        f = self.model.f(self.k, self.x)
        df_dx = self.model.df_dx(self.k, self.x)
        Pxf = np.einsum('...yi,...iz', self.Px, df_dx)
        Pf = np.einsum('...iy,...iz', df_dx, Pxf)
        Q = self.model.Q(self.k, self.x)

        # Save internal variables
        self.Pxf = Pxf

        # Update mean, cov and time index
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
        Px = self.prev_Px
        dx_dq = self.dx_dq
        dPx_dq = self.dPx_dq

        # Load saved internal variables
        Pxf = self.Pxf
        
        df_dq = self.df_dq = self.model.df_dq(k, x)
        d2f_dx_dq = self.d2f_dx_dq = self.model.d2f_dx_dq(k, x)
        d2f_dx2 = self.d2f_dx2 = self.model.d2f_dx2(k, x)
        dQ_dq = self.dQ_dq = self.model.dQ_dq(k, x)
        dQ_dx = self.dQ_dx = self.model.dQ_dx(k, x)
        
        Df_Dq = df_dq + np.einsum('...iz,...yi', df_dx, dx_dq)
        dPxf_dq = np.einsum('...xyi,...iz', dPx_dq, df_dx)
        dPxf_dq += np.einsum('...yi,...xiz', Px, d2f_dx_dq)
        dPxf_dq += np.einsum('...yi,...xiz,...wx', Px, d2f_dx2, dx_dq)
        dPf_dq = np.einsum('...iy,...xiz', df_dx, dPxf_dq)
        dPf_dq += np.einsum('...xiy,...iz', df_dx_dq, Pxf)
        dPf_dq += np.einsum('...wx,...xiy,...iz', dx_dq, d2f_dx2, Pxf)        
        DQ_Dq = dQ_dq + np.einsum('...ij,...jkl', self.dx_dq, dQ_dx)
        
        # Update derivatives
        self.prev_dx_dq = self.dx_dq
        self.prev_dPx_dq = self.dPx_dq        
        self.dx_dq = Df_Dq
        self.dPx_dq = dPf_dq + DQ_Dq
    
    def prediction_diff2(self):
        """Calculate the second derivatives of the prediction."""
        k = self.k - 1 
        x = self.prev_x
        Px = self.prev_Px
        dx_dq = self.prev_dx_dq
        dPx_dq = self.prev_dPx_dq
        d2x_dq2 = self.d2x_dq2
        d2Px_dq2 = self.d2Px_dq2

        # Load saved internal variables
        Pxf = self.Pxf
        d2f_dx_dq = self.d2f_dx_dq
        
        d3f_dx_dq2 = self.model.d3f_dx_dq2(k, x)
        d3f_dx2_dq = self.model.d3f_dx2_dq(k, x)
        
        D2f_Dq2 = d2f_dq2 + np.einsum('...iz,...xyi', df_dx, d2x_dq2)
        D2f_Dq2 += np.einsum('...xiz,...yi', d2f_dx_dq, dx_dq)
        D2f_Dq2 += np.einsum('...xj,...jiz,...yi', dx_dq, d2f_dx2, dx_dq)
        d2Pxf_dq2 = np.einsum('...xyi,...wiz', dPx_dq, d2f_dx_dq)
        d2Pxf_dq2 += np.einsum('...xyi,...jiz,...wj', dPx_dq, d2f_dx2, dx_dq)
        d2Pxf_dq2 += np.einsum('...wxyi,...iz', d2Px_dq2, df_dx)
        d2Pxf_dq2 += np.einsum('...wyi,...xiz', dPx_dq, d2f_dx_dq)
        d2Pxf_dq2 += np.einsum('...yi,...wxiz', Px, d3f_dx_dq2)
        d2Pxf_dq2 += np.einsum('...yi,...xjiz,...wj', Px, d3f_dx2_dq, dx_dq)

        # TODO
        
        #dPxf_dq += np.einsum('...yi,...xiz,...wx', Px, d2f_dx2, dx_dq)
        #dPf_dq = np.einsum('...iy,...xiz', df_dx, dPxf_dq)
        #dPf_dq += np.einsum('...xiy,...iz', df_dx_dq, Pxf)
        #dPf_dq += np.einsum('...wx,...xiy,...iz', dx_dq, d2f_dx2, Pxf)        
        #DQ_Dq = dQ_dq + np.einsum('...ij,...jkl', self.dx_dq, dQ_dx)

        self.d2x_dq2 = D2f_Dq2
        self.d2Px_dq2 = D2Pf_Dq2 + D2Q_Dq2

    def prediction_crosscov(self):
        return self.Pxf


class DTCorrector(base.DTFilter):
    
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

        # Evaluate the model functions
        h = self.model.h(self.k, self.x)[..., active]
        dh_dx = self.model.dh_dx(self.k, self.x)[..., active]

        # Calculate the covariances and gain
        Pxh = np.einsum('...ij,...jz', self.Px, dh_dx)
        Ph = np.einsum('...iy,...iz', dh_dx, Pxh)
        Py = Ph + R
        PyI = np.linalg.inv(Py)
        K = np.einsum('...ik,...kj', Pxh, PyI)
        
        # Perform correction
        e = y - h
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
        self.K = K
        return x_corr, Px_corr
    
    def correction_diff(self):
        """Calculate the derivatives of the correction."""
        if not np.any(self.active):
            return
    
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
        self.dL_dq -= np.einsum('...ai,...ij,...j', de_dq, PyI, e)
        self.dL_dq -= 0.5 * np.einsum('...i,...aij,...j', e, dPyI_dq, e)
    
    def likelihood_diff2(self):
        """Calculate measurement log-likelihood derivatives."""
        if not np.any(self.active):
            return
        
        # Get the work variables
        e = self.e
        PyI = self.PyI
        de_dq = self.de_dq
        dPyI_dq = self.dPyI_dq
        d2e_dq2 = self.d2e_dq2
        d2PyI_dq2 = self.d2PyI_dq2
        
        # Calculate the likelihood derivatives


class DTFilter(DTPredictor, DTCorrector):
    pass

