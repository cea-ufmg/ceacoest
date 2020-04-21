"""Auxiliary code for generated SDE models."""


import numpy as np

from . import stats


class ConditionalGaussianTransition:
    def trans_pdf(self, k, x, xnext):
        """Unnormalized state transition probability density function."""
        f = self.f(k, x)
        Q = self.Q(k, x)
        return stats.multivariate_normal_pdf(xnext, f, Q)

    def trans_rvs(self, k, x, w=None):
        """Samples random variates from the transition distribution."""
        f = self.f(k, x)
        g = self.g(k, x)
        
        nw = self.nw
        if w is None:
            w = np.random.randn(*f.shape[:-1], nw)
        else:
            assert np.shape(w) == f.shape[:,-1] + (nw,)
        gw = np.einsum('...ij,...j', g, w)
        return f + gw


class ConditionalGaussianMeasurement:
    def meas_pdf(self, k, x, y):
        """Unnormalized state transition probability density function."""
        h = self.h(k, x)
        R = self.R()
        return stats.multivariate_normal_pdf(y, h, R)


class ConditionalGaussianModel(ConditionalGaussianTransition, 
                               ConditionalGaussianMeasurement):
    pass


