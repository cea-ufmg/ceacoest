"""Runge--Kutta integration and collocation."""


import collections.abc

import numpy as np
import scipy
from numpy import linalg, polynomial


def lgl_points(n):
    """Legendre Gauss Lobatto collocation points.
    
    Parameters
    ----------
    n : int
        The collocation order, which equals the number of points.
    
    Returns
    -------
    points : (n,) array
        The collocation points, beginning at 0 and ending at 1.

    >>> lgl_points(2)
    array([ 0.,  1.])
    >>> lgl_points(3)
    array([ 0. ,  0.5,  1. ])
    
    """
    if n < 2:
        raise ValueError("Collocation order of Lobatto methods must be > 1.")
    
    ln = polynomial.Legendre([0]*(n - 1) + [1], [0, 1])
    return np.r_[0.0, ln.deriv().roots(), 1.0]
  

def lagrange_basis(points):
    """Lagrange basis polynomials for interpolation at given points."""
    l = np.empty(len(points), object)
    for i, point in enumerate(points):
        roots = np.r_[points[:i], points[i+1:]]
        l[i] = polynomial.Polynomial.fromroots(roots)/np.prod(point - roots)
     
    return l


def pdinteg(p, limits, m=1):
    """Polynomial definite integral."""
    if isinstance(p, polynomial.Polynomial):
        return p.integ(m=m,lbnd=limits[0])(limits[-1])
    elif isinstance(p, collections.abc.Iterable):
        return np.array([pdinteg(pi, limits, m) for pi in p])
    else:
        raise TypeError("Unrecognized type for argument `p`.")


class LGLCollocation:
    """Legendre--Gauss--Lobatto integral collocation."""
    
    def __init__(self, n):
        points = lgl_points(n)
        l = lagrange_basis(points)
        interv_integ = [pdinteg(l, points[i:i+2]) for i in range(n - 1)]
        
        self.n = n
        """Order of the collocation (number of collocation points)."""
        
        self.points = points
        """Collocation points for the [0, 1] interval."""
        
        self.ninterv = n - 1
        """Number of collocation intervals."""

        self.J = np.array(interv_integ)
        """Coefficients of the interpolant's integral across each interval."""
         
        self.K = np.sum(interv_integ, axis=0)
        """Coefficients of the quadrature across the whole piece."""        

        self.JP = np.linalg.pinv(self.J)
        """Moore--Penrose pseudoinverse of the J matrix."""

        self.JT_range = scipy.linalg.orth(self.J.T)
        """Orthogonal basis for the range of the J.T matrix."""
    
    def grid(self, t_piece):
        """Construct a collocation grid (fine) from a piece grid (coarse)."""
        t_piece = np.asarray(t_piece)
        ti, tf = t_piece[[0, -1]]
        piece_len = np.diff(t_piece)
        increments = piece_len[:, None] * self.points[:-1]
        return np.r_[np.ravel(t_piece[:-1, None] + increments), tf]

