"""Symbolic statistics utilities."""


import sympy
import sym2num


def normal_logpdf(x, mean, std):
    """Logarithm of normal distribution probability density function."""
    logpdf = -(x - mean) ** 2 / (2 * std ** 2) - sympy.log(std)
    logpdf += -sympy.log(2 * sympy.pi) / 2
    return 1.0 * ~sym2num.getmaskarray(x) * logpdf


def normal_logpdf1(x, mean, std):
    """Normal distribution log density without constant terms."""
    logpdf = -(x - mean) ** 2 / (2 * std ** 2) - sympy.log(std)
    return 1.0 * ~sym2num.getmaskarray(x) * logpdf
