#!/usr/bin/env python

"""Brachistochrone optimal control example."""


import importlib

import numpy as np
import sympy
import sym2num.model

from ceacoest.modelling import symoc, symcol, symoptim


# Reload modules for testing
for m in (symoptim, symcol, symoc):
    importlib.reload(m)


class BrachistochroneModel(symoc.OCModel):
    """Symbolic Brachistochrone optimal control model."""
    
    collocation_order = 3

    def __init__(self):
        v = self.Variables(
            x=['x', 'y', 'v'],
            u=['theta']
        )
        super().__init__(v)

    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, a):
        g = 9.80665
        xdot = [a.v * sympy.sin(a.theta),
                a.v * sympy.cos(a.theta),
                g * sympy.cos(a.theta)]
        return xdot


if __name__ == '__main__':
    model = BrachistochroneModel()
    
