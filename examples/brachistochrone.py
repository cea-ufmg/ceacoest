#!/usr/bin/env python

"""Brachistochrone optimal control example."""


import importlib

import numpy as np
import sympy
import sym2num.model

from ceacoest.modelling import genoptim, symoc, symcol, symoptim


# Reload modules for testing
for m in (genoptim, symoptim, symcol, symoc):
    importlib.reload(m)


class SymbolicBrachistochroneModel(symoc.OCModel):
    """Symbolic Brachistochrone optimal control model."""
    
    collocation_order = 3

    def __init__(self):
        v = self.Variables(
            x=['x', 'y', 'v'],
            u=['theta'],
            p=['tf']
        )
        super().__init__(v)

    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""        
        g = 9.80665
        xdot = [s.v * sympy.sin(s.theta) * s.tf,
                s.v * sympy.cos(s.theta) * s.tf,
                g * sympy.cos(s.theta) * s.tf]
        return xdot
    
    @sym2num.model.collect_symbols
    def M(self, xe, p, *, s):
        """Mayer (endpoint) cost."""
        return s.tf


if __name__ == '__main__':
    symmodel = SymbolicBrachistochroneModel()
    GeneratedBrachistochrone = symmodel.compile_class()    
    model = GeneratedBrachistochrone()
