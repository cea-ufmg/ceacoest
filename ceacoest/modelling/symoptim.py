"""Symbolic optimization model for code generation."""

import inspect
import itertools

import numpy as np
import sym2num.model
import sym2num.var
import sympy


class OptimizationModel(sym2num.model.Base):
    """Symbolic optimization model base."""

    def __init__(self):
        # Initialize base class
        super().__init__()
        
        # Initialize model variables
        self.decision = set()
        """Names of decision variables."""
        
        self.constraints = {}
        """Constraint function descriptions."""
        
        self.objectives = {}
        """Objective function descriptions."""

        self.sparse_nzind = {}
        """Nonzero indices of sparse functions"""

        self.sparse_nnz = {}
        """Number of nonzero elements of sparse functions."""

        self.generate_functions = set()
        """Names of functions to generate code."""

    @property
    def generate_assignments(self):
        """Dictionary of assignments in generated class code."""
        a = dict(
            constraints=self.constraints, 
            objectives=self.objectives,
            base_shapes={d: self.variables[d].shape for d in self.decision}
        )
        for k, v in self.sparse_nzind.items():
            a[f'{k}_ind'] = v
        for k, v in self.sparse_nnz.items():
            a[f'{k}_nnz'] = v
        return a
    
    def add_objective(self, fname, derivatives=2):
        der1 = {}
        der2 = {}
        desc = dict(der1=der1, der2=der2)
        self.objectives[fname] = desc
        self.generate_functions.add(fname)
        
        if not derivatives:
            return
        
        # Get variables needed for derivative calculation
        args = self.function_codegen_arguments(fname)
        wrt = set(args).intersection(self.decision)

        # Calculate first derivatives
        if derivatives >= 1:
            for argname in wrt:
                derivname = self.first_derivative_name(fname, argname)
                self.add_derivative(fname, argname, derivname)
                self.generate_functions.add(derivname)
                der1[argname,] = derivname
        
        # Calculate second derivatives
        if derivatives >= 2:
            for pair in itertools.combinations_with_replacement(wrt, 2):
                derivname = self.second_derivative_name(fname, pair)
                self.add_sparse_derivative(fname, pair, derivname)
                der2[pair] = derivname
    
    def add_constraint(self, fname, derivatives=2):
        fshape = self.default_function_output(fname).shape
        
        der1 = {}
        der2 = {}
        desc = dict(shape=fshape, der1=der1, der2=der2)
        self.constraints[fname] = desc
        self.generate_functions.add(fname)
                
        if not derivatives:
            return

        # Get variables needed for derivative calculation
        args = self.function_codegen_arguments(fname)
        wrt = set(args).intersection(self.decision)

        # Calculate first derivatives
        if derivatives >= 1:
            for argname in wrt:
                derivname = self.first_derivative_name(fname, argname)
                self.add_sparse_derivative(fname, argname, derivname)
                der1[argname,] = derivname
        
        # Calculate second derivatives
        if derivatives >= 2:
            for pair in itertools.combinations_with_replacement(wrt, 2):
                derivname = self.second_derivative_name(fname, pair)
                self.add_sparse_derivative(fname, pair, derivname)
                der2[pair] = derivname
    
    def add_sparse_derivative(self, fname, wrt, dname, sel='tril', gen=True):
        if isinstance(wrt, str):
            wrt = (wrt,)
        
        fsize = self.default_function_output(fname).size
        wrt_sizes = tuple(self.variables[name].size for name in reversed(wrt))
        
        self.add_derivative(fname, wrt, dname)
        expr = self.default_function_output(dname)
        expr = np.reshape(expr, wrt_sizes + (fsize,))
        
        # Choose selector
        if len(wrt) == 2 and wrt[0] == wrt[1] and sel == 'tril':
            keepind = lambda ind: ind[0] <= ind[1]
        else:
            keepind = lambda ind: True
        
        # Find nonzero elements
        nzexpr = []
        nzind = []
        for ind, val in np.ndenumerate(expr):
            if keepind(ind) and val != 0:
                nzexpr.append(val)
                nzind.append(ind)
        
        # Convert to ndarray
        nzexpr = np.asarray(nzexpr, dtype=object)
        nzind = np.asarray(nzind, dtype=int).T.reshape(len(wrt) + 1, -1)
        
        # Save indices and number of nonzero elements
        self.sparse_nzind[dname] = nzind
        self.sparse_nnz[dname] = nzexpr.size
        
        # Create symbolic function
        fargs = self.function_codegen_arguments(fname)
        valfun = sym2num.function.SymbolicSubsFunction(fargs, nzexpr)
        valfun_name = f'{dname}_val'
        setattr(self, valfun_name, valfun)
        
        # Include in set of functions to generate code
        if gen:
            self.generate_functions.add(valfun_name)
    
    def first_derivative_name(self, fname, wrtname):
        """Generator of default name of first derivatives."""
        return f'd{fname}_d{wrtname}'
    
    def second_derivative_name(self, fname, wrt):
        """Generator of default name of second derivatives."""
        if not isinstance(wrt, tuple) or len(wrt) != 2:
            raise ValueError("wrt must be a two-element tuple")
        
        if wrt[0] == wrt[1]:
            return f'd2{fname}_d{wrt[0]}2'
        else:
            return f'd2{fname}_d{wrt[0]}_d{wrt[1]}'
    
