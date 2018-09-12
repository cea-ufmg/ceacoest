"""Optimal control."""


import itertools

import numpy as np

from . import optim, rk, utils


class Problem(optim.Problem):
    """Optimal control problem with LGL direct collocation."""
    
    def __init__(self, model, t, **options):
        self.model = model
        """Underlying cost, constraint and dynamical system model."""

        self.collocation = col = rk.LGLCollocation(model.collocation_order)
        """Collocation method."""
        
        self.piece_len = np.diff(t)
        """Normalized length of each collocation piece."""

        assert np.all(self.piece_len > 0)
        self.npieces = npieces = len(self.piece_len)
        """Number of collocation pieces."""
        
        self.tc = self.collocation.grid(t)
        """Normalized collocation time grid."""
        
        super().__init__()
        npoints = self.tc.size #Total number of collocation points
        x = self.register_decision('x', model.nx, npoints)
        u = self.register_decision('u', model.nu, npoints)
        p = self.register_decision('p', model.np)
        self.register_derived('xp', PieceRavelledVariable(self, 'x'))
        self.register_derived('up', PieceRavelledVariable(self, 'u'))
        self.register_derived('xe', XEVariable(self))
        
        self.register_constraint('g', model.g, ('x','u','p'), model.ng, npoints)
        self.register_constraint('h', model.h, ('xe', 'p'), model.nh)
        self.register_constraint(
            'e', model.e, ('xp','up','p', 'piece_len'), model.ne, npieces
        )
        self._register_model_constraint_derivatives('g', ('x', 'u', 'p'))
        self._register_model_constraint_derivatives('h', ('xe', 'p'))
        self._register_model_constraint_derivatives('e', ('xp', 'up', 'p'))
        
        self.register_merit('M', model.M, ('xe', 'p'))
        self.register_merit(
            'IL', model.IL, ('xp', 'up', 'p', 'piece_len'), npieces
        )
        self._register_model_merit_derivatives('M', ('xe', 'p'))
        self._register_model_merit_derivatives('IL', ('xp', 'up', 'p'))
    
    def _register_model_merit_gradient(self, merit_name, wrt_name):
        grad = getattr(self.model, f'd{merit_name}_d{wrt_name}')
        self.register_merit_gradient(merit_name, wrt_name, grad)

    def _register_model_merit_hessian(self, merit_name, wrt_names):
        hess_name = utils.double_deriv_name(merit_name, wrt_names)
        val = getattr(self.model, f'{hess_name}_val')
        ind = getattr(self.model, f'{hess_name}_ind')
        self.register_merit_hessian(merit_name, wrt_names, val, ind)

    def _register_model_merit_derivatives(self, merit_name, wrt_names):
        for wrt_name in wrt_names:
            self._register_model_merit_gradient(merit_name, wrt_name)
        for comb in itertools.combinations_with_replacement(wrt_names, 2):
            self._register_model_merit_hessian(merit_name, comb)
    
    def _register_model_constraint_jacobian(self, constraint_name, wrt_name):
        val = getattr(self.model, f'd{constraint_name}_d{wrt_name}_val')
        ind = getattr(self.model, f'd{constraint_name}_d{wrt_name}_ind')
        self.register_constraint_jacobian(constraint_name, wrt_name, val, ind)

    def _register_model_constraint_hessian(self, constraint_name, wrt_names):
        hess_name = utils.double_deriv_name(constraint_name, wrt_names)
        val = getattr(self.model, f'{hess_name}_val')
        ind = getattr(self.model, f'{hess_name}_ind')
        self.register_constraint_hessian(constraint_name, wrt_names, val, ind)
    
    def _register_model_constraint_derivatives(self, cons_name, wrt_names):
        for wrt_name in wrt_names:
            self._register_model_constraint_jacobian(cons_name, wrt_name)
        for comb in itertools.combinations_with_replacement(wrt_names, 2):
            self._register_model_constraint_hessian(cons_name, comb)
        
    def variables(self, dvec):
        """Get all variables needed to evaluate problem functions."""
        return {'piece_len': self.piece_len, **super().variables(dvec)}


class PieceRavelledVariable:
    def __init__(self, problem, var_name):
        self.p = problem
        self.var_name = var_name
    
    @property
    def var(self):
        return self.p.decision[self.var_name]
    
    @property
    def nvar(self):
        return self.var.shape[1]
    
    @property
    def shape(self):
        return (self.p.npieces, self.p.collocation.n, self.nvar)

    @property
    def tiling(self):
        return self.p.npieces
    
    def build(self, variables):
        v = variables[self.var_name]
        assert v.shape == self.var.shape
        vp = np.zeros(self.shape)
        vp[:, :-1].flat = v[:-1, :].flat
        vp[:-1, -1] = vp[1:, 0]
        vp[-1, -1] = v[-1]
        return vp
    
    def add_to(self, destination, value):
        vp = np.asarray(value)
        assert vp.shape == self.shape
        v = np.zeros(self.var.shape)
        v[:-1].flat = vp[:, :-1].flatten()
        v[self.p.collocation.n-1::self.p.collocation.n-1] += vp[:, -1]
        self.var.add_to(destination, v)
    
    def expand_indices(self, ind):
        npieces = self.p.npieces
        increments = self.p.collocation.ninterv * self.nvar
        return ind + np.arange(npieces)[:, None] * increments + self.var.offset


class XEVariable:
    def __init__(self, problem):
        self.p = problem

    @property
    def shape(self):
        return (2, self.p.model.nx)
    
    def build(self, variables):
        x = variables['x']
        return x[[0,-1]]
    
    def add_to(self, destination, value):
        nx = self.p.model.nx
        x_offset = self.p.decision['x'].offset
        npoints = self.p.tc.size
        xe = np.asarray(value)
        assert xe.shape == (2, nx)        
        destination[x_offset:][:nx] += xe[0]
        destination[x_offset + (npoints - 1)*nx:][:nx] += xe[1]
    
    def expand_indices(self, ind):
        nx = self.p.model.nx
        npoints = self.p.tc.size
        x_offset = self.p.decision['x'].offset
        ind = np.asarray(ind, dtype=int)
        return ind + x_offset + (ind >= nx) * (npoints - 2) * nx
