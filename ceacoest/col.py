"""Common code for collocated optimization problems."""


import itertools

import numpy as np

from . import optim, rk, utils


class Problem(optim.Problem):
    """Collocated optimization problem base."""
    
    def __init__(self, model, t):
        # Initialize base class
        super().__init__()
        
        self.model = model
        """Underlying model."""
        
        col = rk.LGLCollocation(model.collocation_order)
        self.collocation = col
        """Collocation method."""
        
        assert np.ndim(t) == 1
        self.piece_len = np.diff(t)
        """Normalized length of each collocation piece."""
        
        assert np.all(self.piece_len > 0)
        npieces = len(self.piece_len)
        self.npieces = npieces
        """Number of collocation pieces."""
        
        self.tc = self.collocation.grid(t)
        """Normalized collocation time grid."""
        
        npoints = self.tc.size
        self.npoints = npoints
        """Total number of collocation points."""
        
        self.add_decision('x', (self.npoints, model.nx))
        self.add_constraint(model.e, (npieces, col.ninterv, model.nx))
        #self.register_derived('xp', PieceRavelledVariable(self, 'x'))
        
        #self.register_constraint(
        #    'e', model.e, ('xp','up','p', 'piece_len'), model.ne, npieces
        #)


class OldCollocatedProblem(optim.OldProblem):
    """Collocated optimization problem base."""
    
    def __init__(self, model, t):
        self.model = model
        """Underlying model."""
        
        self.collocation = col = rk.LGLCollocation(model.collocation_order)
        """Collocation method."""
        
        assert np.ndim(t) == 1
        self.piece_len = np.diff(t)
        """Normalized length of each collocation piece."""
        
        assert np.all(self.piece_len > 0)
        self.npieces = npieces = len(self.piece_len)
        """Number of collocation pieces."""
        
        self.tc = self.collocation.grid(t)
        """Normalized collocation time grid."""
        
        npoints = self.tc.size
        self.npoints = npoints
        """Total number of collocation points."""
        
        super().__init__()
        x = self.register_decision('x', model.nx, npoints)
        p = self.register_decision('p', model.np)
        self.register_derived('xp', PieceRavelledVariable(self, 'x'))
        
        self.register_constraint(
            'e', model.e, ('xp','up','p', 'piece_len'), model.ne, npieces
        )
    
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
    
    def set_decision_item(self, name, value, dvec):
        self._set_decision_item(name, value, self.model.symbol_index_map, dvec)

    def set_defect_scale(self, name, value, cvec):
        component_name, index = self.model.symbol_index_map[name]
        if component_name != 'x':
            raise ValueError(f"'{name}' is not a component of the state vector")
        e = self.constraints['e'].unpack_from(cvec)
        e = e.reshape((self.npieces, self.collocation.ninterv, self.model.nx))
        e[(..., *index)] = value


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

