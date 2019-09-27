"""Joint MAP state-path and parameter estimator."""

import itertools

import numpy as np

from ceacoest import optim, rk, utils


class Problem(optim.Problem):
    """Joint MAP state-path and parameter estimation problem."""
    
    def __init__(self, model, t, y, u):
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
        """Collocation time grid."""
        
        npoints = self.tc.size
        ncoarse = len(t)
        self.npoints = npoints
        """Total number of collocation points."""
        
        super().__init__()
        x = self.register_decision('x', model.nx, npoints)
        wp = self.register_decision('wp', (col.n, model.nw), npieces)
        p = self.register_decision('p', model.np)
        self.register_derived('xp', PieceRavelledVariable(self, 'x'))
        
        self.register_constraint(
            'e', model.e, ('xp','wp','up','p', 'piece_len'), model.ne, npieces
        )

        assert isinstance(y, np.ndarray)
        assert y.shape == (ncoarse, self.model.ny)
        
        ymask = np.ma.getmaskarray(y)
        kmeas_coarse, = np.nonzero(np.any(~ymask, axis=1))
        self.kmeas = kmeas_coarse * self.collocation.ninterv
        """Collocation time indices with active measurements."""
        
        self.y = y[kmeas_coarse]
        """Measurements at the time indices with active measurements."""
        
        self.nmeas = np.size(self.kmeas)
        """Number of measurement indices."""

        if callable(u):
            u = u(self.tc)
        assert isinstance(u, np.ndarray)
        assert u.shape == (self.npoints, model.nu)
        self.u = u
        """The inputs at the fine grid points."""
        
        self.um = self.u[self.kmeas]
        """The inputs at the measurement points."""

        up = np.zeros((npieces, self.collocation.n, model.nu))
        up[:, :-1].flat = u[:-1, :].flat
        up[:-1, -1] = up[1:, 0]
        up[-1, -1] = u[-1]
        self.up = up
        """Piece-ravelled inputs."""
        
        self.register_derived('xm', XMVariable(self))
        self._register_model_constraint_derivatives('e', ('xp','wp','p'))
        
        self.register_merit('L', model.L, ('y','xm','um','p'), self.nmeas)
        self.register_merit(
            'J', model.J, ('xp', 'wp', 'up', 'p', 'piece_len'), npieces
        )
        self._register_model_merit_derivatives('L', ('xm', 'p'))
        self._register_model_merit_derivatives('J', ('xp', 'wp', 'p'))
        
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
        return {'y': self.y, 'um': self.um, 'u': self.u, 'up': self.up,
                'piece_len': self.piece_len, **super().variables(dvec)}
    
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


class XMVariable:
    def __init__(self, problem):
        self.p = problem

    @property
    def tiling(self):
        return self.p.nmeas
    
    @property
    def shape(self):
        return (self.p.nmeas, self.p.model.nx)
    
    def build(self, variables):
        x = variables['x']
        return x[self.p.kmeas]
    
    def add_to(self, destination, value):
        assert np.shape(value) == self.shape
        x = self.p.decision['x'].unpack_from(destination)
        x[self.p.kmeas] += value
    
    def expand_indices(self, ind):
        x_offset = self.p.decision['x'].offset
        ind = np.asarray(ind, dtype=int)        
        return ind + x_offset + self.p.kmeas[:, None] * self.p.model.nx
