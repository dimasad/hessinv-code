"""Output error method estimation of a Van der Pol oscillator."""


import datetime
import functools
import os

import numpy as np
import sympy
import sym2num.model
import sym2num.utils
from numpy import ma
from scipy import integrate, interpolate, sparse
from sym2num import var
from sympy import cos, sin

from ceacoest import oem, optim
from ceacoest.modelling import symoem, symstats


@symoem.collocate(order=2)
class VanDerPol:
    """Symbolic model of a Van der Pol oscillator."""
    
    @sym2num.utils.classproperty
    @functools.lru_cache()
    def variables(cls):
        """Model variables definition."""
        vars = [var.SymbolArray('x', ['x1', 'x2']),
                var.SymbolArray('y', ['x1_meas']),
                var.SymbolArray('u', []),
                var.SymbolArray('p', ['mu', 'meas_std'])]
        return var.make_dict(vars)
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """ODE function."""
        x1d = s.x2
        x2d = s.mu * (1 - s.x1**2) * s.x2 - s.x1
        return [x1d, x2d]
    
    @sym2num.model.collect_symbols
    def L(self, y, x, u, p, *, s):
        """Measurement log likelihood."""
        return symstats.normal_logpdf1(s.x1_meas, s.x1, s.meas_std)


def sim(model, p):
    def xdot(t, x):
        return model.f(x, [], p)
    x0 = [0, 1]
    tspan = [0, 20]
    sol = integrate.solve_ivp(xdot, tspan, x0, dense_output=True)
    
    t = np.linspace(*tspan, 2001)
    x = sol.sol(t).T
    return (t, x)


def output(t, x, std, downsample, seed=0):
    np.random.seed(seed)
    y = ma.masked_all((t.size, 1))
    
    nmeas = y[::downsample].shape[0]
    y[::downsample, 0] = x[::downsample, 0] + std*np.random.randn(nmeas)
    return y


def estimate(model, t, y):
    u = lambda t: np.zeros((t.size, 0))
    problem = oem.Problem(model, t, y, u)
    tc = problem.tc
        
    # Set initial guess
    dec0 = np.zeros(problem.ndec)
    x1_guess = interpolate.interp1d(t[::downsample], y[::downsample, 0])(tc)
    problem.set_decision_item('x1', x1_guess, dec0)
    problem.set_decision_item('meas_std', 0.5, dec0)
    
    # Set bounds
    constr_bounds = np.zeros((2, problem.ncons))
    dec_L, dec_U = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    problem.set_decision_item('meas_std', 0.00001, dec_L)
    problem.set_decision_item('meas_std', 10, dec_U)
    
    with problem.ipopt((dec_L, dec_U), constr_bounds) as nlp:
        nlp.add_num_option('obj_scaling_factor', -1)
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        decopt, info = nlp.solve(dec0)
        
    HL = lag_hess(problem, decopt, info['mult_g'])
    
    opt = problem.variables(decopt)
    return problem, opt, decopt, info


def lag_hess(problem, d, lmult):
    d2L_dz2_val = problem.lagrangian_hessian_val(d, 1, lmult)
    d2L_dz2_ind = problem.lagrangian_hessian_ind()
    diag = d2L_dz2_ind[0] == d2L_dz2_ind[1]
    
    d2L_dz2_data = (np.r_[d2L_dz2_val, d2L_dz2_val[~diag]],
                    np.c_[d2L_dz2_ind, d2L_dz2_ind[~diag, [1,0]]])
    d2L_dz2 = sparse.coo_matrix(d2L_dz2_data, (problem.ndec,)*2)

    jac_val = problem.constraint_jacobian_val(d)
    jac_ind = problem.constraint_jacobian_ind()
    jac = sparse.coo_matrix((jac_val, jac_ind), (problem.ndec, problem.ncons))
    jacT = sparse.coo_matrix((jac_val, jac_ind[:,[1,0]]), (problem.ndec, problem.ncons))
    
    null = sparse.coo_matrix((problem.ncons,) * 2)
    lag_hess = sparse.bmat([[d2L_dz2, jac], [jacT, null]], 'csr')
    return lag_hess


if __name__ == '__main__':
    symb_model = VanDerPol()
    GeneratedVanDerPol = sym2num.model.compile_class(symb_model)
    model = GeneratedVanDerPol()
    
    mu = 2
    std = 0.1
    downsample = 10
    
    p = np.r_[mu, std]
    (t, x) = sim(model, p)
    
    now = datetime.datetime.today()
    datadir = os.path.join('data', f'vanderpol_{now:%Y-%m-%d_%Hh%Mm%Ss}')
    os.makedirs(datadir)
    np.savetxt(os.path.join(datadir, 'sim.txt'), np.c_[t,x])
    np.savetxt(os.path.join(datadir, 'param.txt'), p)
    
    for seed in range(10):
        y = output(t, x, std, downsample, seed)
        problem, opt, decopt = estimate(model, t, y)
        
