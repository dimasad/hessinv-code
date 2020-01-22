"""Duffing SDE JME estimation test."""


import json
import os
import sys

import numpy as np
import sympy
import sym2num.model
import scipy.sparse.linalg
from scipy import interpolate, sparse, stats
from sksparse import cholmod

from ceacoest.modelling import symstats

import jme
import symjme


@symjme.collocate(order=2)
class Duffing:
    """Symbolic Duffing oscillator JME model."""

    @property
    def variables(self):
        v = super().variables
        v['x'] = ['x1', 'z1']
        v['w'] = ['w1']
        v['y'] = ['z_meas']
        v['u'] = ['u1']
        v['p'] = ['a', 'b', 'd', 'meas_std']
        v['self'] = {'consts': ['sigmad']}
        return v
    
    @sym2num.model.collect_symbols
    def f(self, x, u, p, *, s):
        """Drift function."""
        dx1 = -s.a*s.z1**3 - s.b*s.z1 - s.d*s.x1 + s.u1
        dz1 = s.x1
        return [dx1, dz1]

    @sym2num.model.collect_symbols
    def G(self, *, s):
        """Diffusion matrix."""
        return [[s.sigmad], [0]]
    
    @sym2num.model.collect_symbols
    def L(self, y, x, u, p, *, s):
        """Measurement log likelihood."""
        return symstats.normal_logpdf1(s.z_meas, s.z1, s.meas_std)


def lag_hess(problem, d, lmult):
    d2L_dd2_val = problem.lagrangian_hessian_val(d, 1, lmult)
    d2L_dd2_ind = problem.lagrangian_hessian_ind()
    diag = d2L_dd2_ind[0] == d2L_dd2_ind[1]
    
    d2L_dd2_data = (np.r_[d2L_dd2_val, d2L_dd2_val[~diag]],
                    np.c_[d2L_dd2_ind, d2L_dd2_ind[::-1, ~diag]])
    d2L_dd2 = sparse.coo_matrix(d2L_dd2_data, (problem.ndec,)*2)
    
    jac_val = problem.constraint_jacobian_val(d)
    jac_ind = problem.constraint_jacobian_ind()
    jac = sparse.coo_matrix((jac_val, jac_ind), (problem.ndec, problem.ncons))
    jacT = jac.transpose()
    
    null = sparse.coo_matrix((problem.ncons,) * 2)
    lag_hess = sparse.bmat([[d2L_dd2, jac], [jacT, null]], 'csc')
    return lag_hess


if __name__ == '__main__':
    try:
        data_dir = sys.argv[1]
    except IndexError:
        data_dir = 'data/duffing'
    
    # Load data
    with open(f'{data_dir}/params.json') as param_file:
        given = json.load(param_file)    
    meas = np.loadtxt(f'{data_dir}/meas.txt')
    tmeas = meas[:, 0]
    nmeas = len(meas)
    u = lambda t: given['gamma'] * np.cos(np.asarray(t)[..., None])
    
    # Compile and instantiate model
    symb_mdl = Duffing()
    GeneratedModel = sym2num.model.compile_class(symb_mdl)
    model = GeneratedModel(sigmad=given['sigmad'])
    
    # Choose estimation problem coarse time grid and prepare measurements
    subdivide = 2
    t = np.linspace(tmeas[0], tmeas[-1], subdivide*(nmeas - 1) + 1)
    y = np.ma.masked_all((t.size, 1))
    y[::subdivide] = meas[:, 1:]

    # Create estimation problem
    problem = jme.Problem(model, t, y, u)
    tc = problem.tc
    
    # Create LSQ spline approximation of the measurements for the initial guess
    Tknot = 1.05
    knots = np.arange(tmeas[0] + 2 * Tknot, tmeas[-1] - 2 * Tknot, Tknot)
    z_guess = interpolate.LSQUnivariateSpline(tmeas, meas[:, 1], knots, k=5)

    # Make Linear Least Squares approximation of the parameters
    psi = np.c_[-z_guess(tmeas)**3, -z_guess(tmeas), -z_guess(tmeas, 1)]
    (a, b, d), *_ = np.linalg.lstsq(psi, z_guess(tmeas, 2), rcond=None)
    meas_std = np.std(y.flatten() - z_guess(t))
    
    # Set bounds
    lower = {'meas_std': 1e-3}
    constr_bounds = np.zeros((2, problem.ncons))
    dec_bounds = np.repeat([[-np.inf], [np.inf]], problem.ndec, axis=-1)
    dec_L, dec_U = dec_bounds
    for k,v in lower.items():
        problem.set_decision_item(k, v, dec_L)    
    
    # Set initial guess
    dec0 = np.zeros(problem.ndec)
    problem.set_decision_item('z1', z_guess(tc), dec0)
    problem.set_decision_item('x1', z_guess.derivative()(tc), dec0)
    problem.set_decision_item('meas_std', meas_std, dec0)
    problem.set_decision_item('a', a, dec0)
    problem.set_decision_item('b', b, dec0)
    problem.set_decision_item('d', d, dec0)
    
    # Set problem scaling
    dec_scale = np.ones(problem.ndec)
    constr_scale = np.ones(problem.ncons)
    
    # Call NLP solver
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 1000)
        nlp.set_scaling(-1, dec_scale, constr_scale)
        decopt, info = nlp.solve(dec0)

    # Unpack optimal solution
    opt = problem.variables(decopt)
    opt_merit = info['obj_val']
    wpopt = opt['wp']
    xopt = opt['x']
    popt = opt['p']

    # Save MAP estimates
    #np.savez(f'{data_dir}/map.npz', tc=tc, xopt=xopt, popt=popt)

    # Construct and factorize the Lagrangian Hessian
    HL = lag_hess(problem, decopt, info['mult_g'])
    HL_inv = sparse.linalg.factorized(HL)
    n = HL.shape[0]
    
    # Select the independent variables
    x_slice = problem.decision['x'].slice
    p_slice = problem.decision['p'].slice
    independent = np.r_[
        np.arange(x_slice.start, x_slice.stop, 2),
        np.arange(p_slice.start, p_slice.stop, 1),
        x_slice.start + 1
    ]
    nindep = independent.size

    # Approximate the covariance matrix of the independent variables
    X = np.zeros((problem.ndec, nindep))
    for j, i in enumerate(independent):
        b = np.zeros(n)
        b[i] = -1
        X[:, j] = HL_inv(b)[:problem.ndec]
        print(f'Calculating inverse Hessian column {j+1} of {nindep}')
    
    cov_i = np.repeat(np.arange(problem.ndec), nindep)
    cov_j = np.tile(independent, problem.ndec)
    cov_val = X.ravel()
    cov_dec = sparse.csc_matrix((cov_val, (cov_i, cov_j)))
    cov_compressed = cov_dec[:, independent][independent, :]
    
    # Save the covariance
    #np.savez(f'{data_dir}/map_cov.npz', cov_dec=cov_dec)

    # Compute the correlation coefficients for plotting
    k = xopt.shape[0] // 2
    x1_slice = slice(x_slice.start, x_slice.stop, 2)
    x1_std = np.sqrt(cov_dec.diagonal()[x1_slice])
    p_std = np.sqrt(cov_dec.diagonal()[p_slice.start:p_slice.stop])
    x1_acorr = cov_dec[x1_slice, 2*k].toarray().ravel() / x1_std / x1_std[k]
    
    poff = p_slice.start
    x1_p1_corr = cov_dec[x1_slice, poff+1].toarray().ravel() / x1_std / p_std[1]

    # Save the correlation coefficients for plotting
    np.savetxt(f'{data_dir}/x1_acorr.txt', np.c_[tc, x1_acorr])
    np.savetxt(f'{data_dir}/x1_p1_corr.txt', np.c_[tc, x1_p1_corr])

    # Compute Cholesky factorization of the covariance
    factor = cholmod.cholesky(cov_compressed, beta=0)
    L = factor.L()
    
    # Run MCMC
    np.random.seed(1)
    Nsamp = 2000
    sg = 1.8 / np.sqrt(nindep)
    xchain = np.zeros((Nsamp,) + xopt.shape)
    pchain = np.zeros((Nsamp, model.np))

    accepted = 0
    xchain[0] = xopt
    pchain[0] = popt

    # Initialize chain
    dec_chain = decopt.copy()
    dec_chain[independent] += sg * factor.apply_Pt(L * np.random.randn(nindep))
    dec_bounds[:, independent] = dec_chain[independent]
    
    # Solve for dependent variables
    with problem.ipopt(dec_bounds, constr_bounds) as nlp:
        nlp.add_str_option('linear_solver', 'ma57')
        nlp.add_num_option('tol', 1e-6)
        nlp.add_int_option('max_iter', 100)
        nlp.set_scaling(-1, dec_scale, constr_scale)
        dec_chain, info_chain0 = nlp.solve(dec_chain)
    
    prev_logdens = info_chain0['obj_val']

    i = 1
    j = 0
    steps = 0
    while i < Nsamp:
        # Sample perturbations
        step  = 'Gibbs' if i % 15 else 'MH'
        
        if step == 'MH':
            # Do full Metropolis--Hastings jump
            perturb = sg * np.random.randn(nindep)
            dec_candidate = dec_chain.copy()
            dec_candidate[independent] += factor.apply_Pt(L * perturb)
        else:
            # Perform Gibbs jump
            perturb = 3 * np.random.randn()
            dec_candidate = dec_chain.copy()
            dec_candidate[independent[j]] += L[j, j] * perturb
        
        # Fix candidate decision variables and solve for the dependent
        dec_bounds[:, independent] = dec_candidate[independent]
        
        # Call NLP solver
        with problem.ipopt(dec_bounds, constr_bounds) as nlp:
            nlp.add_str_option('linear_solver', 'ma57')
            nlp.add_num_option('tol', 1e-6)
            nlp.add_int_option('max_iter', 100)
            nlp.set_scaling(-1, dec_scale, constr_scale)
            dec_candidate, info_candidate = nlp.solve(dec_candidate)
        
        candidate_logdens = info_candidate['obj_val']
        aprob = min(1, np.exp(candidate_logdens - prev_logdens))
    
        r = np.random.random()
        if r < aprob:
            dec_chain = dec_candidate
            prev_logdens = candidate_logdens
            accepted += 1
            print(f"chain {step} step accepted, r={r} prev_ld={prev_logdens} "
                  f"candidate_ld={candidate_logdens}")
        else:
            print(f"chain {step} step rejected, r={r} prev_ld={prev_logdens} "
                  f"candidate_ld={candidate_logdens}")
        steps += 1
        print(f"accepted = {accepted} of {steps}  ({accepted*100/steps}%)")

        if step == 'MH':
            var = problem.variables(dec_chain)
            xchain[i] = var['x']
            pchain[i] = var['p']
            i += 1
        else:
            j = (j + 1) % nindep
            if j == 0:
                var = problem.variables(dec_chain)
                xchain[i] = var['x']
                pchain[i] = var['p']
                i += 1
    
    # Save chain
    np.savez(f'{data_dir}/chain.npz', xchain=xchain, pchain=pchain)

    # Save sigma_y histogram
    h = np.histogram(pchain[:,-1], bins=30, density=True)
    hist_data = np.c_[h[1], np.r_[h[0], 0]]
    np.savetxt(f'{data_dir}/sigma_y_hist.txt', hist_data)
    

