"""Analyze results of the OEM estimation of a Van der Pol oscillator."""


import os
import re

import numpy as np


def get_result(datadir, prefix, ind=slice(None)):
    out = []
    pattern =  f'{prefix}-\\d+.txt$'
    for f in os.listdir(datadir):
        if re.match(pattern, f):
            out.append(np.loadtxt(os.path.join(datadir, f))[ind])
    return np.array(out)


if __name__ == '__main__':
    datadir = 'data/vanderpol_2018-10-08_23h34m53s'
    
    popt = get_result(datadir, 'popt')
    pcov = get_result(datadir, 'pcov')
    pvar = np.diagonal(pcov, axis1=1, axis2=2)
    pstd = np.sqrt(pvar)    
    tblp = np.c_[np.std(popt, 0), np.mean(pstd, 0), np.std(pstd, 0)]
    
    x0opt = get_result(datadir, 'xopt', 0)
    xvar = get_result(datadir, 'xvar')
    xstd = np.sqrt(xvar)
    tblx0 = np.c_[np.std(x0opt, 0), np.mean(xstd, 0), np.std(xstd, 0)]
    
    with open('tbl.txt', 'w') as f:
        print(r'$\mu$', *tblp[0], file=f, sep=', ')
        print(r'$\sigma$', *tblp[1], file=f, sep=', ')
        print(r'$x_1(0)$', *tblx0[0], file=f, sep=', ')
        print(r'$x_2(0)$', *tblx0[1], file=f, sep=', ')

    np.savetxt('mu_est.txt', popt[:, 0])
    np.savetxt('sigma_est.txt', popt[:, 1])
    np.savetxt('x1_t0_est.txt', x0opt[:, 0])
    np.savetxt('x2_t0_est.txt', x0opt[:, 1])
    

