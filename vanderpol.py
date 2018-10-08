"""Output error method estimation of a Van der Pol oscillator."""


import functools

import numpy as np
import sympy
import sym2num.model
import sym2num.utils
from numpy import ma
from scipy import integrate, interpolate
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


def generate_data(model, p, seed=0):
    def xdot(t, x):
        return model.f(x, [], p)
    x0 = [0, 1]
    tspan = [0, 20]
    sol = integrate.solve_ivp(xdot, tspan, x0, dense_output=True)
    
    np.random.seed(seed)    
    meas_std = p[1]
    
    t = np.linspace(*tspan, 2001)
    x = sol.sol(t).T
    y = ma.masked_all((t.size, 2))

    downsample = 10
    nmeas = y[::downsample].shape[0]
    y[::downsample, 0] = x[::downsample, 0] + meas_std*np.random.randn(nmeas)
    return (t, y, x)


if __name__ == '__main__':
    symb_model = VanDerPol()
    GeneratedVanDerPol = sym2num.model.compile_class(symb_model)
    model = GeneratedVanDerPol()

    p = np.r_[2, 0.1]
    (t, y, x) = generate_data(model, p)
    
    
