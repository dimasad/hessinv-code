"""Duffing SDE JME estimation test."""


import datetime
import json
import os

import numpy as np
import sympy
import sym2num.model

from ceacoest import oem, optim
from ceacoest.modelling import symstats

import symjme


class DuffingSim(sym2num.model.Base):
    """Symbolic Duffing oscillator simulation model."""

    @property
    def derivatives(self):
        """List of the model function derivatives to calculate."""
        derivatives =  [('df_dt', 'f', 't'),
                        ('df_dx', 'f', 'x'),
                        ('d2f_dx2', 'df_dx', 'x')]
        return getattr(super(), 'derivatives', []) + derivatives
    
    @property
    def variables(self):
        v = super().variables
        v['x'] = ['x1', 'z1']
        v['t'] = 't'
        v['self'] = {
            'consts': ['a', 'b', 'd', 'gamma', 'meas_std', 'sigmad'],
            'dt': 'dt'
        }
        return v
    
    @sym2num.model.collect_symbols
    def f(self, x, t, *, s):
        """Drift function."""
        dx1 = -s.a*s.z1**3 - s.b*s.z1 - s.d*s.x1 + s.gamma*sympy.cos(s.t)
        dz1 = s.x1
        return [dx1, dz1]

    @sym2num.model.collect_symbols
    def G(self, *, s):
        """Diffusion matrix."""
        return [[s.sigmad], [0]]
    
    def fd(self, x, t):
        """Discretized drift for simulation."""
        x = sympy.Array(x)
        f = self.f(x, t)
        df_dt = self.df_dt(x, t)
        df_dx = self.df_dx(x, t).tomatrix()
        d2f_dx2 = self.d2f_dx2(x, t)
        G = self.G()
        dt = self.dt[()]
        
        nw = G.shape[1]
        nx = len(x)
        
        # Calculate L0f
        L0f = df_dt.tolist()
        for i, k in np.ndindex(nx, nx):
            L0f[k] += df_dx[i, k] * f[i]
        for k, j, p, q in np.ndindex(nx, nw, nx, nx):
            L0f[k] += 0.5 * G[p, j] * G[q, j] * d2f_dx2[p, q, k]
        L0f = sympy.Array(L0f)
        
        # Discretized drift
        fd = x + f * dt + 0.5 * L0f * dt ** 2
        return fd
    
    def Gd(self, x, t):
        """Discrete-time noise gain."""
        df_dx = self.df_dx(x, t).tomatrix()
        G = self.G().tomatrix()
        dt = self.dt[()]
        
        nw = G.shape[1]
        nx = len(x)
        Lf = df_dx.T * G
        Gd = sympy.MutableDenseNDimArray.zeros(nx, 2*nw)
        Gd[:, :nw] = G * dt ** 0.5 + 0.5 * Lf * dt ** 1.5
        Gd[:, -nw:] = 0.5 / sympy.sqrt(3) * Lf * dt ** 1.5
        return Gd
    
    @property
    def generate_functions(self):
        """Iterable of the model functions to generate."""
        gen = {'f', 'df_dx', 'fd', 'Gd'}
        return getattr(super(), 'generate_functions', set()) | gen
    
    @property
    def generate_assignments(self):
        gen = {'nx': len(self.variables['x']),
               'symbol_index_map': self.symbol_index_map,
               'array_shape_map': self.array_shape_map,
               'array_element_names': self.array_element_names,
               **getattr(super(), 'generate_assignments', {})}
        return gen
    
    @property
    def generate_imports(self):
        """List of imports to include in the generated class code."""
        return ['sym2num.model'] + getattr(super(), 'generate_imports', [])
    
    @property
    def generated_bases(self):
        """Base classes of the generated model class."""
        bases = ['sym2num.model.ModelArrayInitializer']
        return bases + getattr(super(), 'generated_bases', [])


if __name__ == '__main__':
    # Experiment parameters
    seed = 1
    dt  = 0.005
    tf = 200
    Ts = 0.1
    meas_std = 0.1
    given = {
        'a': 1, 'b': -1, 'd': 0.2, 'gamma': 0.3, 'sigmad': 0.1, 'dt': dt,
        'meas_std': meas_std, 'tf': tf, 'seed': seed, 'Ts': Ts
    }
    
    # Compile and instantiate model
    symb_mdl = DuffingSim()
    GeneratedModel = sym2num.model.compile_class(symb_mdl)
    model = GeneratedModel(**given)

    # Set the random number generator to a known state
    np.random.seed(seed)

    # Simulate the SDE
    N = int(tf / dt) + 1
    tsim = np.arange(N) * dt
    xsim = np.empty((N, model.nx))
    xsim[0] = [1, 1]
    for k in range(N - 1):
        fd = model.fd(xsim[k], tsim[k])
        Gd = model.Gd(xsim[k], tsim[k])
        w = np.random.randn(Gd.shape[1])
        xsim[k + 1] = fd + Gd @ w

    # Generate the measurements
    meas_subsample = int(Ts / dt)
    tmeas = tsim[::meas_subsample]
    y = xsim[::meas_subsample, [1]]
    y += meas_std * np.random.randn(*y.shape)
    
    # Save the data
    now = datetime.datetime.today()
    datadir = os.path.join('data', f'duffing_{now:%Y-%m-%d_%Hh%Mm%Ss}')
    os.makedirs(datadir)
    np.savetxt(os.path.join(datadir, 'sim.txt'), np.c_[tsim, xsim])
    np.savetxt(os.path.join(datadir, 'meas.txt'), np.c_[tmeas, y])
    with open(os.path.join(datadir, 'params.json'), mode='w') as param_file:
        json.dump(given, param_file)
