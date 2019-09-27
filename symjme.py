"""Symbolic JME model common code."""


import collections
import functools

import numpy as np
import sym2num.model
import sym2num.var
import sympy

from ceacoest import utils, rk


class ModelSubclass(sym2num.model.Base):
    """Symbolic LGL-collocation JME model base."""
    
    @property
    def derivatives(self):
        """List of the model function derivatives to calculate."""
        derivatives =  [('df_dx', 'f', 'x'),
                        ('de_dxp', 'e', 'xp_flat'),
                        ('d2e_dxp2', 'de_dxp', 'xp_flat'),
                        ('de_dwp', 'e', 'wp_flat'),
                        ('d2e_dwp2', 'de_dwp', 'wp_flat'),
                        ('de_dp', 'e', 'p'),
                        ('d2e_dp2', 'de_dp', 'p'),
                        ('d2e_dxp_dp', 'de_dxp', 'p'),
                        ('d2e_dxp_dwp', 'de_dxp', 'wp_flat'),
                        ('d2e_dwp_dp', 'de_dwp', 'p'),
                        ('dL_dxm', 'L', 'x'),
                        ('dL_dp', 'L', 'p'),
                        ('d2L_dxm2', 'dL_dxm', 'x'),
                        ('d2L_dp2', 'dL_dp', 'p'),
                        ('d2L_dxm_dp', 'dL_dxm', 'p'),
                        ('dJ_dxp', 'J', 'xp'),
                        ('dJ_dwp', 'J', 'wp'),
                        ('dJ_dp', 'J', 'p'),
                        ('d2J_dxp2', 'J', ('xp_flat', 'xp_flat')),
                        ('d2J_dwp2', 'J', ('wp_flat', 'wp_flat')),
                        ('d2J_dp2', 'dJ_dp', 'p'),
                        ('d2J_dxp_dp', 'J', ('xp_flat', 'p')),
                        ('d2J_dxp_dwp', 'J', ('xp_flat', 'wp_flat')),
                        ('d2J_dwp_dp', 'J', ('wp_flat', 'p'))]
        return getattr(super(), 'derivatives', []) + derivatives
    
    @property
    def generate_functions(self):
        """Iterable of the model functions to generate."""
        gen = {'e', 'f', 'L', 'J', 
               'dL_dxm', 'dL_dp', 'dJ_dxp', 'dJ_dwp', 'dJ_dp'}
        return getattr(super(), 'generate_functions', set()) | gen
    
    @property
    def generate_sparse(self):
        """List of the model functions to generate in a sparse format."""
        gen = ['de_dxp', 'de_dwp', 'de_dp', 
               'd2e_dxp_dp', 'd2e_dxp_dwp', 'd2e_dwp_dp',
               ('d2e_dxp2', lambda i,j,k: i<=j),
               ('d2e_dwp2', lambda i,j,k: i<=j),
               ('d2e_dp2', lambda i,j,k: i<=j),
               ('d2L_dxm2', lambda i,j: i<=j),
               ('d2L_dp2', lambda i,j: i<=j),
               'd2L_dxm_dp',
               ('d2J_dxp2', lambda i,j: i<=j),
               ('d2J_dwp2', lambda i,j: i<=j),
               ('d2J_dp2', lambda i,j: i<=j),
               'd2J_dxp_dp', 'd2J_dxp_dwp', 'd2J_dwp_dp']
        return getattr(super(), 'generate_sparse', []) + gen
    
    @property
    def generate_assignments(self):
        gen = {'nx': len(self.variables['x']),
               'nw': len(self.variables['w']),
               'nu': len(self.variables['u']),
               'np': len(self.variables['p']),
               'ny': len(self.variables['y']),
               'ne': len(self.default_function_output('e')),
               'collocation_order': self.collocation.n,
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
    
    @utils.cached_property
    def collocation(self):
        """Collocation method."""
        collocation_order = getattr(self, 'collocation_order', 2)
        return rk.LGLCollocation(collocation_order)
    
    @utils.cached_property
    def variables(self):
        """Model variables definition."""
        v = super().variables
        ncol = self.collocation.n

        x = [xi.name for xi in v['x']]
        w = [wi.name for wi in v['w']]
        u = [ui.name for ui in v['u']]
        
        # Piece states and controls
        xp = [[f'{n}_piece_{k}' for n in x] for k in range(ncol)]
        wp = [[f'{n}_piece_{k}' for n in w] for k in range(ncol)]
        up = [[f'{n}_piece_{k}' for n in u] for k in range(ncol)]
        
        additional_vars = sym2num.var.make_dict(
            [sym2num.var.SymbolArray('piece_len'),
             sym2num.var.SymbolArray('xp', xp),
             sym2num.var.SymbolArray('wp', wp),
             sym2num.var.SymbolArray('up', up),
             sym2num.var.SymbolArray('xp_flat', sympy.flatten(xp)),
             sym2num.var.SymbolArray('wp_flat', sympy.flatten(wp)),
             sym2num.var.SymbolArray('up_flat', sympy.flatten(up))]
        )
        return collections.OrderedDict([*v.items(), *additional_vars.items()])
    
    def e(self, xp, wp, up, p, piece_len):
        """Collocation defects (error)."""
        G = ndarray(self.G())
        wp = ndarray(wp)
        fp = []
        for i in range(self.collocation.n):
            fi = ndarray(self.f(xp[i, :], up[i, :], p))
            fp += [fi + G.dot(wp[i, :])]        
        fp = sympy.Matrix(fp)
        J = sympy.Matrix(self.collocation.J)
        dt = piece_len[()]
        
        xp = xp.tomatrix()
        defects = xp[1:, :] - xp[:-1, :] - dt * J * fp
        return sympy.Array(defects, len(defects))
    
    def inoisy(self):
        """Indices of noisy states."""
        G = self.G()
        inoisy = set()
        for i,j in np.ndindex(*G.shape):
            if G[i,j]:
                inoisy.add(i)
        return inoisy
    
    def J(self, xp, wp, up, p, piece_len):
        """Onsager--Machlup functional."""
        ncol = self.collocation.n        
        Ip = -0.5 * np.sum(ndarray(wp)**2, 1)
        for i in range(ncol):
            df_dx = self.df_dx(xp[i, :], up[i, :], p)
            divf = sum(df_dx[j,j] for j in self.inoisy())
            Ip[i] += -0.5 * divf
        
        K = self.collocation.K
        dt = piece_len[()]
        J = np.dot(Ip, K) * dt
        return sympy.Array(J)


def collocate(order=2):
    def decorator(BaseModel):
        @functools.wraps(BaseModel, updated=())
        class EstimationModel(ModelSubclass, BaseModel, sym2num.model.Base):
            collocation_order = order
        return EstimationModel
    return decorator


def ndarray(m):
    return np.array(m, dtype=object).reshape(np.shape(m))
