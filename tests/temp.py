import numpy as np
from qsppack.utils import cvx_poly_coef
from qsppack.solver import solve

# set parameters for polynomial approximation
beta = 2
targ = lambda x: np.exp(-beta * x)
deg = 151
parity = deg % 2
delta = 0.2

# options for cvx_poly_coef
opts = {
    'intervals': [delta, 1],
    'objnorm': 2,
    'epsil': 0.2,
    'npts': 500,
    'fscale': 1,
    'isplot': False,
    'method': 'cvxpy',
    'maxiter': 100
}

coef_full = cvx_poly_coef(targ, deg, opts)
coef = coef_full[parity::2]

opts['criteria'] = 1e-12
opts['useReal'] = False
opts['targetPre'] = True
opts['method'] = 'Newton'
angles, out = solve(coef, parity, opts)