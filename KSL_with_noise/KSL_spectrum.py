import itertools

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize

from translational_invariant_KSL import get_KSL_model, get_Delta, get_f
from time_dependence_functions import get_g, get_B
import seaborn as sns
import matplotlib as mpl

def gap(kx, ky, Jx, Jy, Jz, kappa):
    f = get_f(kx, ky, Jx, Jy, Jz)
    Delta = get_Delta(kx, ky, kappa)
    return np.sqrt(abs(f**2)+abs(Delta**2))

if __name__ == '__main__':
    kappa = 1
    Jx = 1
    Jy = 1
    Jz = 1

    kx = np.random.uniform(0, 2*np.pi)
    ky = np.random.uniform(0, 2*np.pi)
    res = minimize(lambda x: gap(x[0], x[1], Jx, Jy, Jz, kappa), (kx, ky), bounds=[(0, 2*np.pi), (0, 2*np.pi)])
    print(res.x)
    print(res.fun)
    print()