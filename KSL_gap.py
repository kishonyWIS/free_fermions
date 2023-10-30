from translational_invariant_KSL import get_KSL_model, get_Delta, get_f
import numpy as np
from scipy.linalg import eigh

kappa = 0.1
Jx = 1.
Jy = 1.
Jz = 1.

n_k_points = 1+6*14

kx_list = np.linspace(-np.pi, np.pi, n_k_points)
ky_list = np.linspace(-np.pi, np.pi, n_k_points)

min_E = 100

for i_kx, kx in enumerate(kx_list):
    print(f'kx={kx}')
    for i_ky, ky in enumerate(ky_list):

        f = get_f(kx, ky, Jx, Jy, Jz)
        Delta = get_Delta(kx, ky, kappa)
        hamiltonian = np.array([[Delta, 1j*f],[-1j*np.conj(f), -Delta]])
        # get the eigenvalues of the hamiltonian
        E, U = eigh(hamiltonian)
        if E[1]<min_E:
            min_E = E[1]
            print(f'kx={kx}, ky={ky}, E={E[1]}')