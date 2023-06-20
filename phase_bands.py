from floquet_defect_invariants import get_unitary, get_Hamiltonian_lambda, get_unitary_odeint, get_unitary_solve_ivp, get_effective_hamiltonian, get_u_eff
import numpy as np
from scipy.linalg import eig, expm, logm
from matplotlib import pyplot as plt
from itertools import permutations

SIGMA_X = np.array([[0,1],[1,0]])
SIGMA_Y = np.array([[0,-1j],[1j,0]])
SIGMA_Z = np.array([[1,0],[0,-1]])


def diagonalize_2_by_2_unitary(u):
    # decompose unitary in the Pauli basis
    # u = cos(alpha) + i sin(alpha) (n_x sigma_x + n_y sigma_y + n_z sigma_z)
    n_x_sin_alpha = np.trace(-1j * u @ SIGMA_X)/2
    n_y_sin_alpha = np.trace(-1j * u @ SIGMA_Y)/2
    n_z_sin_alpha = np.trace(-1j * u @ SIGMA_Z)/2
    n_sin_alpha = np.array([n_x_sin_alpha, n_y_sin_alpha, n_z_sin_alpha])
    sin_alpha = np.linalg.norm(n_sin_alpha)
    n = n_sin_alpha / sin_alpha
    cos_alpha = np.trace(u)/2
    eigenvalues = np.array([cos_alpha + 1j * sin_alpha, cos_alpha - 1j * sin_alpha])
    if n[2] == -1:
        eigenvectors = np.array([[0,1],[1,0]])
    else:
        first_eigenvector = np.array([n[2] + 1, n[0] + 1j * n[1]])
        first_eigenvector /= np.linalg.norm(first_eigenvector)
        second_eigenvector = np.array([-n[0] + 1j * n[1], n[2] + 1])
        second_eigenvector /= np.linalg.norm(second_eigenvector)
        eigenvectors = np.stack([first_eigenvector, second_eigenvector], axis=-1)
    return eigenvalues, eigenvectors


u = expm(1j * np.pi * SIGMA_Z / 2)
eigenvalues, eigenvectors = diagonalize_2_by_2_unitary(u)
print(eigenvalues)
print(eigenvectors)


nsteps = 31

integration_params = dict(name='vode', nsteps=nsteps, rtol=1e-8, atol=1e-10)

kx_list = np.linspace(-np.pi, np.pi, 11)
ky = np.pi
times = np.linspace(0, 1, nsteps)
theta_list = np.linspace(-np.pi, np.pi, 11)
lamb = 0.0

u = np.zeros((len(kx_list), len(theta_list), len(times), 2, 2), dtype=np.complex128)
logu_in_pauli_basis = np.zeros((len(kx_list), len(theta_list), len(times), 3))
s = np.zeros((len(kx_list), len(theta_list), len(times), 3, 3))
phases = np.zeros((len(kx_list), len(theta_list), len(times), 2), dtype=np.complex128)

#iterate over all the values of ky, theta
#and calculate the unitary for each
for i, kx in enumerate(kx_list):
    print(i)
    for k, theta in enumerate(theta_list):
        u[i,k,:,:,:] = get_unitary_solve_ivp(kx, ky, theta, lamb, integration_params, t0=0, tf=1).transpose(2,0,1)
        for it in range(len(times)):
            phases[i,k,it,:], states = eig(u[i,k,it,:,:])
            logu = logm(u[i,k,it,:,:])
            logu_in_pauli_basis[i, k, it, 0] = np.trace(1j * logu @ SIGMA_X)
            logu_in_pauli_basis[i, k, it, 1] = np.trace(1j * logu @ SIGMA_Y)
            logu_in_pauli_basis[i, k, it, 2] = np.trace(1j * logu @ SIGMA_Z)
phases = np.angle(phases)

s = np.stack(np.gradient(logu_in_pauli_basis, axis=[0,1,2]), axis=-1)
sign_det_s = np.sign(np.linalg.det(s))

top_band_phases = phases.max(axis=-1)
sign_det_s[top_band_phases > 3.1415]
plt.plot(top_band_phases.reshape(-1,nsteps).T)
plt.show()