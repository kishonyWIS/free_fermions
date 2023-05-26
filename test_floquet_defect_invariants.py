from floquet_defect_invariants import get_unitary, get_Hamiltonian_lambda
import numpy as np
from scipy.linalg import eig
from matplotlib import pyplot as plt

kx = 0.5
ky = 0.5
theta = np.pi
lamb = 0.5
integration_params = dict(name='vode', nsteps=2000, rtol=1e-8, atol=1e-10)

h = get_Hamiltonian_lambda(kx, ky, theta, lamb, t=0.1)
print(h)
print(eig(h))

u = get_unitary(kx, ky, theta, lamb, integration_params, t0=0, tf=1)

#print the unitary
print('unitary')
print(u)

phases, states = eig(u)
energies = np.angle(phases)
# sort the energies and phases and states according to the energies
sort_indices = np.argsort(energies)
energies = energies[sort_indices]
phases = phases[sort_indices]
states = states[:, sort_indices]

print(phases)
print(states)
