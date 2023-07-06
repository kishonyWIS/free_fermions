from floquet_defect_invariants import get_unitary, get_Hamiltonian_lambda, get_unitary_odeint, get_unitary_solve_ivp, get_effective_hamiltonian, get_u_eff
import numpy as np
from scipy.linalg import eig, expm, logm
from matplotlib import pyplot as plt
from itertools import permutations
from floquet_honeycomb_evolution import get_unitary_evolution

SIGMA_X = np.array([[0,1],[1,0]])
SIGMA_Y = np.array([[0,-1j],[1j,0]])
SIGMA_Z = np.array([[1,0],[0,-1]])


# def diagonalize_2_by_2_unitary(u):
#     # decompose unitary in the Pauli basis
#     # u = cos(alpha) + i sin(alpha) (n_x sigma_x + n_y sigma_y + n_z sigma_z)
#     n_x_sin_alpha = np.trace(-1j * u @ SIGMA_X)/2
#     n_y_sin_alpha = np.trace(-1j * u @ SIGMA_Y)/2
#     n_z_sin_alpha = np.trace(-1j * u @ SIGMA_Z)/2
#     n_sin_alpha = np.array([n_x_sin_alpha, n_y_sin_alpha, n_z_sin_alpha])
#     sin_alpha = np.linalg.norm(n_sin_alpha)
#     n = n_sin_alpha / sin_alpha
#     cos_alpha = np.trace(u)/2
#     eigenvalues = np.array([cos_alpha + 1j * sin_alpha, cos_alpha - 1j * sin_alpha])
#     if n[2] == -1:
#         eigenvectors = np.array([[0,1],[1,0]])
#     else:
#         first_eigenvector = np.array([n[2] + 1, n[0] + 1j * n[1]])
#         first_eigenvector /= np.linalg.norm(first_eigenvector)
#         second_eigenvector = np.array([-n[0] + 1j * n[1], n[2] + 1])
#         second_eigenvector /= np.linalg.norm(second_eigenvector)
#         eigenvectors = np.stack([first_eigenvector, second_eigenvector], axis=-1)
#     return eigenvalues, eigenvectors



# nsteps = 2*3*4*5+1
# nsteps = 2
nsteps = 91

integration_params = dict(name='vode', nsteps=nsteps, rtol=1e-8, atol=1e-10)

kx_list = np.linspace(0, np.pi, 31)
ky = 0.
times = np.linspace(0, 1, nsteps)
theta_list = np.linspace(0, 2*np.pi, 31)
lamb = 0.0

u = np.zeros((len(kx_list), len(theta_list), len(times), 2, 2), dtype=np.complex128)
phases = np.zeros((len(kx_list), len(theta_list), len(times), 2), dtype=np.complex128)
angles = np.zeros((len(kx_list), len(theta_list), len(times), 2))
states = np.zeros((len(kx_list), len(theta_list), len(times), 2, 2), dtype=np.complex128)

#iterate over all the values of ky, theta
#and calculate the unitary for each
# for i_kx, kx in enumerate(kx_list):
#     print(i_kx)
#     for i_theta, theta in enumerate(theta_list):
#         u[i_kx, i_theta, :, :, :] = get_unitary_solve_ivp(kx, ky, theta, lamb, integration_params, t0=0, tf=1).transpose(2, 0, 1)
#         for it in range(len(times)):
#             phases[i_kx, i_theta, it, :], states[i_kx, i_theta, it, :, :] = eig(u[i_kx, i_theta, it, :, :])
#             # sort the eigenvalues and eigenvectors by the phase of the eigenvalues
#             idx = np.argsort(-np.angle(phases[i_kx, i_theta, it, :]))
#             phases[i_kx, i_theta, it, :] = phases[i_kx, i_theta, it, idx]
#             states[i_kx, i_theta, it, :, :] = states[i_kx, i_theta, it, :, idx].T

for i_kx, kx in enumerate(kx_list):
    print(i_kx)
    for i_theta, theta in enumerate(theta_list):
        for i_t, time in enumerate(times):
            u[i_kx, i_theta, i_t, :, :] = get_unitary_evolution(kx, ky, theta, time, constant_sigma_y=0.,J_factor=1.)
            phases[i_kx, i_theta, i_t, :], states[i_kx, i_theta, i_t, :, :] = eig(u[i_kx, i_theta, i_t, :, :])
            # sort the eigenvalues and eigenvectors by the phase of the eigenvalues
            angles[i_kx, i_theta, i_t, :] = np.angle(phases[i_kx, i_theta, i_t, :]).astype(float)
            idx = np.argsort(-angles[i_kx, i_theta, i_t, :])
            angles[i_kx, i_theta, i_t, :] = angles[i_kx, i_theta, i_t, idx]
            states[i_kx, i_theta, i_t, :, :] = states[i_kx, i_theta, i_t, :, idx].T




# phases = np.angle(phases)

top_band_phases = np.abs(angles.max(axis=-1))
topological_singularities_pi = top_band_phases > 3.1415
topological_singularities_0 = top_band_phases < 0.0001
topological_singularities_0[:,:,0] = False
top_band_states = states[:, :, :, :, 0]

plt.plot(top_band_phases.reshape(len(kx_list)*len(theta_list), len(times)).T)

def make_state_continuous(state_vs_kx_theta, reps=5):
    for rep in range(reps):
        for i_kx, kx in enumerate(kx_list):
            for i_theta, theta in enumerate(theta_list):
                if i_kx == 0 and i_theta == 0:
                    continue
                if i_kx == 0:
                    reference_state = state_vs_kx_theta[i_kx, i_theta - 1, :]
                if i_theta == 0:
                    if rep == 0:
                        reference_state = state_vs_kx_theta[i_kx - 1, i_theta, :]
                    else:
                        reference_state = 1 / 2 * (
                                    state_vs_kx_theta[i_kx - 1, i_theta, :] + state_vs_kx_theta[i_kx, i_theta - 1, :])
                if i_kx != 0 and i_theta != 0:
                    reference_state = 1 / 2 * (
                                state_vs_kx_theta[i_kx - 1, i_theta, :] + state_vs_kx_theta[i_kx, i_theta - 1, :])
                D = np.dot(state_vs_kx_theta[i_kx, i_theta, :].conj(), reference_state)
                D = D / np.abs(D)
                state_vs_kx_theta[i_kx, i_theta, :] = state_vs_kx_theta[i_kx, i_theta, :] * D
    return state_vs_kx_theta


def calculate_winding_number(state_vs_theta):
    phase_vs_theta = np.angle(state_vs_theta[0,:].conj() @ state_vs_theta.T)
    plt.figure()
    plt.plot(theta_list, phase_vs_theta)


def get_topological_invariant(state_vs_kx_theta, reps=5):
    state_vs_kx_theta = make_state_continuous(state_vs_kx_theta, reps=reps)
    # check for continuity
    plt.figure()
    plt.imshow(
        np.linalg.norm(np.diff(state_vs_kx_theta, axis=0), axis=-1))
    plt.colorbar()
    plt.figure()
    plt.imshow(
        np.linalg.norm(np.diff(np.concatenate([state_vs_kx_theta, state_vs_kx_theta[:,:10,:]], axis=1), axis=1), axis=-1))
    plt.colorbar()

    # calculate the winding number at kx=0 and kx=pi
    for i_kx in [0,-1]:
        calculate_winding_number(state_vs_kx_theta[i_kx, :, :])



# iterate over all times and set a continuous gauge for the states at each time in the k_x,theta plane

for i_t in range(nsteps):
    for i_kx, kx in enumerate(kx_list):
        for i_theta, theta in enumerate(theta_list):
            if i_kx == 0 and i_theta == 0:
                continue
            if i_kx == 0:
                reference_states = states[i_kx, i_theta - 1, i_t, :, :]
            if i_theta == 0:
                reference_states = states[i_kx - 1, i_theta, i_t, :, :]
            if i_kx != 0 and i_theta != 0:
                reference_states = 1 / 2 * (states[i_kx - 1, i_theta, i_t, :, :] + states[i_kx, i_theta - 1, i_t, :, :])
            D = states[i_kx, i_theta, i_t, :, :].T.conj() @ reference_states
            D = np.diag(np.diag(D) / np.abs(np.diag(D)))
            states[i_kx, i_theta, i_t, :, :] = states[i_kx, i_theta, i_t, :, :] @ D

for rep in range(5):
    for i_t in range(nsteps):
        for i_kx, kx in enumerate(kx_list):
            for i_theta, theta in enumerate(theta_list):
                if i_kx == 0 and i_theta == 0:
                    continue
                if i_kx == 0:
                    reference_states = states[i_kx, i_theta - 1, i_t, :, :]
                if i_theta == 0:
                    reference_states = 1 / 2 * (states[i_kx - 1, i_theta, i_t, :, :] + states[i_kx, i_theta - 1, i_t, :, :])#states[i_kx - 1, i_theta, i_t, :, :]
                if i_kx != 0 and i_theta != 0:
                    reference_states = 1 / 2 * (states[i_kx - 1, i_theta, i_t, :, :] + states[i_kx, i_theta - 1, i_t, :, :])
                D = states[i_kx, i_theta, i_t, :, :].T.conj() @ reference_states
                D = np.diag(np.diag(D)/np.abs(np.diag(D)))
                states[i_kx, i_theta, i_t, :, :] = states[i_kx, i_theta, i_t, :, :] @ D



# for each topological singularity, find the time slice where it occurs
# take the states at the time slice before and after the singularity
# at kx=0,pi diagonalize the states for each theta
# plot the eigenvalues of the diagonalized states as a function of theta

# loop over time slices and check for topological singularities
for i_t in range(nsteps):
    if topological_singularities_pi[:, :, i_t].any():
        print(i_t)
        # diagonalize the states at the time slice before and after the singularity as a function of theta for kx=0,pi
        for ikx in [0, len(kx_list)-1]:
            it_before = i_t - 1
            it_after = i_t + 1
            phases_before = np.zeros((len(theta_list), 2))
            phases_after = np.zeros((len(theta_list), 2))
            for i_theta, theta in enumerate(theta_list):
                phases_before[i_theta, :] = np.angle(eig(states[ikx, i_theta, 0, :, :])[0])
                phases_after[i_theta, :] = np.angle(eig(states[ikx, i_theta, -1, :, :])[0])
            plt.plot(theta_list, phases_before, 'o')
            plt.plot(theta_list, phases_after, 'o')
            plt.show()



plt.plot(top_band_phases.reshape(-1,nsteps).T)
plt.show()

