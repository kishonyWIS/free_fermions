import numpy as np
from scipy.linalg import expm
from matplotlib import pyplot as plt
from scipy.linalg import eig

SIGMA_X = np.array([[0,1],[1,0]])
SIGMA_Y = np.array([[0,-1j],[1j,0]])
SIGMA_Z = np.array([[1,0],[0,-1]])


def get_ux(kx,ky,duration,constant_sigma_y=0.,J_factor=1.):
    strength = duration*3*np.pi/2 * J_factor
    # return expm(-1j * strength * SIGMA_Y * (1+constant_sigma_y))
    return np.cos(strength) * np.eye(2) - 1j * np.sin(strength) * SIGMA_Y


def get_uy(kx,ky,duration,constant_sigma_y=0.,J_factor=1.):
    strength = duration*3*np.pi/2 * J_factor
    # return expm(-1j * strength * ((np.cos(kx)+constant_sigma_y)*SIGMA_Y + np.sin(kx)*SIGMA_X))
    return np.cos(strength) * np.eye(2) - 1j * np.sin(strength) * (np.cos(kx)*SIGMA_Y + np.sin(kx)*SIGMA_X)


def get_uz(kx,ky,duration,constant_sigma_y=0.,J_factor=1.):
    strength = duration*3*np.pi/2 * J_factor
    # return expm(-1j * strength * ((np.cos(ky)+constant_sigma_y)*SIGMA_Y + np.sin(ky)*SIGMA_X))
    return np.cos(strength) * np.eye(2) - 1j * np.sin(strength) * (np.cos(ky)*SIGMA_Y + np.sin(ky)*SIGMA_X)


def get_duration(start_time, end_time, delay, t):
    combined_start_time = max(start_time, delay)
    combined_end_time = min(end_time, delay + t)
    duration = combined_end_time - combined_start_time
    if duration < 0:
        return 0.
    return duration


def get_unitary_evolution(kx,ky,theta,t,constant_sigma_y=0.,J_factor=1.):
    delay = theta/(2 * np.pi)
    pulse_end_points = np.linspace(0,2,7)
    unitary = np.eye(2, dtype=complex)
    for i in range(len(pulse_end_points)-1):
        start_time = pulse_end_points[i]
        end_time = pulse_end_points[i+1]
        duration = get_duration(start_time, end_time, delay, t)
        if duration == 0:
            continue
        if i % 3 == 0:
            unitary = get_ux(kx,ky,duration,constant_sigma_y=constant_sigma_y,J_factor=J_factor) @ unitary
        elif i % 3 == 1:
            unitary = get_uy(kx,ky,duration,constant_sigma_y=constant_sigma_y,J_factor=J_factor) @ unitary
        elif i % 3 == 2:
            unitary = get_uz(kx,ky,duration,constant_sigma_y=constant_sigma_y,J_factor=J_factor) @ unitary
    return unitary


def diagonalize_unitary_at_k_theta_time(kx, ky, theta, time):
    unitary = get_unitary_evolution(kx, ky, theta, time, constant_sigma_y=0., J_factor=1.)
    phases, states = eig(unitary)
    # sort the eigenvalues and eigenvectors by the phase of the eigenvalues
    angles = np.angle(phases).astype(float)
    idx = np.argsort(-angles)
    angles = angles[idx]
    states = states[:, idx]
    return angles, states


def make_state_continuous(state_vs_kx_theta, reps=5):
    for rep in range(reps):
        for i_kx in range(state_vs_kx_theta.shape[0]):
            for i_theta in range(state_vs_kx_theta.shape[1]):
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
    plt.plot(phase_vs_theta)
    plt.xlabel('theta')
    plt.ylabel('phase of eigenstate')


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
    for i_kx, kx in zip([0,-1], [0,np.pi]):
        calculate_winding_number(state_vs_kx_theta[i_kx, :, :])
        plt.title('kx = {}'.format(kx))
