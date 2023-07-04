import numpy as np
from scipy.linalg import expm

SIGMA_X = np.array([[0,1],[1,0]])
SIGMA_Y = np.array([[0,-1j],[1j,0]])
SIGMA_Z = np.array([[1,0],[0,-1]])


def get_ux(kx,ky,duration,constant_sigma_y=0.,J_factor=1.):
    strength = duration*3*np.pi/2 * J_factor
    return expm(-1j * strength * SIGMA_Y * (1+constant_sigma_y))
    return np.cos(strength) * np.eye(2) - 1j * np.sin(strength) * SIGMA_Y


def get_uy(kx,ky,duration,constant_sigma_y=0.,J_factor=1.):
    strength = duration*3*np.pi/2 * J_factor
    return expm(-1j * strength * ((np.cos(kx)+constant_sigma_y)*SIGMA_Y + np.sin(kx)*SIGMA_X))
    return np.cos(strength) * np.eye(2) - 1j * np.sin(strength) * (np.cos(kx)*SIGMA_Y + np.sin(kx)*SIGMA_X)


def get_uz(kx,ky,duration,constant_sigma_y=0.,J_factor=1.):
    strength = duration*3*np.pi/2 * J_factor
    return expm(-1j * strength * ((np.cos(ky)+constant_sigma_y)*SIGMA_Y + np.sin(ky)*SIGMA_X))
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
