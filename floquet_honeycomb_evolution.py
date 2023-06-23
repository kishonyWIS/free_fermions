import numpy as np

SIGMA_X = np.array([[0,1],[1,0]])
SIGMA_Y = np.array([[0,-1j],[1j,0]])
SIGMA_Z = np.array([[1,0],[0,-1]])


def get_ux(kx,ky,duration):
    strength = duration*3*np.pi/2
    return np.cos(strength) * np.eye(2) - 1j * np.sin(strength) * SIGMA_Y


def get_uy(kx,ky,duration):
    strength = duration*3*np.pi/2
    return np.cos(strength) * np.eye(2) - 1j * np.sin(strength) * (np.cos(kx)*SIGMA_Y + np.sin(kx)*SIGMA_X)


def get_uz(kx,ky,duration):
    strength = duration*3*np.pi/2
    return np.cos(strength) * np.eye(2) - 1j * np.sin(strength) * (np.cos(ky)*SIGMA_Y + np.sin(ky)*SIGMA_X)


def get_duration(start_time, end_time, delay, t):
    combined_start_time = max(start_time, delay)
    combined_end_time = min(end_time, delay + t)
    duration = combined_end_time - combined_start_time
    if duration < 0:
        return 0.
    return duration


def get_unitary_evolution(kx,ky,theta,t):
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
            unitary = get_ux(kx,ky,duration) @ unitary
        elif i % 3 == 1:
            unitary = get_uy(kx,ky,duration) @ unitary
        elif i % 3 == 2:
            unitary = get_uz(kx,ky,duration) @ unitary
    return unitary