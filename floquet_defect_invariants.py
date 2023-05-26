import numpy as np
from scipy.integrate import complex_ode


def get_J_pulse(t, pulse_length=1/3, delay=0):
# periodic function with period 1 applying a plus between t=0 and t=pulse_length
    return int(t - delay - np.floor(t - delay) < pulse_length)/pulse_length * np.pi/2


def get_J_cos(t, delay=0):
    return 1 + np.cos(2*np.pi*(t - delay))


def get_Hamiltonian_vortex(kx, ky, theta, t):
    J_x = get_J_pulse(t, pulse_length=1/3, delay=0 + theta/(2 * np.pi))
    J_y = get_J_pulse(t, pulse_length=1/3, delay=1/3 + theta/(2 * np.pi))
    J_z = get_J_pulse(t, pulse_length=1/3, delay=2/3 + theta/(2 * np.pi))
    tau_x_coef = J_y * np.sin(kx) + J_z * np.sin(ky)
    tau_y_coef = J_x + J_y * np.cos(kx) + J_z * np.cos(ky)
    tau_x = np.array([[0, 1], [1, 0]])
    tau_y = np.array([[0, -1j], [1j, 0]])
    return tau_x_coef * tau_x + tau_y_coef * tau_y


def get_Hamiltonian_lambda(kx, ky, theta, lamb, t):
    # TODO: add Hc - proportional to tau_z?
    Hc = np.array([[0,0],[0,0]])
    return get_Hamiltonian_vortex(kx, ky, 0, t) * lamb + get_Hamiltonian_vortex(kx, ky, theta, t) * (1 - lamb) + lamb * (1 - lamb) * Hc


def get_du_dt(kx, ky, theta, lamb, t, u):
    M = get_Hamiltonian_lambda(kx, ky, theta, lamb, t)
    return -1j * (M @ u.reshape(M.shape[1], -1)).reshape(-1)


def get_unitary(kx, ky, theta, lamb, integration_params, t0=0, tf=1):
    ode_instance = complex_ode(lambda t, u: get_du_dt(kx, ky, theta, lamb, t, u))
    ode_instance.set_integrator(**integration_params)
    u0 = np.eye(2).reshape(-1)
    ode_instance.set_initial_value(u0, t0)
    ode_instance.integrate(tf)
    return ode_instance.y.reshape(2,2)
