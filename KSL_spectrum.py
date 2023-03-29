import numpy as np
from matplotlib import pyplot as plt
from translational_invariant_KSL import get_KSL_model
from one_d_ising import get_smoothed_func, get_g, get_B


B0 = 3.
B1 = 0.
g0 = 0.5

T = 30.
t1 = T / 4

smoothed_g_before_zeroing = lambda t: get_smoothed_func(t, lambda tt: get_g(tt, g0, T, t1), T / 10)
smoothed_B_before_zeroing = lambda t: get_smoothed_func(t, lambda tt: get_B(tt, B0, B1, T, t1), T / 10)
smoothed_g = lambda t: smoothed_g_before_zeroing(t) - smoothed_g_before_zeroing(T)
smoothed_B = lambda t: smoothed_B_before_zeroing(t) - smoothed_B_before_zeroing(T)

prop_cycle = plt.rcParams['axes.prop_cycle']

t_list = np.linspace(0,T,100)
spectrum = np.zeros((len(t_list),12))
for it, t in enumerate(t_list):
    Delta = 0.5

    Jx = 1
    Jy = 1
    Jz = 1

    n_k_points = 1+6*2


    def get_f(kx, ky):
        return -Jx-Jy*np.exp(-1j*kx)-Jz*np.exp(-1j*ky)


    kx = 0#-2/3*np.pi
    ky = 0#2/3*np.pi

    num_cooling_sublattices = 1

    f = get_f(kx, ky)
    f_real = np.real(f)
    f_imag = np.imag(f)

    hamiltonian, S, E_gs = \
        get_KSL_model(f_real=f_real, f_imag=f_imag, Delta=Delta, g=smoothed_g, B=smoothed_B, initial_state='random', num_cooling_sublattices=num_cooling_sublattices)

    spectrum[it,:] = hamiltonian.get_excitation_spectrum(t)
plt.plot(t_list, spectrum, 'b')
plt.show()