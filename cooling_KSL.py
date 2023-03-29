import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from one_d_ising import get_smoothed_func, get_g, get_B
from translational_invariant_KSL import get_KSL_model

g0 = 0.5
B1 = 0.
B0 = 3.

Delta = 0.5
Jx = 0.5
Jy = 0.5
Jz = 0.5

n_k_points = 1+6*2


def get_f(kx, ky):
    return -Jx-Jy*np.exp(-1j*kx)-Jz*np.exp(-1j*ky)


for T in [50]:#np.linspace(5,50,19):
    t1 = T / 4

    smoothed_g_before_zeroing = lambda t: get_smoothed_func(t, lambda tt: get_g(tt, g0, T, t1), T / 10)
    smoothed_B_before_zeroing = lambda t: get_smoothed_func(t, lambda tt: get_B(tt, B0, B1, T, t1), T / 10)
    smoothed_g = lambda t: smoothed_g_before_zeroing(t) - smoothed_g_before_zeroing(T)
    smoothed_B = lambda t: smoothed_B_before_zeroing(t) - smoothed_B_before_zeroing(T)

    trotter_steps = 100

    cycles = 50
    r_tol = 1e-5

    kx_list = np.linspace(-np.pi, np.pi, n_k_points)
    ky_list = np.linspace(-np.pi, np.pi, n_k_points)
    # kx_list = [-2/3*np.pi]
    # ky_list = [2/3*np.pi]

    for num_cooling_sublattices in [1,2]:

        E_diff = np.zeros((len(kx_list), len(ky_list)))

        for i_kx, kx in enumerate(kx_list):
            for i_ky, ky in enumerate(ky_list):

                f = get_f(kx, ky)
                f_real = np.real(f)
                f_imag = np.imag(f)

                hamiltonian, S, E_gs = \
                    get_KSL_model(f_real=f_real, f_imag=f_imag, Delta=Delta, g=smoothed_g, B=smoothed_B, initial_state='random', num_cooling_sublattices=num_cooling_sublattices)
                Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
                Es = []
                cycle = 0
                Es.append(S.get_energy(hamiltonian.get_matrix(T)))
                while True:
                    if cycle >= cycles and abs((Es[-1] - E_gs)/(Es[-2] - E_gs) - 1) < r_tol:
                        # finished all cycles
                        break
                    else:
                        # print('apply a full cycle unitary')
                        S.evolve_with_unitary(Ud)

                        # print('reset')
                        S.reset_all_tau()
                        Es.append(S.get_energy(hamiltonian.get_matrix(T)))
                        cycle += 1
                        # print(cycle)

                print(f'kx={kx}, ky={ky}')
                print(Es[-1])
                print('ground state energy = ' + str(E_gs))
                E_diff[i_kx, i_ky] = Es[-1] - E_gs

                # plt.figure()
                # plt.plot(Es)
                # plt.plot([E_gs] * len(Es))
                # plt.show()



        # results_df = pd.DataFrame({'Jx': Jx, 'Jy': Jy, 'Jz': Jz, 'Delta': Delta, 'Nt': trotter_steps, 'n_k_points': n_k_points,
        #                            'T': T, 'num_cooling_sublattices': num_cooling_sublattices,
        #                            'energy_density': np.mean(E_diff)}, index=[0])
        #
        # with open("KSL.csv", 'a') as f:
        #     results_df.to_csv(f, mode='a', header=f.tell()==0, index=False)

        plt.imshow(E_diff)
        plt.colorbar()
        plt.xlabel('Real(f)')
        plt.ylabel('Imag(f)')
        plt.show()