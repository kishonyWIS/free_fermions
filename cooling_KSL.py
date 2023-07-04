import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from time_dependence_functions import get_g, get_B
from translational_invariant_KSL import get_KSL_model, get_Delta, get_f

g0 = 0.5
B1 = 0.
B0 = 5.

kappa = 0.1
Jx = 1.
Jy = 1.
Jz = 1.

n_k_points = 1+6*2


integration_params = dict(name='vode', nsteps=2000, rtol=1e-6, atol=1e-10)


for T in np.linspace(10,90,17):
    t1 = T / 4

    smoothed_g = lambda t: get_g(t, g0, T, t1)
    smoothed_B = lambda t: get_B(t, B0, B1, T)

    cycles = 50
    r_tol = 1e-4

    kx_list = np.linspace(-np.pi, np.pi, n_k_points)
    ky_list = np.linspace(-np.pi, np.pi, n_k_points)
    # kx_list = [-2/3*np.pi]
    # ky_list = [2/3*np.pi]

    for num_cooling_sublattices in [2]:

        E_diff = np.zeros((len(kx_list), len(ky_list)))

        for i_kx, kx in enumerate(kx_list):
            for i_ky, ky in enumerate(ky_list):

                f = get_f(kx, ky, Jx, Jy, Jz)
                Delta = get_Delta(kx, ky, kappa)

                hamiltonian, S, E_gs = \
                    get_KSL_model(f=f, Delta=Delta, g=smoothed_g, B=smoothed_B, initial_state='random', num_cooling_sublattices=num_cooling_sublattices)
                Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
                Es = []
                cycle = 0
                Es.append(S.get_energy(hamiltonian.get_matrix(T)))
                while True:
                    if cycle >= cycles and abs((Es[-1] - Es[-2])/(Es[-2] - E_gs)) < r_tol:
                        # finished all cycles
                        break
                    else:
                        # print('apply a full cycle unitary')
                        S.evolve_with_unitary(Ud)

                        # print('reset')
                        S.reset_all_tau()
                        Es.append(S.get_energy(hamiltonian.get_matrix(T)))
                        cycle += 1
                        print(cycle)

                print(f'kx={kx}, ky={ky}')
                print(Es[-1])
                print('ground state energy = ' + str(E_gs))
                E_diff[i_kx, i_ky] = Es[-1] - E_gs

                # plt.figure()
                # plt.plot(Es)
                # plt.plot([E_gs] * len(Es))
                # plt.show()



        results_df = pd.DataFrame({'Jx': Jx, 'Jy': Jy, 'Jz': Jz, 'kappa': kappa, 'B0': B0, 'g0': g0,
                                   'n_k_points': n_k_points, 'T': T, 'num_cooling_sublattices': num_cooling_sublattices,
                                   'energy_density': np.mean(E_diff)}, index=[0])

        with open("KSL_complex.csv", 'a') as f:
            results_df.to_csv(f, mode='a', header=f.tell()==0, index=False)

        # plt.imshow(E_diff)
        # plt.colorbar()
        # plt.xlabel('Real(f)')
        # plt.ylabel('Imag(f)')
        # plt.show()