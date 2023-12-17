from itertools import product

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from time_dependence_functions import get_g, get_B
from translational_invariant_KSL import get_KSL_model, get_Delta, get_f
from scipy.linalg import expm

g0 = 0.5
B1 = 0.
B0 = 7.

kappa = 1.
Jx = 1.
Jy = 1.
Jz = 1.

integration_params = dict(name='vode', nsteps=6000, rtol=1e-6, atol=1e-10)


cycles = 1


def get_chern_number_from_single_particle_dm(single_particle_dm):
    dP_dkx = np.diff(single_particle_dm, axis=0)[:,:-1,:,:]
    dP_dky = np.diff(single_particle_dm, axis=1)[:-1,:,:,:]
    P = single_particle_dm[:-1,:-1,:,:]
    integrand = np.zeros(P.shape[0:2],dtype=complex)
    for i_kx, i_ky in product(range(P.shape[0]), repeat=2):
        integrand[i_kx,i_ky] = np.trace(P[i_kx,i_ky,:,:] @ (dP_dkx[i_kx,i_ky,:,:] @ dP_dky[i_kx,i_ky,:,:] - dP_dky[i_kx,i_ky,:,:] @ dP_dkx[i_kx,i_ky,:,:]))
    return (np.sum(integrand)/(2*np.pi)).imag


for n_k_points in [1+6*nn for nn in [14]]:

    kx_list = np.linspace(-np.pi, np.pi, n_k_points)
    ky_list = np.linspace(-np.pi, np.pi, n_k_points)


    for T in [10]:#10,30,50,70,90
        t1 = T / 4

        smoothed_g = lambda t: get_g(t, g0, T, t1)
        smoothed_B = lambda t: get_B(t, B0, B1, T)

        num_cooling_sublattices = 2

        E_diff = np.zeros((len(kx_list), len(ky_list)))

        single_particle_dm = np.zeros((n_k_points, n_k_points, 6, 6), dtype=complex)

        # t_list = np.linspace(0, T, 100)
        # spectrum = np.zeros((len(t_list), 6))

        for i_kx, kx in enumerate(kx_list):
            print(f'kx={kx}')
            for i_ky, ky in enumerate(ky_list):

                f = get_f(kx, ky, Jx, Jy, Jz)
                Delta = get_Delta(kx, ky, kappa)

                hamiltonian, S, E_gs = \
                    get_KSL_model(f=f, Delta=Delta, g=smoothed_g, B=smoothed_B, initial_state='product', num_cooling_sublattices=num_cooling_sublattices)
                # for i_t, t in enumerate(t_list):
                #     spectrum[i_t,:] = 2*hamiltonian.get_excitation_spectrum(t)
                # plt.figure()
                # plt.plot(t_list, spectrum)

                Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)

                Es = []
                cycle = 0
                Es.append(S.get_energy(hamiltonian.get_matrix(T)))
                while True:
                    if cycle >= cycles:
                        # finished all cycles
                        break
                    else:
                        # print('apply a full cycle unitary')
                        S.evolve_with_unitary(Ud)

                        # print('reset')
                        # S.reset_all_tau()
                        Es.append(S.get_energy(hamiltonian.get_matrix(T)))
                        cycle += 1
                        # print(cycle)

                # print(f'kx={kx}, ky={ky}')
                # print(Es[-1])
                # print('ground state energy = ' + str(E_gs))
                E_diff[i_kx, i_ky] = Es[-1] - E_gs
                single_particle_dm[i_kx,i_ky,:,:] = S.matrix


        total_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm)
        system_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,:2,:2])
        bath_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,2:,2:])
        # print(f'total chern number = {total_chern_number}')
        # print(f'system chern number = {system_chern_number}')
        # print(f'bath chern number = {bath_chern_number}')


        results_df = pd.DataFrame({'Jx': Jx, 'Jy': Jy, 'Jz': Jz, 'kappa': kappa, 'B0': B0, 'g0': g0,
                                   'n_k_points': n_k_points, 'T': T, 'num_cooling_sublattices': num_cooling_sublattices,
                                   'energy_density': np.mean(E_diff)/2, 'total_chern_number': total_chern_number,
                                   'system_chern_number': system_chern_number, 'bath_chern_number': bath_chern_number},
                                  index=[0])
        # np.mean(E_diff)/2 because we count k and -k together.

        with open("KSL_complex_chern.csv", 'a') as f:
            results_df.to_csv(f, mode='a', header=f.tell()==0, index=False)
