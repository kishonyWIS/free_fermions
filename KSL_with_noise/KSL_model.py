import numpy as np
from free_fermion_hamiltonian import MajoranaFreeFermionHamiltonian, MajoranaSingleParticleDensityMatrix
from matplotlib import pyplot as plt
import pandas as pd
from time_dependence_functions import get_g, get_B
np.random.seed(0)

def get_KSL_model(num_sites_x, num_sites_y, J, kappa, g, B, initial_state):
    # sublattices are numbered c_0, b^1_0, b^2_0, b^3_0, b^4_0, b^5_0, c_1, b^1_1, b^2_1, b^3_1, b^4_1, b^5_1
    num_sublattices = 12
    # lattice_shape = (num_sites_x, num_sites_y, num_sublattices)
    #
    # system_shape = (num_sites, num_sublattices, num_sites, num_sublattices)
    # non_gauge_shape = (num_sites, 4, num_sites, 4)
    # gauge_shape = (num_sites, 2, num_sites, 2)
    # non_gauge_idxs = np.ix_(range(num_sites), [0, 3, 4, 5], range(num_sites), [0, 3, 4, 5])
    # gauge_idxs = np.ix_(range(num_sites), [1, 2], range(num_sites), [1, 2])

    lattice_shape = (num_sites_x, num_sites_y, num_sublattices)
    system_shape = lattice_shape*2

    hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)

    site_offset_x = (0, 0)
    site_offset_y = (1, 0)
    site_offset_z = (0, 1)

    site_offset_kappa_x = (1, 0)
    site_offset_kappa_y = (0, 1)
    site_offset_kappa_z = (1, -1)


    add_term_with_offset(name='Jx', strength=-J, hamiltonian=hamiltonian, site_offset=site_offset_x, sublattice1=0, sublattice2=3, lattice_shape=lattice_shape)
    add_term_with_offset(name='Jy', strength=J, hamiltonian=hamiltonian, site_offset=site_offset_y, sublattice1=3, sublattice2=0, lattice_shape=lattice_shape)
    add_term_with_offset(name='Jz', strength=J, hamiltonian=hamiltonian, site_offset=site_offset_z, sublattice1=3, sublattice2=0, lattice_shape=lattice_shape)
    add_term_with_offset(name='kappa_x_0', strength=kappa, hamiltonian=hamiltonian, site_offset=site_offset_kappa_x, sublattice1=0, sublattice2=0, lattice_shape=lattice_shape)
    add_term_with_offset(name='kappa_x_1', strength=-kappa, hamiltonian=hamiltonian, site_offset=site_offset_kappa_x, sublattice1=3, sublattice2=3, lattice_shape=lattice_shape)
    add_term_with_offset(name='kappa_y_0', strength=-kappa, hamiltonian=hamiltonian, site_offset=site_offset_kappa_y, sublattice1=0, sublattice2=0, lattice_shape=lattice_shape)
    add_term_with_offset(name='kappa_y_1', strength=kappa, hamiltonian=hamiltonian, site_offset=site_offset_kappa_y, sublattice1=3, sublattice2=3, lattice_shape=lattice_shape)
    add_term_with_offset(name='kappa_z_0', strength=-kappa, hamiltonian=hamiltonian, site_offset=site_offset_kappa_z, sublattice1=0, sublattice2=0, lattice_shape=lattice_shape)
    add_term_with_offset(name='kappa_z_1', strength=kappa, hamiltonian=hamiltonian, site_offset=site_offset_kappa_z, sublattice1=3, sublattice2=3, lattice_shape=lattice_shape)

    add_term_with_offset(name='g_0', strength=-1, hamiltonian=hamiltonian, site_offset=(0, 0), sublattice1=1, sublattice2=0, lattice_shape=lattice_shape, time_dependence=g)
    add_term_with_offset(name='g_1', strength=-1, hamiltonian=hamiltonian, site_offset=(0, 0), sublattice1=4, sublattice2=3, lattice_shape=lattice_shape, time_dependence=g)
    add_term_with_offset(name='B_0', strength=-1, hamiltonian=hamiltonian, site_offset=(0, 0), sublattice1=1, sublattice2=2, lattice_shape=lattice_shape, time_dependence=B)
    add_term_with_offset(name='B_1', strength=-1, hamiltonian=hamiltonian, site_offset=(0, 0), sublattice1=4, sublattice2=5, lattice_shape=lattice_shape, time_dependence=B)

    if initial_state == 'random':
        S = KSLState(system_shape)
        S.randomize()
        S.reset_all_tau()

    return hamiltonian, S

def add_term_with_offset(name, strength, hamiltonian, site_offset, sublattice1, sublattice2, lattice_shape, time_dependence=None):
    site_offset = np.array(site_offset)
    first_site1 = np.abs(site_offset)*(site_offset<0)
    for site in np.ndindex(tuple(lattice_shape[:-1] - np.abs(site_offset))):
        site1 = tuple(np.array(site) + first_site1)
        site2 = tuple(np.array(site1) + np.array(site_offset))
        strength_on_site = strength[site1] if isinstance(strength, np.ndarray) else strength
        hamiltonian.add_term(name=f'{name}_{site1}', strength=strength_on_site, sublattice1=sublattice1, sublattice2=sublattice2, site1=site1, site2=site2,
                             time_dependence=time_dependence)

class KSLState(MajoranaSingleParticleDensityMatrix):
    def reset_all_tau(self):
        self.system_shape[:2]
        for site in np.ndindex(tuple(self.system_shape[:2])):
            self.reset(4, 5, site, site)
            self.reset(1, 2, site, site)



if __name__ == '__main__':
    num_sites_x = 2
    num_sites_y = 2
    g0 = 0.5
    B1 = 0.
    B0 = 5.
    J = 1.
    kappa = 0.1

    cycles = 50

    trotter_steps = 10000

    T = 100.
    t1 = T / 4
    smoothed_g = lambda t: get_g(t, g0, T, t1)
    smoothed_B = lambda t: get_B(t, B0, B1, T)

    hamiltonian, S = get_KSL_model(num_sites_x, num_sites_y, J, kappa, smoothed_g, smoothed_B, initial_state='random')
    E_gs = hamiltonian.get_ground_state(T).get_energy(hamiltonian.get_matrix(T))
    print(E_gs)
    # Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
    integration_params = dict(name='vode', nsteps=50000, rtol=1e-12, atol=1e-16)
    Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
    print()

    Es = []
    Es.append(S.get_energy(hamiltonian.get_matrix(T)))
    S.evolve_with_unitary(Ud)
    Es.append(S.get_energy(hamiltonian.get_matrix(T)))
    print(Es)


    # S = hamiltonian.
    #
    #     hamiltonian, S, decoupled_hamiltonian_with_gauge, E_gs, all_errors_unitaries, errors_effect_gauge = \
    #         get_TFI_model(num_sites, h, J, smoothed_g, smoothed_B, initial_state='random', periodic_bc=periodic_bc)
    #     Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
    #     # Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
    #     average_Es = []
    #     for error_rate in errors_per_cycle_per_qubit:
    #         Es = []
    #         cycle = 0
    #         time_to_error = np.random.exponential(T / (error_rate * num_sites * 2))
    #         time_in_current_cycle = 0.
    #
    #         Es.append(S.get_energy(decoupled_hamiltonian_with_gauge.get_matrix()))
    #         new_row = pd.DataFrame(
    #             {'Ns': num_sites, 'periodic_bc': periodic_bc, 'drop_one_g_for_odd_bath_signs': False, 'J': J,
    #              'h': h, 'V': 0, 'T': T, 'Nt': trotter_steps, 'N_iter': cycle,
    #              'errors_per_cycle_per_qubit': error_rate, 'energy_density': (Es[-1] - E_gs) / num_sites},
    #             index=[0])
    #         results_df = pd.concat([results_df, new_row], ignore_index=True)
    #
    #
    #         while True:
    #             if cycle == cycles:
    #                 # finished all cycles
    #                 break
    #             elif time_to_error == 0:
    #                 # print('apply error')
    #                 error_name = np.random.choice(list(all_errors_unitaries.keys()))
    #                 S.evolve_with_unitary(all_errors_unitaries[error_name])
    #                 # print(error_name)
    #                 if errors_effect_gauge[error_name]:
    #                     # print('recalculating Ud')
    #                     # Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
    #                     Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
    #                 time_to_error = np.random.exponential(T / (error_rate * num_sites * 2))
    #             elif time_to_error > T and time_in_current_cycle == 0:
    #                 # print('apply a full cycle unitary')
    #                 S.evolve_with_unitary(Ud)
    #                 time_to_error -= T
    #                 time_in_current_cycle = T
    #             elif time_to_error < T - time_in_current_cycle:
    #                 # print('apply a partial unitary until error')
    #                 # Ud_temp = hamiltonian.full_cycle_unitary_faster(integration_params,
    #                 #                                                 time_in_current_cycle,
    #                 #                                                 time_in_current_cycle + time_to_error)
    #                 steps = int(trotter_steps * time_to_error / T)
    #                 if steps > 0:
    #                     Ud_temp = hamiltonian.full_cycle_unitary_trotterize(time_in_current_cycle,
    #                                                                         time_in_current_cycle + time_to_error, steps=steps)
    #                     S.evolve_with_unitary(Ud_temp)
    #                 time_in_current_cycle += time_to_error
    #                 time_to_error = 0
    #             elif time_in_current_cycle == T:
    #                 # print('reset')
    #                 S.reset_all_tau()
    #                 cycle += 1
    #                 print(cycle)
    #                 Es.append(S.get_energy(decoupled_hamiltonian_with_gauge.get_matrix()))
    #                 time_in_current_cycle = 0
    #                 new_row = pd.DataFrame(
    #                     {'Ns': num_sites, 'periodic_bc': periodic_bc, 'drop_one_g_for_odd_bath_signs': False, 'J': J,
    #                      'h': h, 'V': 0, 'T': T, 'Nt': trotter_steps, 'N_iter': cycle,
    #                      'errors_per_cycle_per_qubit': error_rate, 'energy_density': (Es[-1] - E_gs) / num_sites},
    #                     index=[0])
    #                 results_df = pd.concat([results_df, new_row], ignore_index=True)
    #             elif time_in_current_cycle > 0:
    #                 # print('finish incomplete cycle')
    #                 # Ud_temp = hamiltonian.full_cycle_unitary_faster(integration_params,
    #                 #                                                 time_in_current_cycle,
    #                 #                                                 T)
    #                 steps = int(trotter_steps * (T - time_in_current_cycle) / T)
    #                 if steps > 0:
    #                     Ud_temp = hamiltonian.full_cycle_unitary_trotterize(time_in_current_cycle, T, steps=steps)
    #                     S.evolve_with_unitary(Ud_temp)
    #                 time_to_error -= T - time_in_current_cycle
    #                 time_in_current_cycle = T
    #             else:
    #                 raise 'invalid cycle state'
    #
    #         print(Es[-1])
    #         print('ground state energy = ' + str(E_gs))
    #         plt.figure(i_h_J)
    #         plt.semilogy(np.arange(len(Es))+1, Es-E_gs)
    #
    #
    #
