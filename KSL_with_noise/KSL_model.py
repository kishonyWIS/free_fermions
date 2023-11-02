import numpy as np
from free_fermion_hamiltonian import MajoranaFreeFermionHamiltonian, MajoranaSingleParticleDensityMatrix
from matplotlib import pyplot as plt
import pandas as pd
from time_dependence_functions import get_g, get_B
np.random.seed(0)

def get_KSL_model(num_sites_x, num_sites_y, J, kappa, g, B, initial_state):
    # sublattices are numbered c_0, b^1_0, b^2_0, b^3_0, b^4_0, b^5_0, c_1, b^1_1, b^2_1, b^3_1, b^4_1, b^5_1
    gauge_sublattices = [1, 2, 3, 7, 8, 9]
    non_gauge_sublattices = [0, 4, 5, 6, 10, 11]

    num_gauge_sublattices = len(gauge_sublattices)
    num_non_gauge_sublattices = len(non_gauge_sublattices)
    num_sublattices = num_gauge_sublattices + num_non_gauge_sublattices

    gauge_lattice_shape = (num_sites_x, num_sites_y, num_gauge_sublattices)
    non_gauge_lattice_shape = (num_sites_x, num_sites_y, num_non_gauge_sublattices)
    lattice_shape = (num_sites_x, num_sites_y, num_sublattices)

    gauge_system_shape = gauge_lattice_shape*2
    non_gauge_system_shape = non_gauge_lattice_shape*2
    system_shape = lattice_shape*2

    gauge_idxs = np.ix_(range(num_sites_x), range(num_sites_y), gauge_sublattices, range(num_sites_x), range(num_sites_y), gauge_sublattices)
    non_gauge_idxs = np.ix_(range(num_sites_x), range(num_sites_y), non_gauge_sublattices, range(num_sites_x), range(num_sites_y), non_gauge_sublattices)

    # here!!!!

    decoupled_hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)
    add_J_term(J, decoupled_hamiltonian, lattice_shape)
    add_kappa_term(decoupled_hamiltonian, kappa, lattice_shape)
    decoupled_hamiltonian_matrix = decoupled_hamiltonian.get_matrix()
    ground_state = decoupled_hamiltonian.get_ground_state()
    S0_tensor = np.zeros(system_shape)

    if initial_state == 'random':
        S_non_gauge = KSLState(non_gauge_system_shape)
        S_non_gauge.randomize()
        S0_tensor[non_gauge_idxs] = S_non_gauge.tensor
    elif initial_state == 'ground':
        S_non_gauge = decoupled_hamiltonian.get_ground_state()
        S0_tensor[non_gauge_idxs] = S_non_gauge.tensor[non_gauge_idxs]

    gauge_setting_hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)
    add_gauge_fixing_term(gauge_setting_hamiltonian, lattice_shape)

    S_gauge = gauge_setting_hamiltonian.get_ground_state()
    S0_tensor[gauge_idxs] = S_gauge.tensor[gauge_idxs]
    S = KSLState(system_shape=system_shape, tensor=S0_tensor)
    S.reset_all_tau()

    hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)
    add_J_term(J, hamiltonian, lattice_shape)
    add_kappa_term(kappa, hamiltonian, lattice_shape)
    add_g_term(g, hamiltonian, lattice_shape)
    add_B_term(B, hamiltonian, lattice_shape) #####gauge!!!
    hamiltonian.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1,
                         gauge_field=S, gauge_sublattice1=2, gauge_sublattice2=1, gauge_site_offset=1, periodic_bc=periodic_bc)
    hamiltonian.add_term(name='g', strength=-1, sublattice1=4, sublattice2=0, site_offset=0, time_dependence=g)
    hamiltonian.add_term(name='B', strength=-1, sublattice1=4, sublattice2=5, site_offset=0, time_dependence=B)
    decoupled_hamiltonian_with_gauge = TransverseFieldIsingHamiltonian(system_shape)
    decoupled_hamiltonian_with_gauge.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
    decoupled_hamiltonian_with_gauge.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1,
                                              gauge_field=S, gauge_sublattice1=2, gauge_sublattice2=1,
                                              gauge_site_offset=1, periodic_bc=periodic_bc)
    E_gs = ground_state.get_energy(decoupled_hamiltonian_matrix)

    spin_to_fermion_sublattices = {'tau_x': {'sublattice1': 4, 'sublattice2': 0},
                                   'tau_y': {'sublattice1': 0, 'sublattice2': 5},
                                   'tau_z': {'sublattice1': 4, 'sublattice2': 5},
                                   'sigma_x': {'sublattice1': 2, 'sublattice2': 3},
                                   'sigma_y': {'sublattice1': 1, 'sublattice2': 3},
                                   'sigma_z': {'sublattice1': 1, 'sublattice2': 2}}

    all_errors_unitaries = {}
    errors_effect_gauge = {}
    for name, sublattices in spin_to_fermion_sublattices.items():
        for i in range(num_sites):
            all_errors_unitaries[name + '_' + str(i)] = get_fermion_bilinear_unitary(system_shape=system_shape, site1=[i],
                                                                                     site2=[i], **sublattices)
            errors_effect_gauge[name + '_' + str(i)] = np.any(
                [(sublattice in [1, 2]) for sublattice in sublattices.values()])

    return hamiltonian, S, decoupled_hamiltonian_with_gauge, E_gs, all_errors_unitaries, errors_effect_gauge
    # hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)
    #
    # add_J_term(J, hamiltonian, lattice_shape)
    # add_kappa_term(hamiltonian, kappa, lattice_shape)
    # add_g_term(g, hamiltonian, lattice_shape)
    # add_B_term(B, hamiltonian, lattice_shape)
    #
    # if initial_state == 'random':
    #     S = KSLState(system_shape)
    #     S.randomize()
    #     S.reset_all_tau()
    #
    # return hamiltonian, S


def add_B_term(B, hamiltonian, lattice_shape):
    hamiltonian.add_term(name='B_0', strength=-1, sublattice1=4, sublattice2=5, site_offset=(0, 0), time_dependence=B)
    hamiltonian.add_term(name='B_1', strength=-1, sublattice1=10, sublattice2=11, site_offset=(0, 0), time_dependence=B)

def add_g_term(g, hamiltonian, lattice_shape):
    hamiltonian.add_term(name='g_0', strength=-1, sublattice1=4, sublattice2=0, site_offset=(0, 0), time_dependence=g)
    hamiltonian.add_term(name='g_1', strength=-1, sublattice1=10, sublattice2=6, site_offset=(0, 0), time_dependence=g)

def add_kappa_term(kappa, hamiltonian, lattice_shape):
    site_offset_kappa_x = (1, 0)
    site_offset_kappa_y = (0, 1)
    site_offset_kappa_z = (1, -1)
    hamiltonian.add_term(name='kappa_x_0', strength=kappa, sublattice1=0, sublattice2=0, site_offset=site_offset_kappa_x)
    hamiltonian.add_term(name='kappa_x_1', strength=-kappa, sublattice1=6, sublattice2=6, site_offset=site_offset_kappa_x)
    hamiltonian.add_term(name='kappa_y_0', strength=-kappa, sublattice1=0, sublattice2=0, site_offset=site_offset_kappa_y)
    hamiltonian.add_term(name='kappa_y_1', strength=kappa, sublattice1=6, sublattice2=6, site_offset=site_offset_kappa_y)
    hamiltonian.add_term(name='kappa_z_0', strength=-kappa, sublattice1=0, sublattice2=0, site_offset=site_offset_kappa_z)
    hamiltonian.add_term(name='kappa_z_1', strength=kappa, sublattice1=6, sublattice2=6, site_offset=site_offset_kappa_z)

def add_J_term(J, hamiltonian, lattice_shape):
    site_offset_x = (0, 0)
    site_offset_y = (1, 0)
    site_offset_z = (0, 1)
    hamiltonian.add_term(name='Jx', strength=-J, sublattice1=0, sublattice2=6, site_offset=site_offset_x)
    hamiltonian.add_term(name='Jy', strength=J, sublattice1=6, sublattice2=0, site_offset=site_offset_y)
    hamiltonian.add_term(name='Jz', strength=J, sublattice1=6, sublattice2=0, site_offset=site_offset_z)

def add_gauge_fixing_term(hamiltonian, lattice_shape):
    site_offset_x = (0, 0)
    site_offset_y = (1, 0)
    site_offset_z = (0, 1)
    hamiltonian.add_term(name='Gx', strength=-1, sublattice1=1, sublattice2=7, site_offset=site_offset_x)
    hamiltonian.add_term(name='Gy', strength=1, sublattice1=8, sublattice2=2, site_offset=site_offset_y)
    hamiltonian.add_term(name='Gz', strength=1, sublattice1=9, sublattice2=3, site_offset=site_offset_z)


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
            self.reset(10, 11, site, site)



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
    Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
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
