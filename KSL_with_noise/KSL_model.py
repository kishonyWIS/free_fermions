import numpy as np

from KSL_with_noise.correcting_fluxes import KSL_flux_corrector
from free_fermion_hamiltonian import MajoranaFreeFermionHamiltonian, MajoranaSingleParticleDensityMatrix, \
    get_fermion_bilinear_unitary
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

    decoupled_hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)
    add_J_term(J, decoupled_hamiltonian)
    add_kappa_term(kappa, decoupled_hamiltonian)
    decoupled_hamiltonian_matrix = decoupled_hamiltonian.get_matrix()
    ground_state = decoupled_hamiltonian.get_ground_state()
    S0_tensor = np.zeros(system_shape)

    if initial_state == 'random':
        S_non_gauge = KSLState(non_gauge_system_shape)
        S_non_gauge.randomize()
        S0_tensor[non_gauge_idxs] = S_non_gauge.tensor
    elif initial_state == 'ground':
        S_non_gauge = ground_state
        S0_tensor[non_gauge_idxs] = S_non_gauge.tensor[non_gauge_idxs]

    gauge_setting_hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)
    add_gauge_fixing_term(gauge_setting_hamiltonian)

    S_gauge = gauge_setting_hamiltonian.get_ground_state()
    S0_tensor[gauge_idxs] = S_gauge.tensor[gauge_idxs]
    S = KSLState(system_shape=system_shape, tensor=S0_tensor)
    S.reset_all_tau()

    hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)
    add_J_term(J, hamiltonian, gauge_field=S)#####gauge!!!
    add_kappa_term(kappa, hamiltonian)#####gauge!!!
    add_g_term(g, hamiltonian)
    add_B_term(B, hamiltonian)
    # hamiltonian.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1,
    #                      gauge_field=S, gauge_sublattice1=2, gauge_sublattice2=1, gauge_site_offset=1, periodic_bc=periodic_bc)
    decoupled_hamiltonian_with_gauge = MajoranaFreeFermionHamiltonian(system_shape)
    add_J_term(J, decoupled_hamiltonian_with_gauge, gauge_field=S) #gauge!!!
    add_kappa_term(kappa, decoupled_hamiltonian_with_gauge) #gauge!!!
    E_gs = ground_state.get_energy(decoupled_hamiltonian_matrix)

    spin_to_fermion_sublattices = {}
    for sublattice_name, sublattice_shift in zip(['A', 'B'], [0, 6]):
        spin_to_fermion_sublattices.update({
            'tau_x_' + sublattice_name: {'sublattice1': 4 + sublattice_shift, 'sublattice2': 0 + sublattice_shift},
            'tau_y_' + sublattice_name: {'sublattice1': 0 + sublattice_shift, 'sublattice2': 5 + sublattice_shift},
            'tau_z_' + sublattice_name: {'sublattice1': 4 + sublattice_shift, 'sublattice2': 5 + sublattice_shift},
            'sigma_x_' + sublattice_name: {'sublattice1': 2 + sublattice_shift, 'sublattice2': 3 + sublattice_shift},
            'sigma_y_' + sublattice_name: {'sublattice1': 1 + sublattice_shift, 'sublattice2': 3 + sublattice_shift},
            'sigma_z_' + sublattice_name: {'sublattice1': 1 + sublattice_shift, 'sublattice2': 2 + sublattice_shift}
        })

    all_errors_unitaries = {}
    errors_effect_gauge = {}
    for name, sublattices in spin_to_fermion_sublattices.items():
        for i_x in range(num_sites_x):
            for i_y in range(num_sites_y):
                all_errors_unitaries[name + '_' + str(i_x) + '_' + str(i_y)] = get_fermion_bilinear_unitary(system_shape=system_shape, site1=[i_x, i_y],
                                                                                     site2=[i_x, i_y], **sublattices)
                errors_effect_gauge[name + '_' + str(i_x) + '_' + str(i_y)] = np.any(
                    [(sublattice in gauge_sublattices) for sublattice in sublattices.values()])

    return hamiltonian, S, decoupled_hamiltonian_with_gauge, E_gs, all_errors_unitaries, errors_effect_gauge


def add_B_term(B, hamiltonian):
    hamiltonian.add_term(name='B_0', strength=-1, sublattice1=4, sublattice2=5, site_offset=(0, 0), time_dependence=B)
    hamiltonian.add_term(name='B_1', strength=-1, sublattice1=10, sublattice2=11, site_offset=(0, 0), time_dependence=B)

def add_g_term(g, hamiltonian):
    hamiltonian.add_term(name='g_0', strength=-1, sublattice1=4, sublattice2=0, site_offset=(0, 0), time_dependence=g)
    hamiltonian.add_term(name='g_1', strength=-1, sublattice1=10, sublattice2=6, site_offset=(0, 0), time_dependence=g)

def add_kappa_term(kappa, hamiltonian):
    x_y_z_to_offset = {'x': (1, 0), 'y': (0, 1), 'z': (1, -1)}
    x_y_z_to_sign = {'x': 1, 'y': -1, 'z': -1}
    sublattice_name_to_sublattice = {'A': 0, 'B': 6}
    sublattice_name_to_sign = {'A': 1, 'B': -1}

    for shift in range(2):
        for x_y_z, offset in x_y_z_to_offset.items():
            for sublattice_name, sublattice in sublattice_name_to_sublattice.items():
                sign = x_y_z_to_sign[x_y_z] * sublattice_name_to_sign[sublattice_name]
                name = 'kappa_' + x_y_z + '_sublattice_' + sublattice_name + '_shift_' + str(shift)
                hamiltonian.add_term(name=name, strength=kappa*sign, sublattice1=sublattice, sublattice2=sublattice, site_offset=offset)
                dim_modulation = np.argmax(np.array(offset)!=0)
                filter_site1 = lambda s1: s1[dim_modulation]*offset[dim_modulation] % 2 == shift
                hamiltonian.terms[name].filter_site1(filter_site1)
                print()

def add_J_term(J, hamiltonian, gauge_field=None):
    site_offset_x = (0, 0)
    site_offset_y = (1, 0)
    site_offset_z = (0, 1)
    hamiltonian.add_term(name='Jx', strength=-J, sublattice1=0, sublattice2=6, site_offset=site_offset_x,
                         gauge_field=gauge_field, gauge_sublattice1=1, gauge_sublattice2=7, gauge_site_offset=site_offset_x)
    hamiltonian.add_term(name='Jy', strength=J, sublattice1=6, sublattice2=0, site_offset=site_offset_y,
                         gauge_field=gauge_field, gauge_sublattice1=8, gauge_sublattice2=2, gauge_site_offset=site_offset_y)
    hamiltonian.add_term(name='Jz', strength=J, sublattice1=6, sublattice2=0, site_offset=site_offset_z,
                         gauge_field=gauge_field, gauge_sublattice1=9, gauge_sublattice2=3, gauge_site_offset=site_offset_z)

def add_gauge_fixing_term(hamiltonian):
    site_offset_x = (0, 0)
    site_offset_y = (1, 0)
    site_offset_z = (0, 1)
    hamiltonian.add_term(name='Gx', strength=-1, sublattice1=1, sublattice2=7, site_offset=site_offset_x)
    hamiltonian.add_term(name='Gy', strength=1, sublattice1=8, sublattice2=2, site_offset=site_offset_y)
    hamiltonian.add_term(name='Gz', strength=1, sublattice1=9, sublattice2=3, site_offset=site_offset_z)


class KSLState(MajoranaSingleParticleDensityMatrix):
    def reset_all_tau(self):
        self.system_shape[:2]
        for site in np.ndindex(tuple(self.system_shape[:2])):
            self.reset(4, 5, site, site)
            self.reset(10, 11, site, site)

    def fluxes(self, periodic_bc=False):
        sites = np.meshgrid(*(np.arange(self.system_shape[dim]) for dim in range(2)),
                                 indexing='ij')
        sites_shifted_x = [(sites[0] + 1) % self.system_shape[0], sites[1]]
        sites_shifted_y = [sites[0], (sites[1] + 1) % self.system_shape[1]]
        sites_shifted_x_y = [(sites[0] + 1) % self.system_shape[0], (sites[1] + 1) % self.system_shape[1]]
        fluxes = np.ones(self.system_shape[:2])
        fluxes *= self.tensor[(*sites_shifted_x, 1, *sites_shifted_x, 7)]
        fluxes *= self.tensor[(*sites_shifted_y, 1, *sites_shifted_y, 7)]
        fluxes *= self.tensor[(*sites_shifted_x, 2, *sites, 8)]
        fluxes *= self.tensor[(*sites_shifted_x_y, 2, *sites_shifted_y, 8)]
        fluxes *= self.tensor[(*sites_shifted_y, 3, *sites, 9)]
        fluxes *= self.tensor[(*sites_shifted_x_y, 3, *sites_shifted_x, 9)]
        if not periodic_bc:
            fluxes = fluxes[:-1, :-1]
        return fluxes


if __name__ == '__main__':
    num_sites_x = 3
    num_sites_y = 3
    g0 = 0.5
    B1 = 0.
    B0 = 5.
    J = 1.
    kappa = 0.0
    periodic_bc = False

    cycles = 50

    trotter_steps = 2000

    T = 90.
    t1 = T / 4
    smoothed_g = lambda t: get_g(t, g0, T, t1)
    smoothed_B = lambda t: get_B(t, B0, B1, T)

    flux_corrector = KSL_flux_corrector(num_sites_x, num_sites_y, periodic_bc)

    hamiltonian, S, decoupled_hamiltonian_with_gauge, E_gs, all_errors_unitaries, errors_effect_gauge = \
        get_KSL_model(num_sites_x, num_sites_y, J, kappa, smoothed_g, smoothed_B, initial_state='random')

    print(E_gs)

    Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
    # integration_params = dict(name='vode', nsteps=50000, rtol=1e-12, atol=1e-16)
    # Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)

    errors_per_cycle_per_qubit = 1e-100
    errors_per_cycle = errors_per_cycle_per_qubit * num_sites_x * num_sites_y * 4
    errors_per_cycle = 0.3

    Es = []
    cycle = 0
    time_to_error = np.random.exponential(T / errors_per_cycle)
    time_in_current_cycle = 0.

    Es.append(S.get_energy(decoupled_hamiltonian_with_gauge.get_matrix()))
    # new_row = pd.DataFrame(
    #     {'Ns': num_sites, 'periodic_bc': periodic_bc, 'drop_one_g_for_odd_bath_signs': False, 'J': J,
    #      'h': h, 'V': 0, 'T': T, 'Nt': trotter_steps, 'N_iter': cycle,
    #      'errors_per_cycle_per_qubit': error_rate, 'energy_density': (Es[-1] - E_gs) / num_sites},
    #     index=[0])
    # results_df = pd.concat([results_df, new_row], ignore_index=True)


    while True:
        if cycle == cycles:
            # finished all cycles
            break
        elif time_to_error == 0:
            # print('apply error')
            error_name = np.random.choice(list(all_errors_unitaries.keys()))
            S.evolve_with_unitary(all_errors_unitaries[error_name])
            # print(error_name)
            if errors_effect_gauge[error_name]:
                # print('recalculating Ud')
                # Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
                Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
            time_to_error = np.random.exponential(T / errors_per_cycle)
        elif time_to_error > T and time_in_current_cycle == 0:
            # print('apply a full cycle unitary')
            S.evolve_with_unitary(Ud)
            time_to_error -= T
            time_in_current_cycle = T
        elif time_to_error < T - time_in_current_cycle:
            # print('apply a partial unitary until error')
            # Ud_temp = hamiltonian.full_cycle_unitary_faster(integration_params,
            #                                                 time_in_current_cycle,
            #                                                 time_in_current_cycle + time_to_error)
            steps = int(trotter_steps * time_to_error / T)
            if steps > 0:
                Ud_temp = hamiltonian.full_cycle_unitary_trotterize(time_in_current_cycle,
                                                                    time_in_current_cycle + time_to_error, steps=steps)
                S.evolve_with_unitary(Ud_temp)
            time_in_current_cycle += time_to_error
            time_to_error = 0
        elif time_in_current_cycle == T:
            # print('reset')
            S.reset_all_tau()
            cycle += 1
            print(cycle)
            Es.append(S.get_energy(decoupled_hamiltonian_with_gauge.get_matrix()))

            # correct fluxes
            # fluxes = S.fluxes(periodic_bc)
            # correction_names = flux_corrector.correct(fluxes)
            # for correction_name in correction_names:
            #     S.evolve_with_unitary(all_errors_unitaries[correction_name])
            # if any(errors_effect_gauge[correction_name] for correction_name in correction_names):
            #     Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)

            time_in_current_cycle = 0
            # new_row = pd.DataFrame(
            #     {'Ns': num_sites, 'periodic_bc': periodic_bc, 'drop_one_g_for_odd_bath_signs': False, 'J': J,
            #      'h': h, 'V': 0, 'T': T, 'Nt': trotter_steps, 'N_iter': cycle,
            #      'errors_per_cycle_per_qubit': error_rate, 'energy_density': (Es[-1] - E_gs) / num_sites},
            #     index=[0])
            # results_df = pd.concat([results_df, new_row], ignore_index=True)
        elif time_in_current_cycle > 0:
            # print('finish incomplete cycle')
            # Ud_temp = hamiltonian.full_cycle_unitary_faster(integration_params,
            #                                                 time_in_current_cycle,
            #                                                 T)
            steps = int(trotter_steps * (T - time_in_current_cycle) / T)
            if steps > 0:
                Ud_temp = hamiltonian.full_cycle_unitary_trotterize(time_in_current_cycle, T, steps=steps)
                S.evolve_with_unitary(Ud_temp)
            time_to_error -= T - time_in_current_cycle
            time_in_current_cycle = T
        else:
            raise 'invalid cycle state'

    print(Es[-1])
    print('ground state energy = ' + str(E_gs))
    plt.figure()
    plt.semilogy(np.arange(len(Es))+1, (Es-E_gs)/num_sites_x/num_sites_y)
    plt.show()



