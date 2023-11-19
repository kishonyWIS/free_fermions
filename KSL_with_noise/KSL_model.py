import numpy as np

from KSL_with_noise.correcting_fluxes import KSL_flux_corrector
from free_fermion_hamiltonian import MajoranaFreeFermionHamiltonian, MajoranaSingleParticleDensityMatrix, \
    get_fermion_bilinear_unitary
from matplotlib import pyplot as plt
import pandas as pd
from time_dependence_functions import get_g, get_B
from space_resolved_energy import SpatialEnergy
np.random.seed(0)
from collections.abc import Iterable


class KSLHamiltonian(MajoranaFreeFermionHamiltonian):
    def __init__(self, system_shape, **kwargs):
        super().__init__(system_shape, **kwargs)
        self._small_unitary_dict = {}
        self.dt_factor = 0.5

    def small_unitary(self, term_name, t):
        if self.terms[term_name].time_dependence is not None:
            return self.terms[term_name].small_unitary(t, dt_factor=self.dt_factor)
        else:
            if term_name not in self._small_unitary_dict:
                self._small_unitary_dict[term_name] = self.terms[term_name].small_unitary(t, dt_factor=self.dt_factor)
            return self._small_unitary_dict[term_name]

    def set_new_gauge(self):
        for term_name, term in self.terms.items():
            if ((term.gauge_field is not None or term_name not in self._small_unitary_dict)
                    and term.time_dependence is None):
                self._small_unitary_dict[term_name] = term.small_unitary(None, dt_factor=self.dt_factor)

    # def _unitary_trotterize_run_step(self, Ud, t):
    #     for term_name in self.terms:
    #         Ud = self.small_unitary(term_name, t + self.dt / 2) @ Ud
    #     return Ud

    def _unitary_trotterize_run_step(self, Ud, t):
        Jx = self.small_unitary('Jx', t + self.dt / 2)
        Jy = self.small_unitary('Jy', t + self.dt / 2)
        Jz = self.small_unitary('Jz', t + self.dt / 2)
        kappa_x_A_0 = self.small_unitary('kappa_x_sublattice_A_shift_0', t + self.dt / 2)
        kappa_y_A_0 = self.small_unitary('kappa_y_sublattice_A_shift_0', t + self.dt / 2)
        kappa_z_A_0 = self.small_unitary('kappa_z_sublattice_A_shift_0', t + self.dt / 2)
        kappa_x_B_0 = self.small_unitary('kappa_x_sublattice_B_shift_0', t + self.dt / 2)
        kappa_y_B_0 = self.small_unitary('kappa_y_sublattice_B_shift_0', t + self.dt / 2)
        kappa_z_B_0 = self.small_unitary('kappa_z_sublattice_B_shift_0', t + self.dt / 2)
        kappa_x_A_1 = self.small_unitary('kappa_x_sublattice_A_shift_1', t + self.dt / 2)
        kappa_y_A_1 = self.small_unitary('kappa_y_sublattice_A_shift_1', t + self.dt / 2)
        kappa_z_A_1 = self.small_unitary('kappa_z_sublattice_A_shift_1', t + self.dt / 2)
        kappa_x_B_1 = self.small_unitary('kappa_x_sublattice_B_shift_1', t + self.dt / 2)
        kappa_y_B_1 = self.small_unitary('kappa_y_sublattice_B_shift_1', t + self.dt / 2)
        kappa_z_B_1 = self.small_unitary('kappa_z_sublattice_B_shift_1', t + self.dt / 2)
        g_0 = self.small_unitary('g_0', t + self.dt / 2)
        g_1 = self.small_unitary('g_1', t + self.dt / 2)
        B_0 = self.small_unitary('B_0', t + self.dt / 2)
        B_1 = self.small_unitary('B_1', t + self.dt / 2)
        small_U = B_0 @ B_1 @ Jx @ g_0 @ g_1 @ kappa_x_A_0 @ kappa_x_B_0 @ kappa_x_A_1 @ kappa_x_B_1 @ kappa_y_A_0 @ kappa_y_B_0 @ kappa_y_A_1 @ kappa_y_B_1 @ kappa_z_A_0 @ kappa_z_B_0 @ kappa_z_A_1 @ kappa_z_B_1 @ Jy @ Jz
        small_U_reversed = Jz @ Jy @ kappa_z_B_1 @ kappa_z_A_1 @ kappa_z_B_0 @ kappa_z_A_0 @ kappa_y_B_1 @ kappa_y_A_1 @ kappa_y_B_0 @ kappa_y_A_0 @ kappa_x_B_1 @ kappa_x_A_1 @ kappa_x_B_0 @ kappa_x_A_0 @ g_1 @ g_0 @ Jx @ B_1 @ B_0
        Ud = (small_U @ small_U_reversed) @ Ud
        return Ud


def get_KSL_model(num_sites_x, num_sites_y, J, kappa, g, B, initial_state, periodic_bc):
    # sublattices are numbered c_0, b^1_0, b^2_0, b^3_0, b^4_0, b^5_0, c_1, b^1_1, b^2_1, b^3_1, b^4_1, b^5_1
    gauge_sublattices = [1, 2, 3, 7, 8, 9]
    non_gauge_sublattices = [0, 4, 5, 6, 10, 11]

    num_gauge_sublattices = len(gauge_sublattices)
    num_non_gauge_sublattices = len(non_gauge_sublattices)
    num_sublattices = num_gauge_sublattices + num_non_gauge_sublattices

    lattice_shape = (num_sites_x, num_sites_y, num_sublattices)

    system_shape = lattice_shape*2

    gauge_idxs = np.ix_(range(num_sites_x), range(num_sites_y), gauge_sublattices, range(num_sites_x), range(num_sites_y), gauge_sublattices)

    # for finding ground state
    hamiltonian_fixed_gauge = KSLHamiltonian(system_shape)
    add_J_term(J, hamiltonian_fixed_gauge, periodic_bc=periodic_bc)
    add_kappa_term(kappa, hamiltonian_fixed_gauge, periodic_bc=periodic_bc)
    add_g_term(g, hamiltonian_fixed_gauge, periodic_bc=periodic_bc)
    add_B_term(B, hamiltonian_fixed_gauge, periodic_bc=periodic_bc)
    add_gauge_fixing_term(hamiltonian_fixed_gauge, periodic_bc=periodic_bc)
    S_gs = hamiltonian_fixed_gauge.get_ground_state(0)

    S0_tensor = np.zeros(system_shape)

    if initial_state == 'random':
        S0_tensor[gauge_idxs] = S_gs.tensor[gauge_idxs].copy()
    elif initial_state == 'ground':
        S0_tensor = S_gs.tensor.copy().astype(np.float64)

    S = KSLState(system_shape=system_shape, tensor=S0_tensor)
    S.reset_all_tau()

    hamiltonian = KSLHamiltonian(system_shape)
    add_J_term(J, hamiltonian, gauge_field=S, periodic_bc=periodic_bc)#####gauge!!!
    add_kappa_term(kappa, hamiltonian, gauge_field=S, periodic_bc=periodic_bc)#####gauge!!!
    add_g_term(g, hamiltonian, periodic_bc=periodic_bc)
    add_B_term(B, hamiltonian, periodic_bc=periodic_bc)

    decoupled_hamiltonian_with_gauge = KSLHamiltonian(system_shape)
    add_J_term(J, decoupled_hamiltonian_with_gauge, gauge_field=S, periodic_bc=periodic_bc) #gauge!!!
    add_kappa_term(kappa, decoupled_hamiltonian_with_gauge, gauge_field=S, periodic_bc=periodic_bc) #gauge!!!

    E_gs = S_gs.get_energy(decoupled_hamiltonian_with_gauge.get_matrix())

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

    return hamiltonian, S, S_gs, decoupled_hamiltonian_with_gauge, hamiltonian_fixed_gauge, E_gs, all_errors_unitaries, errors_effect_gauge

def add_B_term(B, hamiltonian, periodic_bc=False):
    hamiltonian.add_term(name='B_0', strength=-1, sublattice1=4, sublattice2=5, site_offset=(0, 0), time_dependence=B, periodic_bc=periodic_bc)
    hamiltonian.add_term(name='B_1', strength=-1, sublattice1=10, sublattice2=11, site_offset=(0, 0), time_dependence=B, periodic_bc=periodic_bc)

def add_g_term(g, hamiltonian, periodic_bc=False):
    hamiltonian.add_term(name='g_0', strength=-1, sublattice1=4, sublattice2=0, site_offset=(0, 0), time_dependence=g, periodic_bc=periodic_bc)
    hamiltonian.add_term(name='g_1', strength=-1, sublattice1=10, sublattice2=6, site_offset=(0, 0), time_dependence=g, periodic_bc=periodic_bc)

def add_kappa_term(kappa, hamiltonian, gauge_field=None, periodic_bc=False):
    kappa_df = pd.DataFrame(columns=['name', 'x_y_z', 'sublattice_name', 'shift', 'sign', 'sublattice1', 'sublattice2', 'offset', 'gauge_sublattice1', 'gauge_sublattice2', 'gauge_offset1', 'gauge_offset2'])
    kappa_df['x_y_z'] = ['x', 'y', 'z', 'x', 'y', 'z', 'x', 'y', 'z', 'x', 'y', 'z']
    kappa_df['sublattice_name'] = ['A', 'A', 'A', 'B', 'B', 'B', 'A', 'A', 'A', 'B', 'B', 'B']
    kappa_df['shift'] = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1 ,1]
    kappa_df['name'] = 'kappa_' + kappa_df['x_y_z'] + '_sublattice_' + kappa_df['sublattice_name'] + '_shift_' + kappa_df['shift'].astype(str)
    kappa_df['sign'] = kappa_df['x_y_z'].map({'x': 1, 'y': -1, 'z': -1}) * kappa_df['sublattice_name'].map({'A': 1, 'B': -1})
    kappa_df['sublattice1'] = kappa_df['sublattice_name'].map({'A': 0, 'B': 6})
    kappa_df['sublattice2'] = kappa_df['sublattice_name'].map({'A': 0, 'B': 6})
    kappa_df['offset'] = kappa_df['x_y_z'].map({'x': (1, 0), 'y': (0, 1), 'z': (1, -1)})
    kappa_df['gauge_sublattice1'] = [[1,2], [1,3], [3,2], [2,1], [3,1], [2,3]] * 2
    kappa_df['gauge_sublattice2'] = [[7,8], [7,9], [9,8], [8,7], [9,7], [8,9]] * 2
    kappa_df['gauge_offset1'] = [[(0,0),(1,0)], [(0,0), (0,1)], [(0,0),(1,-1)], [(1,0),(1,0)], [(0,1),(0,1)], [(1,0),(1,0)]] * 2
    kappa_df['gauge_offset2'] = [[(0,0),(0,0)], [(0,0), (0,0)], [(0,-1),(0,-1)], [(0,0),(1,0)], [(0,0),(0,1)], [(0,0),(1,-1)]] * 2

    for row in kappa_df.itertuples():
        hamiltonian.add_term(name=row.name, strength=kappa*row.sign, sublattice1=row.sublattice1, sublattice2=row.sublattice2, site_offset=row.offset,
                             gauge_field=gauge_field, gauge_sublattice1=row.gauge_sublattice1, gauge_sublattice2=row.gauge_sublattice2, gauge_site_offset1=row.gauge_offset1, gauge_site_offset2=row.gauge_offset2, periodic_bc=periodic_bc)
        dim_modulation = np.argmax(np.array(row.offset)!=0)
        filter_site1 = lambda s1: (s1[dim_modulation]*row.offset[dim_modulation]) % 2 == row.shift
        hamiltonian.terms[row.name].filter_site1(filter_site1)

def add_J_term(J, hamiltonian, gauge_field=None, periodic_bc=False):
    site_offset_x = (0, 0)
    site_offset_y = (1, 0)
    site_offset_z = (0, 1)
    hamiltonian.add_term(name='Jx', strength=-J, sublattice1=0, sublattice2=6, site_offset=site_offset_x,
                         gauge_field=gauge_field, gauge_sublattice1=1, gauge_sublattice2=7, gauge_site_offset1=(0,0), gauge_site_offset2=site_offset_x, periodic_bc=periodic_bc)
    hamiltonian.add_term(name='Jy', strength=J, sublattice1=6, sublattice2=0, site_offset=site_offset_y,
                         gauge_field=gauge_field, gauge_sublattice1=2, gauge_sublattice2=8, gauge_site_offset1=site_offset_y, gauge_site_offset2=(0,0), periodic_bc=periodic_bc)
    hamiltonian.add_term(name='Jz', strength=J, sublattice1=6, sublattice2=0, site_offset=site_offset_z,
                         gauge_field=gauge_field, gauge_sublattice1=3, gauge_sublattice2=9, gauge_site_offset1=site_offset_z, gauge_site_offset2=(0,0), periodic_bc=periodic_bc)

def add_gauge_fixing_term(hamiltonian, periodic_bc=False):
    site_offset_x = (0, 0)
    site_offset_y = (1, 0)
    site_offset_z = (0, 1)
    hamiltonian.add_term(name='Gx', strength=-1, sublattice1=1, sublattice2=7, site_offset=site_offset_x, periodic_bc=periodic_bc)
    hamiltonian.add_term(name='Gy', strength=1, sublattice1=8, sublattice2=2, site_offset=site_offset_y, periodic_bc=periodic_bc)
    hamiltonian.add_term(name='Gz', strength=1, sublattice1=9, sublattice2=3, site_offset=site_offset_z, periodic_bc=periodic_bc)


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
        if isinstance(periodic_bc, Iterable):
            fluxes = fluxes[:None if periodic_bc[0] else -1, :None if periodic_bc[1] else -1]
        return fluxes

    def fluxes_around_torus(self):
        flux_x = 1
        flux_y = 1
        for i_x in range(self.system_shape[0]):
            flux_x *= self.tensor[(i_x, 0, 1, i_x, 0, 7)]
            flux_x *= self.tensor[((i_x+1)%self.system_shape[0], 0, 2, i_x, 0, 8)]
        for i_y in range(self.system_shape[1]):
            flux_y *= self.tensor[(0, i_y, 1, 0, i_y, 7)]
            flux_y *= self.tensor[(0, (i_y+1)%self.system_shape[1], 3, 0, i_y, 9)]
        return flux_x, flux_y


def cool_KSL(num_sites_x, num_sites_y, J, kappa, smoothed_g, smoothed_B, initial_state, periodic_bc, cycles, errors_per_cycle, trotter_steps, T, flux_corrector, draw_spatial_energy=False, cycles_averaging_buffer=0):
    hamiltonian, S, S_gs, decoupled_hamiltonian_with_gauge, hamiltonian_fixed_gauge, E_gs, all_errors_unitaries, errors_effect_gauge = \
        get_KSL_model(num_sites_x, num_sites_y, J, kappa, smoothed_g, smoothed_B, initial_state=initial_state, periodic_bc=periodic_bc)
    Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
    Es = []
    fluxes_x = []
    fluxes_y = []
    cycle = 0
    time_to_error = np.random.exponential(T / errors_per_cycle)
    time_in_current_cycle = 0.
    Es.append(S.get_energy(decoupled_hamiltonian_with_gauge.get_matrix()))
    if draw_spatial_energy is not False:
        spatial_energy = SpatialEnergy(hamiltonian, hamiltonian_fixed_gauge, S_gs,
                                       [name for name in hamiltonian.terms.keys() if
                                        'J' in name or 'kappa' in name][::-1])
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
                hamiltonian.set_new_gauge()
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
            fluxes = S.fluxes(periodic_bc)
            correction_names = flux_corrector.correct(fluxes)
            for correction_name in correction_names:
                S.evolve_with_unitary(all_errors_unitaries[correction_name])
            if any(errors_effect_gauge[correction_name] for correction_name in correction_names):
                hamiltonian.set_new_gauge()
                Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)

            flux_x, flux_y = S.fluxes_around_torus()
            fluxes_x.append(flux_x)
            fluxes_y.append(flux_y)

            if (draw_spatial_energy == 'average' and cycle > cycles_averaging_buffer) or (draw_spatial_energy == 'last' and cycle == cycles):
                spatial_energy.update_matrix(hamiltonian)
                spatial_energy.update_energies(S)

            time_in_current_cycle = 0

        elif time_in_current_cycle > 0:
            # print('finish incomplete cycle')
            steps = int(trotter_steps * (T - time_in_current_cycle) / T)
            if steps > 0:
                Ud_temp = hamiltonian.full_cycle_unitary_trotterize(time_in_current_cycle, T, steps=steps)
                S.evolve_with_unitary(Ud_temp)
            time_to_error -= T - time_in_current_cycle
            time_in_current_cycle = T
        else:
            raise 'invalid cycle state'

    if draw_spatial_energy is not False:
        spatial_energy.draw(filename=f'KSL_spatial_energy_nx_{num_sites_x}_ny_{num_sites_y}_T_{T}_error_rate_{error_rate}_J_{J}_kappa_{kappa}_g_{g0}_B_{B0}_initial_state_{initial_state}_periodic_bc_{periodic_bc}_cycles_{cycles}_trotter_steps_{trotter_steps}_draw_spatial_energy_{draw_spatial_energy}.pdf')

    return np.array(Es)-E_gs, fluxes_x, fluxes_y



if __name__ == '__main__':
    num_sites_x = 4
    num_sites_y = 4
    g0 = 0.5
    B1 = 0.
    B0 = 5.
    J = 1.
    kappa = 0.1
    periodic_bc = (True, False)
    cycles_averaging_buffer = 3
    initial_state = "ground"
    draw_spatial_energy = 'average'

    cycles = 50

    trotter_steps = 100

    T_list = [20.]

    for T in T_list:

        t1 = T / 4
        smoothed_g = lambda t: get_g(t, g0, T, t1)
        smoothed_B = lambda t: get_B(t, B0, B1, T)

        flux_corrector = KSL_flux_corrector(num_sites_x, num_sites_y, periodic_bc)

        errors_per_cycle_per_qubit = [1e-99] #[1e-99], np.linspace(1e-99, 0.02, 10)

        for error_rate in errors_per_cycle_per_qubit:

            errors_per_cycle = error_rate * num_sites_x * num_sites_y * 4

            energy_above_ground, flux_x, flux_y = cool_KSL(num_sites_x, num_sites_y, J, kappa, smoothed_g, smoothed_B, initial_state=initial_state, periodic_bc=periodic_bc, cycles=cycles, errors_per_cycle=errors_per_cycle, trotter_steps=trotter_steps, T=T, flux_corrector=flux_corrector, cycles_averaging_buffer=cycles_averaging_buffer, draw_spatial_energy=draw_spatial_energy)

            results_df_averaged = pd.Series(
                {'num_sites_x': num_sites_x, 'num_sites_y': num_sites_y, 'periodic_bc': periodic_bc, 'J': J,
                 'kappa': kappa, 'g': g0, 'B': B0, 'T': T, 'Nt': trotter_steps, 'N_iter': cycles,
                 'errors_per_cycle_per_qubit': error_rate, 'energy_density': np.mean(energy_above_ground[cycles_averaging_buffer:]) / num_sites_x / num_sites_y,
                 'energy_density_std': np.std(energy_above_ground[cycles_averaging_buffer:]) / num_sites_x / num_sites_y / np.sqrt(cycles-cycles_averaging_buffer),
                 'initial_state': initial_state}).to_frame().transpose()

            results_df = pd.DataFrame()
            for cycle in range(cycles):
                new_row = pd.Series(
                    {'num_sites_x': num_sites_x, 'num_sites_y': num_sites_y, 'periodic_bc': periodic_bc, 'J': J,
                     'kappa': kappa, 'g': g0, 'B': B0, 'T': T, 'Nt': trotter_steps, 'N_iter': cycle,
                     'errors_per_cycle_per_qubit': error_rate, 'energy_density': energy_above_ground[cycle] / num_sites_x / num_sites_y, 'initial_state': initial_state,
                     'flux_x': flux_x[cycle], 'flux_y': flux_y[cycle]}).to_frame().transpose()
                results_df = pd.concat([results_df, new_row], ignore_index=True)

            # plt.figure()
            # plt.plot(energy_above_ground)
            #
            # plt.figure()
            # plt.plot(flux_x)
            # plt.plot(flux_y)

            print(energy_above_ground[-1])


            with open("KSL_results_averaged.csv", 'a') as f:
                results_df_averaged.to_csv(f, mode='a', header=f.tell()==0, index=False)

            with open("KSL_results.csv", 'a') as f:
                results_df.to_csv(f, mode='a', header=f.tell()==0, index=False)