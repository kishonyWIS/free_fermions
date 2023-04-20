from time import time
from typing import Callable

import numpy as np
import scipy

from free_fermion_hamiltonian import MajoranaFreeFermionHamiltonian, MajoranaSingleParticleDensityMatrix, get_fermion_bilinear_unitary
from scipy.linalg import eigh
from matplotlib import pyplot as plt


def gaussian_filter(t, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-t ** 2 / (2 * sigma ** 2))


def get_g(t: float):
    # return g0 * np.exp(-(t-T/2)**2/(T**2 / 25))
    return np.maximum(np.minimum(g0 * np.ones_like(t), (1 - np.abs(2 * t / T - 1)) * T / (2 * t1) * g0), 0)


def get_B(t: float):
    return np.minimum(np.maximum(B1 * np.ones_like(t), B0 + (B1 - B0) * t / (T - t1)), B0)


def get_smoothed_func(t: float, func: Callable, sigma: float):
    N = 100
    ts = np.linspace(-3 * sigma, 3 * sigma, N)
    dt = ts[1] - ts[0]
    return np.sum(np.array(list(map(func, ts + t))) * gaussian_filter(ts, sigma)) * dt


def get_translationally_invariant_hamiltonian(J, h, k, g, B):
    return np.array([[0, 1j * J * np.exp(-1j * k) - 1j * h, 1j * g, 0],
                     [-1j * J * np.exp(1j * k) + 1j * h, 0, 0, 0],
                     [-1j * g, 0, 0, -1j * B],
                     [0, 0, 1j * B, 0]])


def get_translationally_invariant_spectrum(J, h, k, g, B):
    H = get_translationally_invariant_hamiltonian(J, h, k, g, B)
    E, _ = eigh(H)
    return E


class TransverseFieldIsingModel(MajoranaFreeFermionHamiltonian):
    non_gauge_sublattices = [0, 3, 4, 5]
    gauge_sublattices = [1, 2]
    num_non_gauge_sublattices = len(non_gauge_sublattices)
    num_gauge_sublattices = len(gauge_sublattices)
    num_sublattices = num_non_gauge_sublattices + num_gauge_sublattices

    def __init__(self, num_sites, h, J, g=None, B=None, T=None, G=None):
        self.num_sites = num_sites
        system_shape = (self.num_sites, self.num_sublattices, self.num_sites, self.num_sublattices)
        super().__init__(system_shape=system_shape)
        if G is not None:
            gauge_setting_hamiltonian.add_term(name='G', strength=-1, sublattice1=2, sublattice2=1, site_offset=1)
            return
        if B is None and g is None:

        self.non_gauge_shape = (
        self.num_sites, self.num_non_gauge_sublattices, self.num_sites, self.num_non_gauge_sublattices)
        self.gauge_shape = (self.num_sites, self.num_gauge_sublattices, self.num_sites, self.num_gauge_sublattices)
        self.non_gauge_idxs = np.ix_(range(self.num_sites), self.non_gauge_sublattices, range(self.num_sites),
                                     self.non_gauge_sublattices)
        self.gauge_idxs = np.ix_(range(self.num_sites), self.gauge_sublattices, range(self.num_sites),
                                 self.gauge_sublattices)

    def _unitary_trotterize_run_step(self, Ud, t):
        e_J_2 = self.terms['J'].small_unitary(t + self.dt / 2, dt_factor=0.5)
        e_B_2 = self.terms['B'].small_unitary(t + self.dt / 2, dt_factor=0.5)
        e_h_2 = self.terms['h'].small_unitary(t + self.dt / 2, dt_factor=0.5)
        e_g = self.terms['g'].small_unitary(t + self.dt / 2)
        Ud = e_J_2 @ e_B_2 @ e_h_2 @ e_g @ e_h_2 @ e_B_2 @ e_J_2 @ Ud
        return Ud


if __name__ == '__main__':
    num_sites = 10
    h = 0.7
    J = 1.
    g0 = 0.5
    B1 = 0.
    B0 = 3.
    T = 30.
    t1 = T / 4
    error_rate = 0.000000025  # errors per cycle

    smoothed_g = lambda t: get_smoothed_func(t, get_g, T / 10) - get_smoothed_func(T, get_g, T / 10)
    smoothed_B = lambda t: get_smoothed_func(t, get_B, T / 10) - get_smoothed_func(T, get_B, T / 10)
    ts = np.linspace(0, T, 1000)
    gs = []
    Bs = []
    for t in ts:
        gs.append(smoothed_g(t))
        Bs.append(smoothed_B(t))
    plt.plot(ts, gs, label='g')
    plt.plot(ts, Bs, label='B')
    plt.legend()
    plt.show()

    integration_params = dict(name='vode', nsteps=20000, rtol=1e-8, atol=1e-12)

    decoupled_hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)
    decoupled_hamiltonian.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
    decoupled_hamiltonian.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1)
    decoupled_hamiltonian_matrix = decoupled_hamiltonian.get_matrix()
    ground_state = decoupled_hamiltonian.get_ground_state()

    S0_tensor = np.zeros(system_shape)

    S_non_gauge = MajoranaSingleParticleDensityMatrix(non_gauge_shape)
    S_non_gauge.randomize()
    S0_tensor[non_gauge_idxs] = S_non_gauge.tensor
    # S_non_gauge = decoupled_hamiltonian.get_ground_state()
    # S0_tensor[non_gauge_idxs] = S_non_gauge.tensor[non_gauge_idxs]

    gauge_setting_hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)
    gauge_setting_hamiltonian.add_term(name='G', strength=-1, sublattice1=2, sublattice2=1, site_offset=1)
    S_gauge = gauge_setting_hamiltonian.get_ground_state()
    S0_tensor[gauge_idxs] = S_gauge.tensor[gauge_idxs]
    S = MajoranaSingleParticleDensityMatrix(system_shape=system_shape, tensor=S0_tensor)
    for i in range(num_sites):
        S.reset(4, 5, i, i)

    hamiltonian = MajoranaFreeFermionHamiltonian(system_shape, dt=1.)
    hamiltonian.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
    hamiltonian.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1,
                         gauge_field=S, gauge_sublattice1=2, gauge_sublattice2=1, gauge_site_offset=1)
    hamiltonian.add_term(name='g', strength=-1, sublattice1=4, sublattice2=0, site_offset=0, time_dependence=smoothed_g)
    hamiltonian.add_term(name='B', strength=-1, sublattice1=4, sublattice2=5, site_offset=0, time_dependence=smoothed_B)

    decoupled_hamiltonian_with_gauge = MajoranaFreeFermionHamiltonian(system_shape)
    decoupled_hamiltonian_with_gauge.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
    decoupled_hamiltonian_with_gauge.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1,
                                              gauge_field=S, gauge_sublattice1=2, gauge_sublattice2=1,
                                              gauge_site_offset=1)

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
            all_errors_unitaries[name + '_' + str(i)] = get_fermion_bilinear_unitary(system_shape=system_shape, site1=i,
                                                                                     site2=i,
                                                                                     integration_params=integration_params,
                                                                                     **sublattices)
            errors_effect_gauge[name + '_' + str(i)] = np.any(
                [(sublattice in [1, 2]) for sublattice in sublattices.values()])

    Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)

    steps_list = np.logspace(0, 4, 15, dtype=int)
    errors = []
    L1_errors = []
    for steps in steps_list:
        t0 = time()
        Ud_trotter = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=steps)
        tf = time()
        print(tf - t0)
        errors.append(np.sum(np.abs(Ud - Ud_trotter)) / np.sum(np.abs(Ud)))
        L1_errors.append(np.linalg.norm(1j * scipy.linalg.logm(Ud @ Ud_trotter.T.conj()), ord=2))
    plt.loglog(steps_list, errors, '.')
    plt.figure()
    plt.loglog(steps_list, L1_errors, '.')
    plt.show()


    def reset_all_b4b5(S):
        for i in range(num_sites):
            S.reset(4, 5, i, i)


    Es = []
    cycle = 0
    time_to_error = np.random.exponential(T / error_rate)
    time_in_current_cycle = 0.
    while True:
        if cycle == 50:
            # finished all cycles
            break
        elif time_to_error == 0:
            print('apply error')
            error_name = np.random.choice(list(all_errors_unitaries.keys()))
            S.evolve_with_unitary(all_errors_unitaries[error_name])
            print(error_name)
            if errors_effect_gauge[error_name]:
                print('recalculating Ud')
                Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
            time_to_error = np.random.exponential(T / error_rate)
        elif time_to_error > T and time_in_current_cycle == 0:
            print('apply a full cycle unitary')
            S.evolve_with_unitary(Ud)
            time_to_error -= T
            time_in_current_cycle = T
        elif time_to_error < T - time_in_current_cycle:
            print('apply a partial unitary until error')
            Ud_temp = hamiltonian.full_cycle_unitary_faster(integration_params,
                                                            time_in_current_cycle,
                                                            time_in_current_cycle + time_to_error)
            S.evolve_with_unitary(Ud_temp)
            time_in_current_cycle += time_to_error
            time_to_error = 0
        elif time_in_current_cycle == T:
            print('reset')
            reset_all_b4b5(S)
            Es.append(S.get_energy(decoupled_hamiltonian_with_gauge.get_matrix()))
            cycle += 1
            time_in_current_cycle = 0
        elif time_in_current_cycle > 0:
            print('finish incomplete cycle')
            Ud_temp = hamiltonian.full_cycle_unitary_faster(integration_params,
                                                            time_in_current_cycle,
                                                            T)
            S.evolve_with_unitary(Ud_temp)
            time_to_error -= T - time_in_current_cycle
            time_in_current_cycle = T
        else:
            raise 'invalid cycle state'

    E_gs = ground_state.get_energy(decoupled_hamiltonian_matrix)
    print(Es[-1])
    print('ground state energy = ' + str(E_gs))
    plt.figure()
    plt.plot(Es)
    plt.plot([E_gs] * len(Es))
    plt.show()
