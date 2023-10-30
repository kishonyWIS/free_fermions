from time import time
from typing import Callable

import numpy as np
import scipy

from free_fermion_hamiltonian import MajoranaFreeFermionHamiltonian, MajoranaSingleParticleDensityMatrix, get_fermion_bilinear_unitary
from scipy.linalg import eigh
from matplotlib import pyplot as plt


# def gaussian_filter(t, sigma):
#     return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-t**2/(2*sigma**2))
#
#
# def get_g(t: float, g0, T, t1):
#     # return g0 * np.exp(-(t-T/2)**2/(T**2 / 25))
#     return np.maximum(np.minimum(g0 * np.ones_like(t), (1 - np.abs(2 * t / T - 1)) * T / (2 * t1) * g0), 0)
#
#
# def get_B(t: float, B0, B1, T, t1):
#     return np.minimum(np.maximum(B1 * np.ones_like(t), B0 + (B1 - B0) * t / (T - t1)), B0)
#
#
# def get_smoothed_func(t: float, func: Callable,  sigma: float):
#     N = 100
#     ts = np.linspace(-3*sigma, 3*sigma, N)
#     dt = ts[1]-ts[0]
#     return np.sum(np.array(list(map(func,ts+t)))*gaussian_filter(ts,sigma))*dt


def get_translationally_invariant_hamiltonian(J, h, k, g, B):
    return np.array([[0, 1j*J*np.exp(-1j*k) - 1j*h, 1j * g, 0],
                     [-1j*J*np.exp(1j*k) + 1j*h, 0, 0, 0],
                     [-1j*g, 0, 0, -1j*B],
                     [0, 0, 1j*B, 0]])


def get_translationally_invariant_spectrum(J, h, k, g, B):
    H = get_translationally_invariant_hamiltonian(J, h, k, g, B)
    E, _ = eigh(H)
    return E


class TransverseFieldIsingHamiltonian(MajoranaFreeFermionHamiltonian):
    def _unitary_trotterize_run_step(self, Ud, t):
        e_J_2 = self.terms['J'].small_unitary(t + self.dt / 2, dt_factor=0.5)
        e_B_2 = self.terms['B'].small_unitary(t + self.dt / 2, dt_factor=0.5)
        e_h_2 = self.terms['h'].small_unitary(t + self.dt / 2, dt_factor=0.5)
        e_g = self.terms['g'].small_unitary(t + self.dt / 2)
        Ud = e_J_2 @ e_B_2 @ e_h_2 @ e_g @ e_h_2 @ e_B_2 @ e_J_2 @ Ud
        return Ud


class TransverseFieldIsingState(MajoranaSingleParticleDensityMatrix):
    def reset_all_tau(self):
        for i in range(self.system_shape[0]):
            self.reset(4, 5, i, i)


def get_TFI_model(num_sites, h, J, g, B, initial_state = 'random', periodic_bc=False):
    num_sublattices = 6
    system_shape = (num_sites, num_sublattices, num_sites, num_sublattices)
    non_gauge_shape = (num_sites, 4, num_sites, 4)
    gauge_shape = (num_sites, 2, num_sites, 2)
    non_gauge_idxs = np.ix_(range(num_sites), [0, 3, 4, 5], range(num_sites), [0, 3, 4, 5])
    gauge_idxs = np.ix_(range(num_sites), [1, 2], range(num_sites), [1, 2])

    decoupled_hamiltonian = TransverseFieldIsingHamiltonian(system_shape)
    decoupled_hamiltonian.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
    decoupled_hamiltonian.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1, periodic_bc=periodic_bc)
    decoupled_hamiltonian_matrix = decoupled_hamiltonian.get_matrix()
    ground_state = decoupled_hamiltonian.get_ground_state()
    S0_tensor = np.zeros(system_shape)

    if initial_state == 'random':
        S_non_gauge = TransverseFieldIsingState(non_gauge_shape)
        S_non_gauge.randomize()
        S0_tensor[non_gauge_idxs] = S_non_gauge.tensor
    elif initial_state == 'ground':
        S_non_gauge = decoupled_hamiltonian.get_ground_state()
        S0_tensor[non_gauge_idxs] = S_non_gauge.tensor[non_gauge_idxs]

    gauge_setting_hamiltonian = TransverseFieldIsingHamiltonian(system_shape)
    gauge_setting_hamiltonian.add_term(name='G', strength=-1, sublattice1=2, sublattice2=1, site_offset=1, periodic_bc=periodic_bc)
    S_gauge = gauge_setting_hamiltonian.get_ground_state()
    S0_tensor[gauge_idxs] = S_gauge.tensor[gauge_idxs]
    S = TransverseFieldIsingState(system_shape=system_shape, tensor=S0_tensor)
    S.reset_all_tau()
    hamiltonian = TransverseFieldIsingHamiltonian(system_shape, dt=1.)
    hamiltonian.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
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
