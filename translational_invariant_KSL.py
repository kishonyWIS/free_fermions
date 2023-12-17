from free_fermion_hamiltonian import ComplexFreeFermionHamiltonian, ComplexSingleParticleDensityMatrix
import numpy as np


class TranslationInvariantKSLHamiltonian(ComplexFreeFermionHamiltonian):
    pass
    # def _unitary_trotterize_run_step(self, Ud, t):
    #     e_g = self.terms['g_a'].small_unitary(t + self.dt / 2) @ self.terms['g_d'].small_unitary(t + self.dt / 2)
    #     e_b_2 = self.terms['B_a'].small_unitary(t + self.dt / 2, dt_factor=0.5) @ \
    #             self.terms['B_d'].small_unitary(t + self.dt / 2, dt_factor=0.5) @ \
    #             self.terms['f_imag_a_d'].small_unitary(t + self.dt / 2, dt_factor=0.5) @ \
    #             self.terms['f_imag_d_a'].small_unitary(t + self.dt / 2, dt_factor=0.5)
    #     e_d_2 = self.terms['Delta'].small_unitary(t + self.dt / 2, dt_factor=0.5)
    #     e_r_4 = self.terms['f_real_a'].small_unitary(t + self.dt / 2, dt_factor=0.25) @ \
    #             self.terms['f_real_d'].small_unitary(t + self.dt / 2, dt_factor=0.25)
    #     e_r_b_2 = e_r_4 @ e_b_2 @ e_r_4
    #     Ud = e_r_b_2 @ e_d_2 @ e_g @ e_d_2 @ e_r_b_2 @ Ud
    #     return Ud


class TranslationInvariantKSLState(ComplexSingleParticleDensityMatrix):
    def reset_all_tau(self):
        for i in range(self.system_shape[0]):
            self.reset(2, 4, i, i)
            self.reset(3, 5, i, i)


def get_KSL_model(f, Delta, g, B, initial_state='random', num_cooling_sublattices = 2):
    # A,B are the labels of the sites.
    # within each site, the operators are ordered by cA, cB, b4A, b4B, b5A, B5b
    num_sublattices = 6
    num_sites = 1
    system_shape = (num_sites, num_sublattices, num_sites, num_sublattices)

    hamiltonian = TranslationInvariantKSLHamiltonian(system_shape, dt=1.)
    hamiltonian.add_term(name='Delta_plus', strength=Delta, sublattice1=0, sublattice2=0, site_offset=(0,))
    hamiltonian.add_term(name='Delta_minus', strength=-Delta, sublattice1=1, sublattice2=1, site_offset=(0,))
    hamiltonian.add_term(name='f', strength=2*1j*f, sublattice1=0, sublattice2=1, site_offset=(0,))

    ground_state = hamiltonian.get_ground_state()
    E_gs = ground_state.get_energy(hamiltonian.get_matrix())

    g_strengthA = -1j
    if num_cooling_sublattices == 2:
        g_strengthB = -1j
    if num_cooling_sublattices == 1:
        g_strengthB = 0

    hamiltonian.add_term(name='g_A', strength=4*g_strengthA, sublattice1=0, sublattice2=2, site_offset=(0,), time_dependence=g)
    hamiltonian.add_term(name='g_B', strength=4*g_strengthB, sublattice1=1, sublattice2=3, site_offset=(0,), time_dependence=g)
    hamiltonian.add_term(name='B_A', strength=4j, sublattice1=2, sublattice2=4, site_offset=(0,), time_dependence=B)
    hamiltonian.add_term(name='B_B', strength=4j, sublattice1=3, sublattice2=5, site_offset=(0,), time_dependence=B)

    if initial_state == 'random':
        S = TranslationInvariantKSLState(system_shape)
        S.randomize()
    elif initial_state == 'ground':
        S = TranslationInvariantKSLState(system_shape=system_shape, tensor=ground_state.tensor)
    elif initial_state == 'product':
        S = TranslationInvariantKSLState(system_shape=system_shape, tensor=np.eye(6, dtype=complex).reshape(system_shape))

    S.reset_all_tau()
    return hamiltonian, S, E_gs


def get_f(kx, ky, Jx, Jy, Jz):
    return 2*(-Jx-Jy*np.exp(-1j*kx)-Jz*np.exp(-1j*ky))


def get_Delta(kx, ky, kappa):
    return 4*kappa*(np.sin(kx) - np.sin(ky) + np.sin(ky-kx))
