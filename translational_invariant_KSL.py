from free_fermion_hamiltonian import FreeFermionHamiltonian, SingleParticleDensityMatrix
import numpy as np


class TranslationInvariantKSLHamiltonian(FreeFermionHamiltonian):
    def _unitary_trotterize_run_step(self, Ud, t):
        e_g = self.terms['g_a'].small_unitary(t + self.dt / 2) @ self.terms['g_d'].small_unitary(t + self.dt / 2)
        e_b_2 = self.terms['B_a'].small_unitary(t + self.dt / 2, dt_factor=0.5) @ \
                self.terms['B_d'].small_unitary(t + self.dt / 2, dt_factor=0.5) @ \
                self.terms['f_imag_a_d'].small_unitary(t + self.dt / 2, dt_factor=0.5) @ \
                self.terms['f_imag_d_a'].small_unitary(t + self.dt / 2, dt_factor=0.5)
        e_d_2 = self.terms['Delta'].small_unitary(t + self.dt / 2, dt_factor=0.5)
        e_r_4 = self.terms['f_real_a'].small_unitary(t + self.dt / 2, dt_factor=0.25) @ \
                self.terms['f_real_d'].small_unitary(t + self.dt / 2, dt_factor=0.25)
        e_r_b_2 = e_r_4 @ e_b_2 @ e_r_4
        Ud = e_r_b_2 @ e_d_2 @ e_g @ e_d_2 @ e_r_b_2 @ Ud
        return Ud


class TranslationInvariantKSLState(SingleParticleDensityMatrix):
    def reset_all_tau(self):
        for i in range(self.system_shape[0]):
            self.reset(1, 2, i, i)
            self.reset(4, 5, i, i)


def get_KSL_model(f_real, f_imag, Delta, g, B, initial_state='random', num_cooling_sublattices = 2):
    # A,B are the labels of the sites.
    # within each site, the operators are ordered by a0, a4, a5, d0, d4, d5
    num_sublattices = 6
    num_sites = 2
    system_shape = (num_sites, num_sublattices, num_sites, num_sublattices)

    hamiltonian = TranslationInvariantKSLHamiltonian(system_shape, dt=1.)
    hamiltonian.add_term(name='Delta', strength=np.array([0.25*Delta, -0.25*Delta]), sublattice1=0, sublattice2=3, site_offset=0)
    hamiltonian.add_term(name='f_real_a', strength=0.25*f_real, sublattice1=0, sublattice2=0, site_offset=1)
    hamiltonian.add_term(name='f_real_d', strength=0.25*f_real, sublattice1=3, sublattice2=3, site_offset=1)
    hamiltonian.add_term(name='f_imag_a_d', strength=-0.25*f_imag, sublattice1=0, sublattice2=3, site_offset=1)
    hamiltonian.add_term(name='f_imag_d_a', strength=0.25*f_imag, sublattice1=3, sublattice2=0, site_offset=1)

    ground_state = hamiltonian.get_ground_state()
    E_gs = ground_state.get_energy(hamiltonian.get_matrix())

    if num_cooling_sublattices == 2:
        g_strength = np.array([-0.25, -0.25])
    if num_cooling_sublattices == 1:
        g_strength = np.array([-0.25, 0])

    hamiltonian.add_term(name='g_a', strength=g_strength, sublattice1=1, sublattice2=0, site_offset=0, time_dependence=g)
    hamiltonian.add_term(name='g_d', strength=g_strength, sublattice1=4, sublattice2=3, site_offset=0, time_dependence=g)
    hamiltonian.add_term(name='B_a', strength=-0.25, sublattice1=1, sublattice2=2, site_offset=0, time_dependence=B)
    hamiltonian.add_term(name='B_d', strength=-0.25, sublattice1=4, sublattice2=5, site_offset=0, time_dependence=B)

    if initial_state == 'random':
        S = TranslationInvariantKSLState(system_shape)
        S.randomize()
    elif initial_state == 'ground':
        S = TranslationInvariantKSLState(system_shape=system_shape, tensor=ground_state.tensor)

    S.reset_all_tau()
    return hamiltonian, S, E_gs
