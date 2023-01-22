import numpy as np
from free_fermion_hamiltonian import FreeFermionHamiltonian, SingleParticleDensityMatrix, \
    HamiltonianTerm, get_fermion_bilinear_unitary, fidelity
from scipy.linalg import eigh
from matplotlib import pyplot as plt


def get_g(t: float):
    return np.minimum(g0 * np.ones_like(t), (1 - np.abs(2 * t / T - 1)) * T / (2 * t1) * g0)


def get_B(t: float):
    return np.maximum(B1 * np.ones_like(t), B0 + (B1 - B0) * t / (T - t1))

def get_translationally_invariant_hamiltonian(J, h, k, g, B):
    return np.array([[0, 1j*J*np.exp(-1j*k) - 1j*h, 1j * g, 0],
                     [-1j*J*np.exp(1j*k) + 1j*h, 0, 0, 0],
                     [-1j*g, 0, 0, -1j*B],
                     [0, 0, 1j*B, 0]])

def get_translationally_invariant_spectrum(J, h, k, g, B):
    H = get_translationally_invariant_hamiltonian(J, h, k, g, B)
    E, _ = eigh(H)
    return E

if __name__ == '__main__':
    num_sites = 3
    num_sublattices = 6
    system_shape = (num_sites, num_sublattices, num_sites, num_sublattices)
    non_gauge_shape = (num_sites, 4, num_sites, 4)
    gauge_shape = (num_sites, 2, num_sites, 2)
    non_gauge_idxs = np.ix_(range(num_sites), [0, 3, 4, 5], range(num_sites), [0, 3, 4, 5])
    gauge_idxs = np.ix_(range(num_sites), [1, 2], range(num_sites), [1, 2])
    h = 1
    J = 0.7
    g0 = 0.3
    B1 = 0.
    B0 = 3.
    T = 300.
    t1 = T / 4

    # Es = []
    # Bs = np.linspace(0,3,100)
    # for B in Bs:
    #     Es.append(get_translationally_invariant_spectrum(J=J,h=h,k=0,g=0,B=B))
    # plt.plot(Bs, Es)
    # plt.show()


    integration_params = dict(name='vode', nsteps=20000, rtol=1e-8, atol=1e-12)

    decoupled_hamiltonian = FreeFermionHamiltonian(system_shape)
    decoupled_hamiltonian.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
    decoupled_hamiltonian.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1)
    decoupled_hamiltonian_matrix = decoupled_hamiltonian.get_matrix()
    ground_state = decoupled_hamiltonian.get_ground_state()

    S0_tensor = np.zeros(system_shape, dtype=complex)

    S_non_gauge = SingleParticleDensityMatrix(non_gauge_shape)
    S_non_gauge.randomize()
    S0_tensor[non_gauge_idxs] = S_non_gauge.tensor
    # S_non_gauge = decoupled_hamiltonian.get_ground_state()
    # S0_tensor[non_gauge_idxs] = S_non_gauge.tensor[non_gauge_idxs]

    gauge_setting_hamiltonian = FreeFermionHamiltonian(system_shape)
    gauge_setting_hamiltonian.add_term(name='G', strength=-1, sublattice1=2, sublattice2=1, site_offset=1)
    S_gauge = gauge_setting_hamiltonian.get_ground_state()
    S0_tensor[gauge_idxs] = S_gauge.tensor[gauge_idxs]
    S = SingleParticleDensityMatrix(system_shape=system_shape, tensor=S0_tensor)
    for i in range(num_sites):
        S.reset(4, 5, i, i)

    hamiltonian = FreeFermionHamiltonian(system_shape, dt=1.)
    hamiltonian.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
    hamiltonian.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1,
                         gauge_field=S, gauge_sublattice1=2, gauge_sublattice2=1, gauge_site_offset=1)
    hamiltonian.add_term(name='g', strength=-1, sublattice1=4, sublattice2=0, site_offset=0, time_dependence=get_g)
    hamiltonian.add_term(name='B', strength=-1, sublattice1=4, sublattice2=5, site_offset=0, time_dependence=get_B)

    decoupled_hamiltonian_with_gauge = FreeFermionHamiltonian(system_shape)
    decoupled_hamiltonian_with_gauge.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
    decoupled_hamiltonian_with_gauge.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1,
                                              gauge_field=S, gauge_sublattice1=2, gauge_sublattice2=1, gauge_site_offset=1)

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
            all_errors_unitaries[name + '_' + str(i)] = get_fermion_bilinear_unitary(system_shape=system_shape, site1=i, site2=i,
                                                                                     integration_params=integration_params, **sublattices)
            errors_effect_gauge[name + '_' + str(i)] = np.any([(sublattice in [1,2]) for sublattice in sublattices.values()])



    Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
    # Ud_trotter = hamiltonian.full_cycle_unitary_trotterize(0, T, rtol=1e-8)
    # for dt in [1.,0.1,0.01]:
    #     hamiltonian.dt = dt
    #     Ud_trotter = hamiltonian.full_cycle_unitary_trotterize(0, T)
    #     print(np.sum(np.abs(Ud-Ud_trotter)))

    Es = []
    Fs = []
    for _ in range(50):
        if np.random.rand() < 0.:
            error_name = np.random.choice(list(all_errors_unitaries.keys()))
            S.evolve_with_unitary(all_errors_unitaries[error_name])
            print(error_name)
            if errors_effect_gauge[error_name]:
                print('recalculating Ud')
                Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
        S.evolve_with_unitary(Ud)
        for i in range(num_sites):
            S.reset(4, 5, i, i)
        Es.append(S.get_energy(decoupled_hamiltonian_with_gauge.get_matrix()))
        Fs.append(fidelity(S.matrix, ground_state.matrix))
    E_gs = ground_state.get_energy(decoupled_hamiltonian_matrix)
    print(Es[-1])
    print('ground state energy = ' + str(E_gs))
    plt.figure()
    plt.plot(Es)
    plt.plot([E_gs]*len(Es))

    plt.figure()
    plt.plot(Fs)
    plt.show()