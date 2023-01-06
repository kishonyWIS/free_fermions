from typing import List, Union, Tuple, Callable, Optional
import numpy as np
from scipy.integrate import complex_ode
from scipy.linalg import eigh, eig
from matplotlib import pyplot as plt
from scipy.stats import special_ortho_group

# TODO: trotterize
# TODO: add noise
# TODO: guage field


def add_to_diag(arr: np.ndarray, to_add: Union[int, List]):
    idx = np.diag_indices_from(arr)
    arr[idx] += to_add


def tensor_to_matrix(tensor: np.ndarray, system_shape: Tuple[int, ...]) -> np.ndarray:
    matrix_shape = get_system_matrix_shape(system_shape)
    return tensor.reshape(matrix_shape[0], matrix_shape[1])


def matrix_to_tensor(matrix: np.ndarray, system_shape: Tuple[int, ...]) -> np.ndarray:
    return matrix.reshape(system_shape)


def get_system_matrix_shape(system_shape):
    num_dims = len(system_shape)
    shape1 = np.prod(system_shape[:num_dims // 2])
    shape2 = np.prod(system_shape[num_dims // 2:])
    return shape1, shape2


def site_and_sublattice_to_flat_index(site, sublattice, system_shape):
    return np.ravel_multi_index((site, sublattice), system_shape[:2])


class HamiltonianTerm:
    def __init__(self,
                 strength: Union[float, List[float]],
                 sublattice1: int,
                 sublattice2: int,
                 site_offset: Union[int, Tuple[int]],
                 system_shape: Tuple[int, ...],
                 time_dependence: Optional[Callable] = None):
        self.system_shape = system_shape
        self.site_offset = site_offset
        self.sublattice1 = sublattice1
        self.sublattice2 = sublattice2
        self.strength = strength
        self.time_dependence = time_dependence

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, new_strength):
        self._strength = new_strength
        self._time_independent_tensor = np.zeros(self.system_shape)
        add_to_diag(
            self._time_independent_tensor[:self.system_shape[0] - self.site_offset, self.sublattice1, self.site_offset:,
            self.sublattice2], 2 * self._strength)
        add_to_diag(
            self._time_independent_tensor[self.site_offset:, self.sublattice2, :self.system_shape[2] - self.site_offset,
            self.sublattice1], -2 * self._strength)
        self._time_independent_matrix = tensor_to_matrix(self._time_independent_tensor, self.system_shape)

    @property
    def time_independent_tensor(self):
        return self._time_independent_tensor

    @property
    def time_independent_matrix(self):
        return self._time_independent_matrix

    def apply_time_dependence(self, arr: np.ndarray, t: float = None) -> np.ndarray:
        if self.time_dependence is None:
            return arr
        elif t is not None:
            return arr * self.time_dependence(t)
        else:
            raise "time not provided for time dependent term"

    def get_time_dependent_tensor(self, t: float = None):
        return self.apply_time_dependence(self.time_independent_tensor, t)

    def get_time_dependent_matrix(self, t: float = None):
        return self.apply_time_dependence(self.time_independent_matrix, t)


class FreeFermionHamiltonian:
    def __init__(self, system_shape: Tuple[int, ...]):
        self.terms = {}
        self.system_shape = system_shape

    def add_term(self,
                 name: str,
                 strength: Union[float, List[float]],
                 sublattice1: int,
                 sublattice2: int,
                 site_offset: Union[int, Tuple[int, ...]],
                 time_dependence: Optional[Callable] = None):
        """Adds a term sum_on_j{i*strength_j*c_j^sublattice1*c_j+site_offset^sublattice2}
        This term is symmetricized as
        1/2*sum_on_j{i*strength_j*c_j^sublattice1*c_j+site_offset^sublattice2}
        - 1/2*sum_on_j{i*strength_j*c_j+site_offset^sublattice2*c_j^sublattice1}
        The matrix form of the Hamiltonian is defined as H=i/4*c^T*M*c, therefore to M we add
        M_j^sublattice1_j+site_offset^sublattice2 += 2*strength_j
        M_j+site_offset^sublattice2_j^sublattice1 += -2*strength_j
        """
        self.terms[name] = HamiltonianTerm(strength, sublattice1, sublattice2, site_offset, self.system_shape,
                                           time_dependence)

    def get_tensor(self, t=None):
        return sum([x.get_time_dependent_tensor(t) for x in self.terms.values()])

    def get_matrix(self, t=None):
        return sum([x.get_time_dependent_matrix(t) for x in self.terms.values()])

    def dcdt(self, t, c):
        return np.dot(self.get_matrix(t), c.reshape(-1, 1)).reshape(-1)

    def evolve_single_fermion(self, c0, integration_params, t0, tf):
        ode = complex_ode(self.dcdt)
        ode.set_integrator(**integration_params)
        ode.set_initial_value(c0, t0)
        ode.integrate(tf)
        return ode.y

    def full_cycle_unitary(self, integration_params, t0, tf):
        c_basis = np.eye(self.system_shape[0] * self.system_shape[1], dtype=complex)
        evolve_vectorize = np.vectorize(self.evolve_single_fermion, signature='(m),(),(),()->(m)')
        return evolve_vectorize(c_basis, integration_params=integration_params, t0=t0, tf=tf).T

    def get_ground_state(self, t: float = None):
        M = self.get_matrix(t)
        e,Q = eigh(1j*M)
        S = 1j * Q @ np.diag(np.sign(e)) @ Q.conj().T
        return SingleParticleDensityMatrix(system_shape=self.system_shape, matrix=S)


class SingleParticleDensityMatrix:
    def __init__(self, system_shape: Tuple[int, ...], matrix: np.ndarray = None):
        self.system_shape = system_shape
        self._matrix = matrix

    @property
    def matrix(self):
        return self._matrix

    def randomize(self):
        self._matrix = np.zeros(get_system_matrix_shape(self.system_shape))
        v = np.random.rand(self._matrix.shape[0] // 2)
        add_to_diag(self._matrix[1::2, :-1:2], v)
        add_to_diag(self._matrix[:-1:2, 1::2], -v)
        O = special_ortho_group.rvs(dim=self._matrix.shape[0])
        self._matrix = O @ self._matrix @ O.T

    def evolve_with_unitary(self, Ud: np.ndarray):
        self._matrix = Ud @ self._matrix @ Ud.conj().T

    def reset(self, sublattice1: int, sublattice2: int, site1: int, site2: int):
        """resets i*c^sublattice1_site1*c^sublattice2_site2 -> 1"""
        flat_idx1 = site_and_sublattice_to_flat_index(site1, sublattice1, self.system_shape)
        flat_idx2 = site_and_sublattice_to_flat_index(site2, sublattice2, self.system_shape)
        self._matrix[:, [flat_idx1, flat_idx2]] = 0
        self._matrix[[flat_idx1, flat_idx2], :] = 0
        self._matrix[np.ix_([flat_idx1, flat_idx2], [flat_idx1, flat_idx2])] = np.array([[0, 1], [-1, 0]])

    def get_energy(self, hamiltonian_matrix: np.ndarray):
        return 1/4*np.trace(self.matrix @ hamiltonian_matrix.T)


def get_g(t):
    return np.minimum(g0 * np.ones_like(t), (1 - np.abs(2 * t / T - 1)) * T / (2 * t1) * g0)


def get_B(t):
    return np.maximum(B1 * np.ones_like(t), B0 + (B1 - B0) * t / (T - t1))


num_sites = 10
num_sublattices = 4
system_shape = (num_sites, num_sublattices, num_sites, num_sublattices)
h = 1
J = 0.7
g0 = 0.25
B1 = 0.7
B0 = 5.
T = 20.
t1 = T / 4
u = np.ones(num_sites - 1)

hamiltonian = FreeFermionHamiltonian(system_shape)
hamiltonian.add_term(name='h', strength=h, sublattice1=1, sublattice2=0, site_offset=0, time_dependence=None)
hamiltonian.add_term(name='J', strength=-J * u, sublattice1=1, sublattice2=0, site_offset=1, time_dependence=None)
hamiltonian.add_term(name='g', strength=-1, sublattice1=2, sublattice2=0, site_offset=0, time_dependence=get_g)
hamiltonian.add_term(name='B', strength=-1, sublattice1=2, sublattice2=3, site_offset=0, time_dependence=get_B)

decoupled_hamiltonian = FreeFermionHamiltonian(system_shape)
decoupled_hamiltonian.add_term(name='h', strength=h, sublattice1=1, sublattice2=0, site_offset=0, time_dependence=None)
decoupled_hamiltonian.add_term(name='J', strength=-J * u, sublattice1=1, sublattice2=0, site_offset=1, time_dependence=None)
decoupled_hamiltonian_matrix = decoupled_hamiltonian.get_matrix()
ground_state = decoupled_hamiltonian.get_ground_state()

integration_params = dict(name='vode', nsteps=2000, rtol=1e-6, atol=1e-12)


Ud = hamiltonian.full_cycle_unitary(integration_params, 0, T)

S = SingleParticleDensityMatrix(system_shape)
S.randomize()
for _ in range(50):
    print(S.get_energy(decoupled_hamiltonian_matrix))
    # plt.imshow(np.real(S.matrix))
    # plt.show()
    S.evolve_with_unitary(Ud)
    for i in range(num_sites):
        S.reset(2,3,i,i)
print(ground_state.get_energy(decoupled_hamiltonian_matrix))