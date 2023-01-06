from typing import Union, Callable, Optional
import numpy as np
from scipy.integrate import complex_ode
from scipy.linalg import eigh, eig
from matplotlib import pyplot as plt
from scipy.stats import special_ortho_group

# TODO: trotterize
# TODO: add noise
# TODO: gauge field


def add_to_diag(arr: np.ndarray, to_add: Union[int, list]):
    idx = np.diag_indices_from(arr)
    arr[idx] += to_add


def tensor_to_matrix(tensor: np.ndarray, system_shape: tuple[int, ...]) -> np.ndarray:
    matrix_shape = get_system_matrix_shape(system_shape)
    return tensor.reshape(matrix_shape[0], matrix_shape[1])


def matrix_to_tensor(matrix: np.ndarray, system_shape: tuple[int, ...]) -> np.ndarray:
    return matrix.reshape(system_shape)


def get_system_matrix_shape(system_shape: tuple[int]):
    num_dims = len(system_shape)
    shape1 = np.prod(system_shape[:num_dims // 2])
    shape2 = np.prod(system_shape[num_dims // 2:])
    return shape1, shape2


def site_and_sublattice_to_flat_index(site: int, sublattice: int, system_shape: tuple[int]):
    return np.ravel_multi_index((site, sublattice), system_shape[:2])


class SingleParticleDensityMatrix:
    def __init__(self, system_shape: tuple[int, ...], matrix: np.ndarray = None, tensor: np.ndarray = None):
        self.system_shape = system_shape
        self.matrix = None
        if matrix is not None:
            self.matrix = matrix
        if tensor is not None:
            self.tensor = tensor

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix: np.ndarray):
        self._matrix = new_matrix

    @property
    def tensor(self):
        return matrix_to_tensor(self.matrix, self.system_shape)

    @tensor.setter
    def tensor(self, new_tensor: np.ndarray):
        self._matrix = tensor_to_matrix(new_tensor, self.system_shape)

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


class HamiltonianTerm:
    def __init__(self,
                 strength: Union[float, list[float]],
                 sublattice1: int,
                 sublattice2: int,
                 system_shape: tuple[int, ...],
                 site1: Optional[int] = None,
                 site2: Optional[int] = None,
                 site_offset: Union[int, tuple[int]] = None,
                 time_dependence: Optional[Callable] = None,
                 gauge_field: Optional[SingleParticleDensityMatrix] = None,
                 gauge_sublattice1: Optional[int] = None,
                 gauge_sublattice2: Optional[int] = None,
                 gauge_site_offset: Union[int, tuple[int]] = None):
        self.system_shape = system_shape
        self.site_offset = site_offset
        self.sublattice1 = sublattice1
        self.sublattice2 = sublattice2
        self.site1 = site1
        self.site2 = site2
        self.time_dependence = time_dependence
        self.gauge_field = gauge_field
        self.gauge_sublattice1 = gauge_sublattice1
        self.gauge_sublattice2 = gauge_sublattice2
        self.gauge_site_offset = gauge_site_offset
        self.strength = strength

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, new_strength: Union[float, list[float]]):
        self._strength = new_strength
        self._time_independent_tensor = np.zeros(self.system_shape, dtype=complex)
        if self.gauge_field is not None:
            return
        if self.site1 is not None and self.site2 is not None:
            self._time_independent_tensor[self.site1, self.sublattice1, self.site2, self.sublattice2] += \
                2*self._strength
            self._time_independent_tensor[self.site2, self.sublattice2, self.site1, self.sublattice1] -= \
                2*self._strength
        elif self.site_offset is not None:
            add_to_diag(
                self._time_independent_tensor[:self.system_shape[0] - self.site_offset, self.sublattice1, self.site_offset:,
                self.sublattice2], 2 * self._strength)
            add_to_diag(
                self._time_independent_tensor[self.site_offset:, self.sublattice2, :self.system_shape[2] - self.site_offset,
                self.sublattice1], -2 * self._strength)

    @property
    def time_independent_tensor(self):
        if self.gauge_field is not None:
            # Setting strength to be a gauge field
            self._time_independent_tensor = np.zeros(self.system_shape, dtype=complex)
            add_to_diag(
                self._time_independent_tensor[:self.system_shape[0] - self.site_offset, self.sublattice1,
                self.site_offset:, self.sublattice2],
                2 * self._strength * np.diag(self.gauge_field.tensor[:self.system_shape[0] - self.gauge_site_offset,
                                     self.gauge_sublattice1, self.gauge_site_offset:, self.gauge_sublattice2]))
            add_to_diag(
                self._time_independent_tensor[self.site_offset:, self.sublattice2,
                :self.system_shape[2] - self.site_offset, self.sublattice1],
                -2 * self._strength * np.diag(self.gauge_field.tensor[:self.system_shape[0] - self.gauge_site_offset,
                                      self.gauge_sublattice1, self.gauge_site_offset:, self.gauge_sublattice2]))
        return self._time_independent_tensor

    @property
    def time_independent_matrix(self):
        return tensor_to_matrix(self.time_independent_tensor, self.system_shape)

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
    def __init__(self, system_shape: tuple[int, ...]):
        self.terms = {}
        self.system_shape = system_shape

    def add_term(self,
                 name: str,
                 strength: Union[float, list[float]],
                 sublattice1: int,
                 sublattice2: int,
                 site1: Optional[int] = None,
                 site2: Optional[int] = None,
                 site_offset: Union[int, tuple[int, ...]] = None,
                 time_dependence: Optional[Callable] = None,
                 gauge_field: Optional[SingleParticleDensityMatrix] = None,
                 gauge_sublattice1: Optional[int] = None,
                 gauge_sublattice2: Optional[int] = None,
                 gauge_site_offset: Union[int, tuple[int]] = None
                 ):
        """Adds a term sum_on_j{i*strength_j*c_j^sublattice1*c_j+site_offset^sublattice2}
        This term is symmetricized as
        1/2*sum_on_j{i*strength_j*c_j^sublattice1*c_j+site_offset^sublattice2}
        - 1/2*sum_on_j{i*strength_j*c_j+site_offset^sublattice2*c_j^sublattice1}
        The matrix form of the Hamiltonian is defined as H=i/4*c^T*M*c, therefore to M we add
        M_j^sublattice1_j+site_offset^sublattice2 += 2*strength_j
        M_j+site_offset^sublattice2_j^sublattice1 += -2*strength_j
        """
        self.terms[name] = HamiltonianTerm(strength=strength, sublattice1=sublattice1, sublattice2=sublattice2,
                                           site1=site1, site2=site2, site_offset=site_offset,
                                           system_shape=self.system_shape, time_dependence=time_dependence,
                                           gauge_field=gauge_field, gauge_sublattice1=gauge_sublattice1,
                                           gauge_sublattice2=gauge_sublattice2, gauge_site_offset=gauge_site_offset
                                           )

    def get_tensor(self, t: float = None):
        return sum([x.get_time_dependent_tensor(t) for x in self.terms.values()])

    def get_matrix(self, t: float = None):
        return sum([x.get_time_dependent_matrix(t) for x in self.terms.values()])

    def dcdt(self, t: float, c: np.ndarray):
        return np.dot(self.get_matrix(t), c.reshape(-1, 1)).reshape(-1)

    def evolve_single_fermion(self, c0: np.ndarray, integration_params: dict, t0: float, tf: float):
        ode = complex_ode(self.dcdt)
        ode.set_integrator(**integration_params)
        ode.set_initial_value(c0, t0)
        ode.integrate(tf)
        return ode.y

    def full_cycle_unitary(self, integration_params: dict, t0: float, tf: float):
        c_basis = np.eye(self.system_shape[0] * self.system_shape[1], dtype=complex)
        evolve_vectorize = np.vectorize(self.evolve_single_fermion, signature='(m),(),(),()->(m)')
        return evolve_vectorize(c_basis, integration_params=integration_params, t0=t0, tf=tf).T

    def get_ground_state(self, t: float = None):
        M = self.get_matrix(t)
        e,Q = eigh(1j*M)
        S = 1j * Q @ np.diag(np.sign(e)) @ Q.conj().T
        return SingleParticleDensityMatrix(system_shape=self.system_shape, matrix=S)


def get_g(t: float):
    return np.minimum(g0 * np.ones_like(t), (1 - np.abs(2 * t / T - 1)) * T / (2 * t1) * g0)


def get_B(t: float):
    return np.maximum(B1 * np.ones_like(t), B0 + (B1 - B0) * t / (T - t1))


num_sites = 10
num_sublattices = 6
system_shape = (num_sites, num_sublattices, num_sites, num_sublattices)
non_gauge_shape = (num_sites, 4, num_sites, 4)
gauge_shape = (num_sites, 2, num_sites, 2)
non_gauge_idxs = np.ix_(range(num_sites),[0,3,4,5],range(num_sites),[0,3,4,5])
gauge_idxs = np.ix_(range(num_sites),[1,2],range(num_sites),[1,2])
h = 1
J = 0.7
g0 = 0.25
B1 = 0.7
B0 = 5.
T = 20.
t1 = T / 4

decoupled_hamiltonian = FreeFermionHamiltonian(system_shape)
decoupled_hamiltonian.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
decoupled_hamiltonian.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1)
decoupled_hamiltonian_matrix = decoupled_hamiltonian.get_matrix()
ground_state = decoupled_hamiltonian.get_ground_state()

gauge_setting_hamiltonian = FreeFermionHamiltonian(system_shape)
gauge_setting_hamiltonian.add_term(name='G', strength=-1, sublattice1=2, sublattice2=1, site_offset=1)
S_gauge = gauge_setting_hamiltonian.get_ground_state()

integration_params = dict(name='vode', nsteps=2000, rtol=1e-6, atol=1e-12)

S_non_gauge = SingleParticleDensityMatrix(non_gauge_shape)
S_non_gauge.randomize()
S0_tensor = np.zeros(system_shape, dtype=complex)
S0_tensor[non_gauge_idxs] = S_non_gauge.tensor
S0_tensor[gauge_idxs] = S_gauge.tensor[gauge_idxs]
S = SingleParticleDensityMatrix(system_shape=system_shape, tensor=S0_tensor)

hamiltonian = FreeFermionHamiltonian(system_shape)
hamiltonian.add_term(name='h', strength=h, sublattice1=3, sublattice2=0, site_offset=0)
hamiltonian.add_term(name='J', strength=-J, sublattice1=3, sublattice2=0, site_offset=1,
                     gauge_field=S, gauge_sublattice1=2, gauge_sublattice2=1, gauge_site_offset=1)
hamiltonian.add_term(name='g', strength=-1, sublattice1=4, sublattice2=0, site_offset=0, time_dependence=get_g)
hamiltonian.add_term(name='B', strength=-1, sublattice1=4, sublattice2=5, site_offset=0, time_dependence=get_B)

Ud = hamiltonian.full_cycle_unitary(integration_params, 0, T)

for _ in range(50):
    print(S.get_energy(decoupled_hamiltonian_matrix))
    # plt.imshow(np.real(S.matrix))
    # plt.show()
    S.evolve_with_unitary(Ud)
    for i in range(num_sites):
        S.reset(4,5,i,i)
print(ground_state.get_energy(decoupled_hamiltonian_matrix))