from abc import ABCMeta, abstractmethod
from typing import Union, Callable, Optional
import numpy as np
from scipy.integrate import ode, complex_ode
from scipy.linalg import eigh
from scipy.stats import special_ortho_group
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from scipy import sparse


def add_to_diag(arr: np.ndarray, to_add: Union[int, list]):
    idx = np.diag_indices_from(arr)
    arr[idx] += to_add


def tensor_to_matrix(tensor: np.ndarray, system_shape: tuple[int, ...]) -> np.ndarray:
    matrix_shape = get_system_matrix_shape(system_shape)
    return tensor.reshape(matrix_shape[0], matrix_shape[1])


def matrix_to_tensor(matrix: np.ndarray, system_shape: tuple[int, ...]) -> np.ndarray:
    return matrix.reshape(system_shape)


def get_system_matrix_shape(system_shape: tuple[int, ...]):
    num_dims = len(system_shape)
    shape1 = np.prod(system_shape[:num_dims // 2])
    shape2 = np.prod(system_shape[num_dims // 2:])
    return shape1, shape2


def site_and_sublattice_to_flat_index(site: np.ndarray[int], sublattice: int, system_shape: tuple[int]) -> int:
    if isinstance(site, int):
        site = np.array([site])
    return np.ravel_multi_index((*site, sublattice), system_shape[:len(system_shape) // 2])


class SingleParticleDensityMatrix(metaclass=ABCMeta):
    def __init__(self, system_shape: tuple[int, ...], matrix: np.ndarray = None, tensor: np.ndarray = None):
        self.system_shape = system_shape
        self.matrix = None
        if matrix is not None:
            self.matrix = matrix
        if tensor is not None:
            self.tensor = tensor

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, new_matrix: np.ndarray):
        self._matrix = new_matrix

    @property
    def tensor(self) -> np.ndarray:
        return matrix_to_tensor(self.matrix, self.system_shape)

    @tensor.setter
    def tensor(self, new_tensor: np.ndarray):
        self._matrix = tensor_to_matrix(new_tensor, self.system_shape)

    @abstractmethod
    def randomize(self):
        pass

    def evolve_with_unitary(self, Ud: np.ndarray):
        self._matrix = Ud.conj() @ self._matrix @ Ud.T

    @abstractmethod
    def reset(self, sublattice1, sublattice2, site1, site2):
        pass

    def get_energy(self, hamiltonian_matrix: np.ndarray):
        return 1 / 4 * np.trace(self.matrix @ hamiltonian_matrix.T).real


class MajoranaSingleParticleDensityMatrix(SingleParticleDensityMatrix):
    '''This is the expectation value of i*c_a*c_b'''
    def randomize(self):
        self._matrix = np.zeros(get_system_matrix_shape(self.system_shape))
        v = np.random.rand(self._matrix.shape[0] // 2)
        add_to_diag(self._matrix[1::2, :-1:2], v)
        add_to_diag(self._matrix[:-1:2, 1::2], -v)
        O = special_ortho_group.rvs(dim=self._matrix.shape[0])
        self._matrix = O @ self._matrix @ O.T

        self._matrix = np.zeros(get_system_matrix_shape(self.system_shape))

    def reset(self, sublattice1: int, sublattice2: int, site1: np.ndarray[int], site2: np.ndarray[int]):
        """resets i*c^sublattice1_site1*c^sublattice2_site2 -> 1"""
        flat_idx1 = site_and_sublattice_to_flat_index(site1, sublattice1, self.system_shape)
        flat_idx2 = site_and_sublattice_to_flat_index(site2, sublattice2, self.system_shape)
        self._matrix[:, [flat_idx1, flat_idx2]] = 0
        self._matrix[[flat_idx1, flat_idx2], :] = 0
        # NOTE: there should be 1j on the diagonal (at all times) but having it be zero instead allows the matrix to be
        # real and does not effect further time evolution or resets.
        self._matrix[np.ix_([flat_idx1, flat_idx2], [flat_idx1, flat_idx2])] = np.array([[0, 1], [-1, 0]])


class ComplexSingleParticleDensityMatrix(SingleParticleDensityMatrix):
    '''This is the expectation value of c^dag_a*c_b'''
    def randomize(self):
        self._matrix = 0.5*np.eye(get_system_matrix_shape(self.system_shape)[0], dtype=complex)

    def reset(self, sublattice1: int, sublattice2: int, site1: np.ndarray[int], site2: np.ndarray[int]):
        """resets c^dag^sublattice1_site1*c^sublattice2_site2 -> 0.5i"""
        flat_idx1 = site_and_sublattice_to_flat_index(site1, sublattice1, self.system_shape)
        flat_idx2 = site_and_sublattice_to_flat_index(site2, sublattice2, self.system_shape)
        self._matrix[:, [flat_idx1, flat_idx2]] = 0
        self._matrix[[flat_idx1, flat_idx2], :] = 0
        self._matrix[np.ix_([flat_idx1, flat_idx2], [flat_idx1, flat_idx2])] = np.array([[0.5, 0.5*1j], [-0.5*1j, 0.5]])


class FreeFermionHamiltonianTerm(metaclass=ABCMeta):
    def __init__(self,
                 strength: Union[float, list[float], complex],
                 sublattice1: int,
                 sublattice2: int,
                 system_shape: tuple[int, ...],
                 site1: Optional[Union[int, tuple[int, ...], np.ndarray[int]]] = None,
                 site2: Optional[Union[int, tuple[int, ...], np.ndarray[int]]] = None,
                 site_offset: Optional[Union[int, tuple[int, ...], np.ndarray[int]]] = None,
                 time_dependence: Optional[Callable] = None,
                 gauge_field: Optional[MajoranaSingleParticleDensityMatrix] = None,
                 gauge_sublattice1: Optional[int] = None,
                 gauge_sublattice2: Optional[int] = None,
                 gauge_site_offset: Optional[Union[int, tuple[int, ...], np.ndarray[int]]] = None,
                 dt: float = None,
                 periodic_bc=False):
        self.periodic_bc = periodic_bc
        self.system_shape = system_shape
        self.site_offset = np.array(site_offset) if site_offset is not None else None
        self.sublattice1 = sublattice1
        self.sublattice2 = sublattice2
        self.site1 = np.array(site1) if site1 is not None else None
        self.site2 = np.array(site2) if site2 is not None else None
        if self.site_offset is not None:
            self.site1, self.site2 = self.offset_to_site1_site2(self.site_offset)
        self.time_dependence = time_dependence
        self.gauge_field = gauge_field
        self.gauge_sublattice1 = gauge_sublattice1
        self.gauge_sublattice2 = gauge_sublattice2
        self.gauge_site_offset = np.array(gauge_site_offset) if gauge_site_offset is not None else None
        if self.gauge_site_offset is not None:
            self.gauge_site1, self.gauge_site2 = self.offset_to_site1_site2(self.gauge_site_offset)
        self.dt = dt
        self.strength = strength

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, new_strength: Union[float, np.ndarray[float]]):
        self._strength = new_strength
        self._time_independent_tensor = self.get_zeros_tensor()
        if self.gauge_field is None:
            self._time_independent_tensor[(*self.site1, self.sublattice1, *self.site2, self.sublattice2)] += \
                2 * self._strength
            self._time_independent_tensor[(*self.site2, self.sublattice2, *self.site1, self.sublattice1)] += \
                2 * self.get_transposed_value(self._strength)

    @abstractmethod
    def get_zeros_tensor(self):
        pass

    @abstractmethod
    def get_transposed_value(self, value: Union[float, list, complex]) -> Union[float, list, complex]:
        pass

    @property
    def time_independent_tensor(self) -> np.ndarray:
        if self.gauge_field is not None:
            # Setting strength to be a gauge field
            self._time_independent_tensor = self.get_zeros_tensor()
            self._time_independent_tensor[(*self.site1, self.sublattice1, *self.site2, self.sublattice2)] += \
                2 * self._strength * self.gauge_field.tensor[(*self.gauge_site1, self.gauge_sublattice1,
                                                              *self.gauge_site2, self.gauge_sublattice2)]
            self._time_independent_tensor[(*self.site2, self.sublattice2, *self.site1, self.sublattice1)] += \
                2 * self.get_transposed_value(self._strength * self.gauge_field.tensor[(*self.gauge_site1,
                                                                                            self.gauge_sublattice1,
                                                                                            *self.gauge_site2,
                                                                                            self.gauge_sublattice2)])
        return self._time_independent_tensor

    @property
    def time_independent_matrix(self) -> np.ndarray:
        return tensor_to_matrix(self.time_independent_tensor, self.system_shape)

    def apply_time_dependence(self, arr: np.ndarray, t: float = None) -> np.ndarray:
        if self.time_dependence is None:
            return arr
        elif t is not None:
            return arr * self.time_dependence(t)
        else:
            raise "time not provided for time dependent term"

    def get_time_dependent_tensor(self, t: float = None) -> np.ndarray:
        return self.apply_time_dependence(self.time_independent_tensor, t)

    def get_time_dependent_matrix(self, t: float = None) -> np.ndarray:
        return self.apply_time_dependence(self.time_independent_matrix, t)

    def offset_to_site1_site2(self, offset):
        if self.periodic_bc:
            sites1 = np.meshgrid(*(np.arange(self.system_shape[dim]) for dim in range(len(offset))),
                                 indexing='ij')
        else:
            sites1 = np.meshgrid(*(np.arange(self.system_shape[dim] - np.abs(offset[dim])) for dim in range(len(offset))),
                                 indexing='ij')
        sites1 = [(sites1[dim] + np.abs(offset[dim])*(offset[dim]<0)) % self.system_shape[dim] for dim in range(len(offset))]
        sites2 = [(sites1[dim] + offset[dim]) % self.system_shape[dim] for dim in range(len(offset))]
        return sites1, sites2

    def filter_site1(self, filter_func):
        idx_to_keep = filter_func(self.site1)
        for dim in range(len(self.site1)):
            self.site1[dim] = self.site1[dim][idx_to_keep]
            self.site2[dim] = self.site2[dim][idx_to_keep]
            if self.gauge_field is not None:
                self.gauge_site1[dim] = self.gauge_site1[dim][idx_to_keep]
                self.gauge_site2[dim] = self.gauge_site2[dim][idx_to_keep]
        if isinstance(self.strength, np.ndarray):
            self.strength = self.strength[idx_to_keep]


class MajoranaFreeFermionHamiltonianTerm(FreeFermionHamiltonianTerm):
    """Adds a term sum_on_j{i*strength_j*c_j^sublattice1*c_j+site_offset^sublattice2}
    This term is symmetricized as
    1/2*sum_on_j{i*strength_j*c_j^sublattice1*c_j+site_offset^sublattice2}
    - 1/2*sum_on_j{i*strength_j*c_j+site_offset^sublattice2*c_j^sublattice1}
    The matrix form of the Hamiltonian is defined as H=i/4*c^T*M*c, therefore to M we add
    M_j^sublattice1_j+site_offset^sublattice2 += 2*strength_j
    M_j+site_offset^sublattice2_j^sublattice1 += -2*strength_j
    """
    def small_unitary(self, t, dt_factor=1) -> sparse.csr_matrix:
        matrix_shape = get_system_matrix_shape(self.system_shape)[0]
        time_dependence = 1 if self.time_dependence is None else self.time_dependence(t)
        cos = np.cos(2 * self.strength * time_dependence * self.dt * dt_factor)
        sin = np.sin(2 * self.strength * time_dependence * self.dt * dt_factor)
        if self.site1 is not None and self.site2 is not None and self.site_offset is None:
            i1 = site_and_sublattice_to_flat_index(self.site1, self.sublattice1, self.system_shape)
            i2 = site_and_sublattice_to_flat_index(self.site2, self.sublattice2, self.system_shape)
            small_U_2by2 = np.array([[cos, sin], [-sin, cos]])
            small_U = sparse.eye(matrix_shape, format='lil')
            small_U[np.ix_([i1, i2], [i1, i2])] = small_U_2by2
        elif self.site_offset is not None:
            small_U = sparse.eye(matrix_shape, format='lil')
            i1 = site_and_sublattice_to_flat_index(self.site1, self.sublattice1, self.system_shape)
            i2 = site_and_sublattice_to_flat_index(self.site2, self.sublattice2, self.system_shape)
            if self.gauge_field is None:
                gauge = 1
            else:
                gauge = self.gauge_field.tensor[(*self.gauge_site1, self.gauge_sublattice1,
                                                    *self.gauge_site2, self.gauge_sublattice2)]
            small_U[i1,i1] = cos
            small_U[i2,i2] = cos
            small_U[i1,i2] = sin * gauge
            small_U[i2,i1] = -sin * gauge
        return small_U

    def get_zeros_tensor(self):
        return np.zeros(self.system_shape)

    def get_transposed_value(self, value: Union[float, list, complex]) -> Union[float, list, complex]:
        return -value


class ComplexFreeFermionHamiltonianTerm(FreeFermionHamiltonianTerm):
    """Adds a term sum_on_j{strength_j*c^dag_j^sublattice1*c_j+site_offset^sublattice2} + h.c.
    The matrix form of the Hamiltonian is defined as H=1/4*c^dag*M*c, therefore to M we add
    M_j^sublattice1_j+site_offset^sublattice2 += 2*strength_j
    M_j+site_offset^sublattice2_j^sublattice1 += -2*strength_j
    """
    def get_zeros_tensor(self):
        return np.zeros(self.system_shape, dtype=complex)

    def get_transposed_value(self, value: Union[float, list, complex]) -> Union[float, list, complex]:
        return np.conj(value)


class FreeFermionHamiltonian(metaclass=ABCMeta):
    def __init__(self, system_shape: tuple[int, ...], dt: float = None):
        self.terms = {}
        self.system_shape = system_shape
        self.dt = dt

    @abstractmethod
    def add_term(self, name: str, **kwargs):
        pass

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value
        for term in self.terms.values():
            term.dt = value

    def get_tensor(self, t: float = None) -> np.ndarray:
        return sum([x.get_time_dependent_tensor(t) for x in self.terms.values()])

    def get_matrix(self, t: float = None) -> np.ndarray:
        return sum([x.get_time_dependent_matrix(t) for x in self.terms.values()])

    @abstractmethod
    def get_ode(self, dcdt):
        pass

    def evolve_single_fermion(self, c0: np.ndarray, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        ode_instance = self.get_ode(self.dcdt)
        ode_instance.set_integrator(**integration_params)
        ode_instance.set_initial_value(c0, t0)
        ode_instance.integrate(tf)
        return ode_instance.y

    def evolve_single_fermion_faster(self, c0: np.ndarray, integration_params: dict, t0: float,
                                     tf: float) -> np.ndarray:
        ode_instance = self.get_ode(self.dcdt_faster)
        ode_instance.set_integrator(**integration_params)
        ode_instance.set_initial_value(c0, t0)
        ode_instance.integrate(tf)
        return ode_instance.y

    def evolve_single_fermion_ivp(self, c0: np.ndarray, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        sol = solve_ivp(self.dcdt_faster, (t0, tf), c0, vectorized=True, dense_output=True, **integration_params)
        return sol.y[:, -1]

    @abstractmethod
    def dcdt(self, t, c):
        pass

    @abstractmethod
    def dcdt_faster(self, t, c):
        pass

    def full_cycle_unitary(self, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        if np.all([x.time_dependence is None for x in self.terms.values()]):
            return expm(self.get_matrix() * (tf - t0))
        c_basis = np.eye(self.system_shape[0] * self.system_shape[1])
        evolve_vectorize = np.vectorize(self.evolve_single_fermion, signature='(m),(),(),()->(m)')
        return evolve_vectorize(c_basis, integration_params=integration_params, t0=t0, tf=tf).T

    def full_cycle_unitary_faster(self, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        if np.all([x.time_dependence is None for x in self.terms.values()]):
            return expm(self.get_matrix() * (tf - t0))
        c_basis = np.eye(np.prod(self.system_shape[:len(self.system_shape)//2])).reshape(-1)
        return self.evolve_single_fermion_faster(c_basis, integration_params=integration_params, t0=t0, tf=tf).reshape(
            *get_system_matrix_shape(self.system_shape))

    def full_cycle_unitary_ivp(self, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        if np.all([x.time_dependence is None for x in self.terms.values()]):
            return expm(self.get_matrix() * (tf - t0))
        c_basis = np.eye(self.system_shape[0] * self.system_shape[1]).reshape(-1)
        return self.evolve_single_fermion_ivp(c_basis, integration_params=integration_params, t0=t0, tf=tf).reshape(
            *get_system_matrix_shape(self.system_shape))

    @abstractmethod
    def get_ground_state(self, t):
        pass

    @abstractmethod
    def get_excitation_spectrum(self, t):
        pass


class MajoranaFreeFermionHamiltonian(FreeFermionHamiltonian):
    def add_term(self, name: str, **kwargs):
        self.terms[name] = MajoranaFreeFermionHamiltonianTerm(system_shape=self.system_shape, dt=self.dt, **kwargs)

    def get_ode(self, dcdt):
        return ode(dcdt)

    def dcdt(self, t: float, c: np.ndarray) -> np.ndarray:
        return np.dot(self.get_matrix(t), c.reshape(-1, 1)).reshape(-1)

    def dcdt_faster(self, t: float, c: np.ndarray) -> np.ndarray:
        M = self.get_matrix(t)
        return (M @ c.reshape(M.shape[1], -1)).reshape(-1)

    def full_cycle_unitary_trotterize(self, t0: float, tf: float, steps: Optional[float] = None,
                                      atol: Optional[float] = None) -> np.ndarray:
        if atol is None:
            if steps is not None:
                self.dt = (tf - t0) / steps
            Ud = self._full_cycle_unitary_trotterize_run(t0, tf)
        else:
            if steps is not None:
                self.dt = (tf - t0) / steps
            else:
                self.dt = (tf - t0) / 100
            Ud_prev = np.full((self.system_shape[0] * self.system_shape[1],
                               self.system_shape[0] * self.system_shape[1]), 100)
            relative_error = np.inf
            while relative_error > atol:
                self.dt = self.dt / 5.
                Ud = self._full_cycle_unitary_trotterize_run(t0, tf)
                relative_error = np.mean(np.abs(np.eye(Ud.shape[0]) - Ud @ Ud_prev.conj().T))
                Ud_prev = Ud
        return Ud

    def _full_cycle_unitary_trotterize_run(self, t0, tf):
        Ud = np.eye(np.prod(self.system_shape[:len(self.system_shape)//2]))
        for t in np.arange(0, int((tf - t0) / self.dt)) * self.dt + t0:
            Ud = self._unitary_trotterize_run_step(Ud, t)
        return Ud

    def _unitary_trotterize_run_step(self, Ud, t):
        for name, term in self.terms.items():
            Ud = term.small_unitary(t + self.dt / 2) @ Ud
        return Ud

    def get_ground_state(self, t: float = None) -> MajoranaSingleParticleDensityMatrix:
        M = self.get_matrix(t)
        e, Q = eigh(1j * M)
        S = 1j * Q @ np.diag(np.sign(e)) @ Q.conj().T
        return MajoranaSingleParticleDensityMatrix(system_shape=self.system_shape, matrix=S)

    def get_excitation_spectrum(self, t: float = None) -> np.ndarray:
        M = self.get_matrix(t)
        e, _ = eigh(1j * M)
        return e


class ComplexFreeFermionHamiltonian(FreeFermionHamiltonian):
    def add_term(self, name: str, **kwargs):
        self.terms[name] = ComplexFreeFermionHamiltonianTerm(system_shape=self.system_shape, dt=self.dt, **kwargs)

    def get_ode(self, dcdt):
        return complex_ode(dcdt)

    def dcdt(self, t: float, c: np.ndarray) -> np.ndarray:
        return 1j/4*np.dot(self.get_matrix(t), c.reshape(-1, 1)).reshape(-1)

    def dcdt_faster(self, t: float, c: np.ndarray) -> np.ndarray:
        M = self.get_matrix(t)
        return 1j/4*(M @ c.reshape(M.shape[1], -1)).reshape(-1)

    def get_ground_state(self, t: float = None) -> ComplexSingleParticleDensityMatrix:
        M = self.get_matrix(t)
        e, Q = eigh(M)
        S = Q.conj() @ np.diag(e < 0) @ Q.T
        return ComplexSingleParticleDensityMatrix(system_shape=self.system_shape, matrix=S)

    def get_excitation_spectrum(self, t: float = None) -> np.ndarray:
        M = self.get_matrix(t)
        e, _ = eigh(1/4*M)
        return e


def get_fermion_bilinear_unitary(system_shape: tuple[int, ...],
                                 sublattice1: int, sublattice2: int, site1: int, site2: int):
    term = MajoranaFreeFermionHamiltonianTerm(system_shape=system_shape, strength=1, sublattice1=sublattice1,
                                              sublattice2=sublattice2,
                                              site1=site1, site2=site2, dt=np.pi / 2)
    return term.small_unitary(0)


def fidelity(rho1, rho2):
    return (np.trace(np.sqrt(np.sqrt(rho1) @ rho2 @ np.sqrt(rho1)))) ** 2.
