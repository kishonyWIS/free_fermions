from typing import Union, Callable, Optional
import numpy as np
from scipy.integrate import complex_ode
from scipy.linalg import eigh
from scipy.stats import special_ortho_group
from scipy.linalg import expm
from scipy.integrate import solve_ivp


# TODO: trotterize
# TODO: add noise during evolution


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


def site_and_sublattice_to_flat_index(site: int, sublattice: int, system_shape: tuple[int]) -> int:
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
        return 1/4*np.trace(self.matrix @ hamiltonian_matrix.T).real


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
                 gauge_site_offset: Union[int, tuple[int]] = None,
                 dt: float = None):
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
        self.dt = dt
        self.strength = strength

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, new_strength: Union[float, list[float]]):
        self._strength = new_strength
        self._time_independent_tensor = np.zeros(self.system_shape, dtype=complex)
        if self.gauge_field is None:
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
    def time_independent_tensor(self) -> np.ndarray:
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
    def time_independent_matrix(self) -> np.ndarray:
        return tensor_to_matrix(self.time_independent_tensor, self.system_shape)

    # def small_unitary(self, t) -> np.ndarray:
    #     small_U = np.eye(get_system_matrix_shape(self.system_shape)[0], dtype=complex)
    #     small_U = matrix_to_tensor(small_U, self.system_shape)
    #     if self.gauge_field is None:
    #         if self.gauge_field is None:
    #             if self.site1 is not None and self.site2 is not None:
    #                 M = np.ndarray()
    #                 self._time_independent_tensor[self.site1, self.sublattice1, self.site2, self.sublattice2] += \
    #                     2 * self._strength
    #                 self._time_independent_tensor[self.site2, self.sublattice2, self.site1, self.sublattice1] -= \
    #                     2 * self._strength
    #             elif self.site_offset is not None:
    #                 add_to_diag(
    #                     self._time_independent_tensor[:self.system_shape[0] - self.site_offset, self.sublattice1,
    #                     self.site_offset:,
    #                     self.sublattice2], 2 * self._strength)
    #                 add_to_diag(
    #                     self._time_independent_tensor[self.site_offset:, self.sublattice2,
    #                     :self.system_shape[2] - self.site_offset,
    #                     self.sublattice1], -2 * self._strength)
    #     return expm(self.get_time_dependent_matrix(t) * self.dt)

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


class FreeFermionHamiltonian:
    def __init__(self, system_shape: tuple[int, ...], dt: float = None):
        self.terms = {}
        self.system_shape = system_shape
        self.dt = dt

    def add_term(self, name: str, **kwargs):
        """Adds a term sum_on_j{i*strength_j*c_j^sublattice1*c_j+site_offset^sublattice2}
        This term is symmetricized as
        1/2*sum_on_j{i*strength_j*c_j^sublattice1*c_j+site_offset^sublattice2}
        - 1/2*sum_on_j{i*strength_j*c_j+site_offset^sublattice2*c_j^sublattice1}
        The matrix form of the Hamiltonian is defined as H=i/4*c^T*M*c, therefore to M we add
        M_j^sublattice1_j+site_offset^sublattice2 += 2*strength_j
        M_j+site_offset^sublattice2_j^sublattice1 += -2*strength_j
        """
        self.terms[name] = HamiltonianTerm(system_shape=self.system_shape, dt=self.dt, **kwargs)

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

    def dcdt(self, t: float, c: np.ndarray) -> np.ndarray:
        return np.dot(self.get_matrix(t), c.reshape(-1, 1)).reshape(-1)

    def dcdt_faster(self, t: float, c: np.ndarray) -> np.ndarray:
        M = self.get_matrix(t)
        return (M @ c.reshape(M.shape[1], -1)).reshape(-1)

    def evolve_single_fermion(self, c0: np.ndarray, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        ode = complex_ode(self.dcdt)
        ode.set_integrator(**integration_params)
        ode.set_initial_value(c0, t0)
        ode.integrate(tf)
        return ode.y

    def evolve_single_fermion_faster(self, c0: np.ndarray, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        ode = complex_ode(self.dcdt_faster)
        ode.set_integrator(**integration_params)
        ode.set_initial_value(c0, t0)
        ode.integrate(tf)
        return ode.y

    def evolve_single_fermion_ivp(self, c0: np.ndarray, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        sol = solve_ivp(self.dcdt_faster, (t0,tf), c0, vectorized=True, dense_output=True, **integration_params)
        return sol.y[:, -1]

    def full_cycle_unitary(self, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        if np.all([x.time_dependence is None for x in self.terms.values()]):
            return expm(self.get_matrix() * (tf-t0))
        c_basis = np.eye(self.system_shape[0] * self.system_shape[1], dtype=complex)
        evolve_vectorize = np.vectorize(self.evolve_single_fermion, signature='(m),(),(),()->(m)')
        return evolve_vectorize(c_basis, integration_params=integration_params, t0=t0, tf=tf).T

    def full_cycle_unitary_faster(self, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        if np.all([x.time_dependence is None for x in self.terms.values()]):
            return expm(self.get_matrix() * (tf-t0))
        c_basis = np.eye(self.system_shape[0] * self.system_shape[1], dtype=complex).reshape(-1)
        return self.evolve_single_fermion_faster(c_basis, integration_params=integration_params, t0=t0, tf=tf).reshape(
            self.system_shape[0] * self.system_shape[1], self.system_shape[0] * self.system_shape[1])

    def full_cycle_unitary_ivp(self, integration_params: dict, t0: float, tf: float) -> np.ndarray:
        if np.all([x.time_dependence is None for x in self.terms.values()]):
            return expm(self.get_matrix() * (tf-t0))
        c_basis = np.eye(self.system_shape[0] * self.system_shape[1], dtype=complex).reshape(-1)
        return self.evolve_single_fermion_ivp(c_basis, integration_params=integration_params, t0=t0, tf=tf).reshape(
            self.system_shape[0] * self.system_shape[1], self.system_shape[0] * self.system_shape[1])

    def full_cycle_unitary_trotterize(self, t0: float, tf: float, dt: Optional[float] = None, rtol: Optional[float] = None) -> np.ndarray:
        if rtol is None:
            if dt is not None:
                self.dt = dt
            Ud = self._full_cycle_unitary_trotterize_run(t0, tf)
        else:
            if dt is not None:
                self.dt = dt
            else:
                self.dt = 1.
            Ud_prev = np.full((self.system_shape[0] * self.system_shape[1],
                               self.system_shape[0] * self.system_shape[1]), np.inf, dtype=complex)
            relative_error = np.inf
            while relative_error > rtol:
                self.dt = self.dt/5.
                Ud = self._full_cycle_unitary_trotterize_run(t0, tf)
                relative_error = np.sum(np.abs(Ud-Ud_prev)) / np.sum(np.abs(Ud))
                Ud_prev = Ud
        return Ud

    def _full_cycle_unitary_trotterize_run(self, t0, tf):
        Ud = np.eye(self.system_shape[0] * self.system_shape[1], dtype=complex)
        for t in np.arange(0, int((tf - t0) / self.dt) + 1) * self.dt + t0:
            for term in self.terms.values():
                Ud = term.small_unitary(t) @ Ud
        return Ud

    def get_ground_state(self, t: float = None) -> SingleParticleDensityMatrix:
        M = self.get_matrix(t)
        e,Q = eigh(1j*M)
        S = 1j * Q @ np.diag(np.sign(e)) @ Q.conj().T
        return SingleParticleDensityMatrix(system_shape=self.system_shape, matrix=S)


def get_fermion_bilinear_unitary(system_shape: tuple[int,...],
                                 sublattice1: int, sublattice2: int, site1: int, site2: int, integration_params: dict):
    H = FreeFermionHamiltonian(system_shape)
    H.add_term(name='fermion_bilinear', strength=1, sublattice1=sublattice1, sublattice2=sublattice2, site1=site1, site2=site2)
    return H.full_cycle_unitary(integration_params, 0, np.pi / 2)


def fidelity(rho1,rho2):
    return (np.trace(np.sqrt(np.sqrt(rho1) @ rho2 @ np.sqrt(rho1))))**2.
