from KSL_model import KSLHamiltonian, KSLState
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from FloquetKSL import hexagonal_lattice_site_to_x_y

def get_energy_on_site(full_tensor:np.ndarray, S:MajoranaSingleParticleDensityMatrix, site:tuple[int]):
    partial_tensor = np.zeros_like(full_tensor)
    partial_tensor[site[0], site[1], :, :, :, :] =\
        full_tensor[site[0], site[1], :, :, :, :]
    partial_tensor[:, :, :, site[0], site[1], :] =\
        full_tensor[:, :, :, site[0], site[1], :]
    partial_matrix = tensor_to_matrix(partial_tensor, partial_tensor.shape)
    ground_state_energy = get_ground_state_energy(partial_matrix, S.system_shape)
    return S.get_energy(partial_matrix), np.linalg.norm(partial_matrix), ground_state_energy

def get_spacial_energy_density(hamiltonian:MajoranaFreeFermionHamiltonian, S:MajoranaSingleParticleDensityMatrix, t=None):
    full_tensor = hamiltonian.get_tensor(t)
    num_sites_x, num_sites_y = hamiltonian.system_shape[0], hamiltonian.system_shape[1]
    energy_density = np.zeros((num_sites_x, num_sites_y))
    norms = np.zeros((num_sites_x, num_sites_y))
    ground_state_energy_density = np.zeros((num_sites_x, num_sites_y))
    for site_x in range(num_sites_x):
        for site_y in range(num_sites_y):
            energy_density[site_x, site_y], norms[site_x, site_y], ground_state_energy_density[site_x, site_y] = get_energy_on_site(full_tensor, S, (site_x, site_y))
    return energy_density, ground_state_energy_density, norms

def get_ground_state_energy(M:np.ndarray, system_shape:tuple[int]):
    e, Q = eigh(1j * M)
    S = 1j * Q @ np.diag(np.sign(e)) @ Q.conj().T
    return MajoranaSingleParticleDensityMatrix(system_shape=system_shape, matrix=S).get_energy(M)

def draw_energy_of_bonds(hamiltonian:MajoranaFreeFermionHamiltonian, S:MajoranaSingleParticleDensityMatrix, term_name:str):
