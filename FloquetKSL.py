from time import time
from typing import Callable

import numpy as np
import scipy

from free_fermion_hamiltonian import MajoranaFreeFermionHamiltonian, MajoranaSingleParticleDensityMatrix, get_fermion_bilinear_unitary
from scipy.linalg import eigh
from matplotlib import pyplot as plt


def get_J(t,T):
    # periodic function with period T applying a plus between t=0 and t=T/3
    return int(t/T - int(t/T) < 1/3)


def get_floquet_KSL_model(num_sites_x, num_sites_y, J):
    num_sublattices = 2
    system_shape = (num_sites_x, num_sites_y, num_sublattices, num_sites_x, num_sites_y, num_sublattices)

    hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)

    site_offset_x = (0, 0)
    site_offset_y = (1, 0)
    site_offset_z = (0, 1)
    for site1 in np.ndindex(tuple(system_shape[0:len(system_shape)//2-1] - np.array(site_offset_x))):
        site2 = tuple(np.array(site1) + np.array(site_offset_x))
        hamiltonian.add_term(name=f'Jx_{site1}', strength=J, sublattice1=0, sublattice2=1, site1=site1, site2=site2, time_dependence = None)
    for site1 in np.ndindex(tuple(system_shape[0:len(system_shape)//2-1] - np.array(site_offset_y))):
        site2 = tuple(np.array(site1) + np.array(site_offset_y))
        hamiltonian.add_term(name=f'Jy_{site1}', strength=J, sublattice1=1, sublattice2=0, site1=site1, site2=site2, time_dependence = None)
    for site1 in np.ndindex(tuple(system_shape[0:len(system_shape)//2-1] - np.array(site_offset_z))):
        site2 = tuple(np.array(site1) + np.array(site_offset_z))
        hamiltonian.add_term(name=f'Jz_{site1}', strength=J, sublattice1=1, sublattice2=0, site1=site1, site2=site2, time_dependence = None)

    return hamiltonian


if __name__ == "__main__":
    hamiltonian = get_floquet_KSL_model(10, 10, 1)
    integration_params = dict(name='vode', nsteps=2000, rtol=1e-6, atol=1e-10)
    print(hamiltonian.full_cycle_unitary_faster(integration_params, 0, 1))
    plt.plot(hamiltonian.get_excitation_spectrum())
    plt.show()
