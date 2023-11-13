import numpy as np
import scipy.sparse as sparse
import pymatching
from itertools import product

class KSL_flux_corrector:
    def __init__(self, n_sites_x, n_sites_y, periodic_bc=False):
        self.n_sites_x = n_sites_x
        self.n_sites_y = n_sites_y
        self.n_plaquettes_x = n_sites_x if (periodic_bc == True or periodic_bc[0]) else n_sites_x - 1
        self.n_plaquettes_y = n_sites_y if (periodic_bc == True or periodic_bc[1]) else n_sites_y - 1
        H = np.zeros((self.n_plaquettes_x*self.n_plaquettes_y, 2*n_sites_x*n_sites_y*3))
        first_plaq_x_shift = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        first_plaq_y_shift = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1])
        first_plaq_sublattice = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        first_plaq_pauli = np.array([1, 2, 0, 2, 0, 1, 0, 1, 0, 2, 1, 2])
        for i_hex, (x_shift, y_shift) in enumerate(product(range(self.n_plaquettes_x), range(self.n_plaquettes_y))):
            H[i_hex,self.x_shift_y_shift_sublattice_pauli_to_index(x_shift+first_plaq_x_shift,
                                                                   y_shift+first_plaq_y_shift,
                                                                   first_plaq_sublattice,
                                                                   first_plaq_pauli)]=1
        H = sparse.csc_matrix(H)
        self.matching = pymatching.Matching(H)

    def x_shift_y_shift_sublattice_pauli_to_index(self, x_shift, y_shift, sublattice, pauli):
        return np.ravel_multi_index((x_shift % self.n_sites_x, y_shift % self.n_sites_y, sublattice, pauli), (self.n_sites_x, self.n_sites_y, 2, 3))

    def correct(self, fluxes):
        prediction = self.matching.decode((fluxes<0).reshape(-1,1))
        operator_indexes = np.where(prediction)[0]
        return [self.operator_index_to_name(index) for index in operator_indexes]

    def operator_index_to_name(self, index):
        index_to_xyz = {0:'x', 1:'y', 2:'z'}
        index_to_sublattice = {0:'A', 1:'B'}
        site_x, site_y, sublattice_index, xyz_index = np.unravel_index(index, (self.n_sites_x, self.n_sites_y, 2, 3))
        return 'sigma_' + index_to_xyz[xyz_index] + '_' + index_to_sublattice[sublattice_index] + '_' + str(site_x) + '_' + str(site_y)