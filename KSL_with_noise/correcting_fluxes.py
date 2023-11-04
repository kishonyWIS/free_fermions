import numpy as np
import scipy.sparse as sparse
import pymatching
from itertools import product

class KSL_flux_corrector:
    def __init__(self, n_sites_x, n_sites_y, periodic_bc=False):
        self.n_sites_x = n_sites_x
        self.n_sites_y = n_sites_y
        H = np.zeros((4,2*n_sites_x*n_sites_y*3))
        relative_op_indexes = np.array([3*1+1, 3*1+2, 3*2+0, 3*2+2, 3*3+0, 3*3+1, 3*6+0, 3*6+1, 3*7+0, 3*7+2, 3*8+1, 3*8+2])
        for i_hex, (x_shift, y_shift) in enumerate(product(range(n_sites_x-1), range(n_sites_y-1))):
            H[i_hex,3*2*y_shift + 3*2*x_shift*n_sites_y + relative_op_indexes]=1
        H = sparse.csc_matrix(H)
        self.matching = pymatching.Matching(H)

    def correct(self, fluxes):
        prediction = self.matching.decode((fluxes<0).reshape(-1,1))
        operator_indexes = np.where(prediction)[0]
        return [self.operator_index_to_name(index) for index in operator_indexes]

    def operator_index_to_name(self, index):
        index_to_xyz = {0:'x', 1:'y', 2:'z'}
        index_to_sublattice = {0:'A', 1:'B'}
        site_x, site_y, sublattice_index, xyz_index = np.unravel_index(index, (self.n_sites_x, self.n_sites_y, 2, 3))
        return 'sigma_' + index_to_xyz[xyz_index] + '_' + index_to_sublattice[sublattice_index] + '_' + str(site_x) + '_' + str(site_y)