from __future__ import annotations
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from FloquetKSL import hexagonal_lattice_site_to_x_y, edit_graph
from matplotlib.collections import LineCollection
from itertools import product
import matplotlib as mpl

class BondEnergy():
    def __init__(self, site1_x_y, site2_x_y, gs_energy, hamiltonian_matrix, line_type='-'):
        self.site1_x_y = site1_x_y
        self.site2_x_y = site2_x_y
        self.gs_energy = gs_energy
        self.line_type = line_type
        self.hamiltonian_matrix = hamiltonian_matrix
        self.energy = 0.
        self.num_samples = 0

    @classmethod
    def from_hamiltonian_and_site1(cls, hamiltonian, hamiltonian_fixed_gauge, S_gs, term_name, site1, num_sites):
        term = deepcopy(hamiltonian.terms[term_name])
        term.filter_site1(lambda s1: np.logical_and(s1[0] == site1[0], s1[1] == site1[1]))
        hamiltonian_matrix = term.time_independent_matrix
        if np.prod(term.site1.shape) == 0:
            return None
        term_fixed_gauge = deepcopy(hamiltonian_fixed_gauge.terms[term_name])
        term_fixed_gauge.filter_site1(lambda s1: np.logical_and(s1[0] == site1[0], s1[1] == site1[1]))
        term_fixed_gauge_matrix = term_fixed_gauge.time_independent_matrix
        gs_energy = S_gs.get_energy(term_fixed_gauge_matrix)
        linetype = '-'
        for dim in range(2):
            if abs(term.site2[dim] - term.site1[dim]) > num_sites[dim] / 2:
                linetype = '--'
                term.site1[dim] = (term.site1[dim] - num_sites[dim] / 2) % num_sites[dim] + \
                                  num_sites[dim] / 2
                term.site2[dim] = (term.site2[dim] - num_sites[dim] / 2) % num_sites[dim] + \
                                  num_sites[dim] / 2
        site1_x_y = hexagonal_lattice_site_to_x_y((term.site1[0], term.site1[1], int(term.sublattice1 / 6)))
        site2_x_y = hexagonal_lattice_site_to_x_y((term.site2[0], term.site2[1], int(term.sublattice2 / 6)))
        return cls(site1_x_y, site2_x_y, gs_energy, hamiltonian_matrix, line_type=linetype)

    def update_matrix(self, hamiltonian, term_name, site1):
        term = deepcopy(hamiltonian.terms[term_name])
        term.filter_site1(lambda s1: np.logical_and(s1[0] == site1[0], s1[1] == site1[1]))
        self.hamiltonian_matrix = term.time_independent_matrix

    def update_energy(self, S):
        energy = S.get_energy(self.hamiltonian_matrix) - self.gs_energy
        self.energy = (self.energy * self.num_samples + energy) / (self.num_samples + 1)
        self.num_samples += 1

class SpatialEnergy():
    def __init__(self, hamiltonian, hamiltonian_fixed_gauge, S_gs, term_names:list[str]):
        self.term_names = term_names
        self.num_sites = (hamiltonian.system_shape[0], hamiltonian.system_shape[1])
        # initialize bonds
        self.bonds_energies = {}
        for term_name, site1 in self.iterate_terms():
            term_name_site = self.term_name_site_to_full_name(term_name, site1)
            bond_energy = BondEnergy.from_hamiltonian_and_site1(hamiltonian, hamiltonian_fixed_gauge, S_gs, term_name, site1, self.num_sites)
            if bond_energy is not None:
                self.bonds_energies[term_name_site] = bond_energy

    def term_name_site_to_full_name(self, term_name, site):
        return term_name + '_' + str(site[0]) + '_' + str(site[1])

    def iterate_terms(self):
        return product(self.term_names, product(range(self.num_sites[0]), range(self.num_sites[1])))

    def update_matrix(self, hamiltonian):
        for term_name, site1 in self.iterate_terms():
            term_name_site = self.term_name_site_to_full_name(term_name, site1)
            if term_name_site in self.bonds_energies:
                self.bonds_energies[term_name_site].update_matrix(hamiltonian, term_name, site1)

    def update_energies(self, S):
        for term in self.bonds_energies.values():
            term.update_energy(S)

    def draw(self, ax=None, filename=None):
        if ax is None:
            fig, ax = plt.subplots()
        energy = []
        site1_x_y = []
        site2_x_y = []
        linetype = []
        for term in self.bonds_energies.values():
            energy.append(term.energy)
            site1_x_y.append(term.site1_x_y)
            site2_x_y.append(term.site2_x_y)
            linetype.append(term.line_type)
        energy = np.array(energy)
        site1_x_y = np.array(site1_x_y).squeeze()
        site2_x_y = np.array(site2_x_y).squeeze()
        segments = np.stack([site1_x_y, site2_x_y], axis=1)
        lc = LineCollection(segments, norm=mpl.colors.CenteredNorm(), cmap='coolwarm', linestyles=linetype, linewidths=3)  # , norm=plt.Normalize(0., energy.max()))
        lc.set_array(energy)
        ax.add_collection(lc)
        # draw the sites at circles
        ax.plot(site1_x_y[:, 0], site1_x_y[:, 1], 'o', color='black', markersize=5)
        ax.plot(site2_x_y[:, 0], site2_x_y[:, 1], 'o', color='black', markersize=5)
        x_min = min(site1_x_y[:, 0].min(), site2_x_y[:, 0].min())
        x_max = max(site1_x_y[:, 0].max(), site2_x_y[:, 0].max())
        y_min = min(site1_x_y[:, 1].min(), site2_x_y[:, 1].min())
        y_max = max(site1_x_y[:, 1].max(), site2_x_y[:, 1].max())
        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(y_min - 0.5, y_max + 0.5)
        edit_graph(None, None, colorbar_title='Energy',
                   colorbar_args={'mappable': lc, 'orientation': 'horizontal', 'pad': -0.0, 'aspect': 30,
                                  'shrink': 0.5})
        ax.axis('equal')
        plt.axis('off')
        if filename:
            plt.savefig("graphs/" + filename)


# def draw_spatial_energy_of_terms(hamiltonian, hamiltonian_fixed_gauge, S, S_gs, term_names:list[str], ax=None, filename=None):
#     if ax is None:
#         fig, ax = plt.subplots()
#     num_sites_x, num_sites_y = hamiltonian.system_shape[0], hamiltonian.system_shape[1]
#     num_sites_tuple = (num_sites_x, num_sites_y)
#     site1_x_y = []
#     site2_x_y = []
#     energy = []
#     linetype = []
#     for term_name in term_names:
#         for i_x in range(num_sites_x):
#             for i_y in range(num_sites_y):
#                 term = deepcopy(hamiltonian.terms[term_name])
#                 term.filter_site1(lambda site1: np.logical_and(site1[0] == i_x, site1[1] == i_y))
#                 if np.prod(term.site1.shape) == 0:
#                     continue
#                 term_matrix = term.time_independent_matrix
#                 term_fixed_gauge = deepcopy(hamiltonian_fixed_gauge.terms[term_name])
#                 term_fixed_gauge.filter_site1(lambda site1: np.logical_and(site1[0] == i_x, site1[1] == i_y))
#                 term_fixed_gauge_matrix = term_fixed_gauge.time_independent_matrix
#                 # energy.append(S_gs.get_energy(term_fixed_gauge_matrix))
#                 energy.append(S.get_energy(term_matrix) - S_gs.get_energy(term_fixed_gauge_matrix))
#                 linetype.append('-')
#                 for dim in range(2):
#                     if abs(term.site2[dim] - term.site1[dim]) > num_sites_tuple[dim]/2:
#                         linetype[-1] = '--'
#                         term.site1[dim] = (term.site1[dim] - num_sites_tuple[dim] / 2) % num_sites_tuple[dim] + \
#                                           num_sites_tuple[dim] / 2
#                         term.site2[dim] = (term.site2[dim] - num_sites_tuple[dim] / 2) % num_sites_tuple[dim] + \
#                                           num_sites_tuple[dim] / 2
#                 site1_x_y.append(
#                     hexagonal_lattice_site_to_x_y((term.site1[0], term.site1[1], int(term.sublattice1 / 6))))
#                 site2_x_y.append(
#                     hexagonal_lattice_site_to_x_y((term.site2[0], term.site2[1], int(term.sublattice2 / 6))))
#     energy = np.array(energy)
#     site1_x_y = np.array(site1_x_y).squeeze()
#     site2_x_y = np.array(site2_x_y).squeeze()
#
#     segments = np.stack([site1_x_y, site2_x_y], axis=1)
#     lc = LineCollection(segments, cmap='viridis',linestyles=linetype)#, norm=plt.Normalize(0., energy.max()))
#     lc.set_array(energy)
#     ax.add_collection(lc)
#     x_min = min(site1_x_y[:, 0].min(), site2_x_y[:, 0].min())
#     x_max = max(site1_x_y[:, 0].max(), site2_x_y[:, 0].max())
#     y_min = min(site1_x_y[:, 1].min(), site2_x_y[:, 1].min())
#     y_max = max(site1_x_y[:, 1].max(), site2_x_y[:, 1].max())
#     ax.set_xlim(x_min - 0.5, x_max + 0.5)
#     ax.set_ylim(y_min - 0.5, y_max + 0.5)
#     edit_graph(None, None, colorbar_title='Energy', colorbar_args={'mappable':lc, 'orientation': 'horizontal', 'pad': -0.0, 'aspect': 30, 'shrink': 0.5})
#     ax.axis('equal')
#     plt.axis('off')
#     if filename:
#         plt.savefig("graphs/"+filename)