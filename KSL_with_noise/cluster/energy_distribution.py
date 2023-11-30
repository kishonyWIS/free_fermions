from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from FloquetKSL import hexagonal_lattice_site_to_x_y, edit_graph
# from KSL_model import KSLHamiltonian, KSLState
from matplotlib.collections import LineCollection
from itertools import product
import matplotlib as mpl
import pickle
# from free_fermion_hamiltonian import MajoranaFreeFermionHamiltonianTerm


def get_term_energy(S, term):
    return (term._strength * term.get_gauge_value() * S.tensor[
        (*term.site1, term.sublattice1, *term.site2, term.sublattice2)]).real


class TermEnergyDistribution:
    def __init__(self):
        self.gs_energy = 0.
        self.energy = 0.
        self.num_samples = 0

    def set_ground_state_and_geometry(self, gs_term, S_gs):
        self.gs_energy = get_term_energy(S_gs, gs_term)
        self.set_geometry(gs_term)
    def set_geometry(self, term):
        num_sites = (term.system_shape[0], term.system_shape[1])
        for dim in range(2):
            edge_mask = np.abs(term.site2[dim] - term.site1[dim]) > num_sites[dim] / 2
            term.site1[dim, edge_mask] = (term.site1[dim, edge_mask] - num_sites[dim] / 2) % num_sites[dim] + \
                                    num_sites[dim] / 2
            term.site2[dim, edge_mask] = (term.site2[dim, edge_mask] - num_sites[dim] / 2) % num_sites[dim] + \
                                    num_sites[dim] / 2
        self.site1 = term.site1
        self.site2 = term.site2
        self.site1_x_y = hexagonal_lattice_site_to_x_y((term.site1[0], term.site1[1], int(term.sublattice1 / 6)))
        self.site2_x_y = hexagonal_lattice_site_to_x_y((term.site2[0], term.site2[1], int(term.sublattice2 / 6)))
        self.site1_outside_system = np.logical_or(np.logical_or(term.site1[0] < 0, term.site1[0] >= num_sites[0]),
                                    np.logical_or(term.site1[1] < 0, term.site1[1] >= num_sites[1]))
        self.site2_outside_system = np.logical_or(np.logical_or(term.site2[0] < 0, term.site2[0] >= num_sites[0]),
                                    np.logical_or(term.site2[1] < 0, term.site2[1] >= num_sites[1]))

    def update_energy(self, term, S):
        new_energy = get_term_energy(S, term)
        self.energy = (self.energy * self.num_samples + new_energy) / (self.num_samples + 1)
        self.num_samples += 1

class EnergyDistribution:
    def __init__(self, term_names: list):
        self.term_energy_dists = {term_name: TermEnergyDistribution() for term_name in term_names}

    def set_ground_state_and_geometry(self, gs_hamiltonian, S_gs):
        gs_hamiltonian_copy = deepcopy(gs_hamiltonian)
        for term_name, term_energy_dist in self.term_energy_dists.items():
            term_energy_dist.set_ground_state_and_geometry(gs_hamiltonian_copy.terms[term_name], S_gs)

    def update_energy(self, hamiltonian, S):
        hamiltonian_copy = deepcopy(hamiltonian)
        for term_name, term_energy_dist in self.term_energy_dists.items():
            term_energy_dist.update_energy(hamiltonian_copy.terms[term_name], S)

    def save(self, filename):
        with open("pickles/" + filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        with open("pickles/" + filename, 'rb') as f:
            return pickle.load(f)

    def draw(self, ax=None, filename=None):
        if ax is None:
            fig, ax = plt.subplots()
        energy = []
        site1_x = []
        site1_y = []
        site2_x = []
        site2_y = []
        site1_outside_system = []
        site2_outside_system = []
        for term in self.term_energy_dists.values():
            energy.extend((term.energy - term.gs_energy).reshape(-1))
            site1_x.extend(term.site1_x_y[0].reshape(-1))
            site1_y.extend(term.site1_x_y[1].reshape(-1))
            site2_x.extend(term.site2_x_y[0].reshape(-1))
            site2_y.extend(term.site2_x_y[1].reshape(-1))
            site1_outside_system.extend(term.site1_outside_system.reshape(-1))
            site2_outside_system.extend(term.site2_outside_system.reshape(-1))
        energy = np.array(energy)
        site1_x_y = np.stack((site1_x, site1_y)).T
        site2_x_y = np.stack((site2_x, site2_y)).T
        site1_outside_system = np.array(site1_outside_system)
        site2_outside_system = np.array(site2_outside_system)
        segments = np.stack([site1_x_y, site2_x_y], axis=1)
        lc = LineCollection(segments, cmap='coolwarm', linewidths=3)  # , norm=plt.Normalize(0., energy.max()))
        lc.set_array(energy)
        ax.add_collection(lc)
        # draw the sites at circles black if inside the system, grey if outside
        ax.plot(site1_x_y[~site1_outside_system, 0], site1_x_y[~site1_outside_system, 1], 'o', color='black', markersize=5)
        ax.plot(site2_x_y[~site2_outside_system, 0], site2_x_y[~site2_outside_system, 1], 'o', color='black', markersize=5)
        ax.plot(site1_x_y[site1_outside_system, 0], site1_x_y[site1_outside_system, 1], 'o', color='grey', markersize=5)
        ax.plot(site2_x_y[site2_outside_system, 0], site2_x_y[site2_outside_system, 1], 'o', color='grey', markersize=5)
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

    def draw_term_on_1d_cut(self, term_names, ax=None, filename=None):
        if ax is None:
            fig, ax = plt.subplots()
        x = []
        energy_x = []
        for term_name in term_names:
            term_energy_dist = self.term_energy_dists[term_name]
            energy = term_energy_dist.energy - term_energy_dist.gs_energy
            site1 = term_energy_dist.site1
            site2 = term_energy_dist.site2
            one_d_mask = site1[0] == 0
            x.extend((site1[1, one_d_mask] + site2[1, one_d_mask])/2)
            energy_x.extend(energy[one_d_mask])
        # sort by x
        x = np.array(x)
        energy_x = np.array(energy_x)
        sort_mask = np.argsort(x)
        x = x[sort_mask]
        energy_x = energy_x[sort_mask]
        plt.plot(x, energy_x)