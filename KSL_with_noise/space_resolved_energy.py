from __future__ import annotations
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from FloquetKSL import hexagonal_lattice_site_to_x_y, edit_graph
import matplotlib as mpl
from matplotlib.collections import LineCollection


def draw_spatial_energy_of_terms(hamiltonian, S, S_gs, term_names:list[str], ax=None, filename=None):
    if ax is None:
        fig, ax = plt.subplots()
    num_sites_x, num_sites_y = hamiltonian.system_shape[0], hamiltonian.system_shape[1]
    site1_x_y = []
    site2_x_y = []
    energy = []
    for term_name in term_names:
        for i_x in range(num_sites_x):
            for i_y in range(num_sites_y):
                term = deepcopy(hamiltonian.terms[term_name])
                term.filter_site1(lambda site1: np.logical_and(site1[0] == i_x, site1[1] == i_y))
                if np.prod(term.site1.shape) == 0:
                    continue
                term_matrix = term.time_independent_matrix
                energy.append(S_gs.get_energy(term_matrix))
                # energy.append(S.get_energy(term_matrix) - S_gs.get_energy(term_matrix))
                site1_x_y.append(hexagonal_lattice_site_to_x_y((term.site1[0], term.site1[1], int(term.sublattice1/6))))
                site2_x_y.append(hexagonal_lattice_site_to_x_y((term.site2[0], term.site2[1], int(term.sublattice2/6))))
    energy = np.array(energy)
    site1_x_y = np.array(site1_x_y).squeeze()
    site2_x_y = np.array(site2_x_y).squeeze()

    segments = np.stack([site1_x_y, site2_x_y], axis=1)
    lc = LineCollection(segments, cmap='viridis')#, norm=plt.Normalize(0., energy.max()))
    lc.set_array(energy)
    ax.add_collection(lc)
    x_min = min(site1_x_y[:, 0].min(), site2_x_y[:, 0].min())
    x_max = max(site1_x_y[:, 0].max(), site2_x_y[:, 0].max())
    y_min = min(site1_x_y[:, 1].min(), site2_x_y[:, 1].min())
    y_max = max(site1_x_y[:, 1].max(), site2_x_y[:, 1].max())
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.set_ylim(y_min - 0.5, y_max + 0.5)
    edit_graph(None, None, colorbar_title='Energy', colorbar_args={'mappable':lc, 'orientation': 'horizontal', 'pad': -0.0, 'aspect': 30, 'shrink': 0.5})
    ax.axis('equal')
    plt.axis('off')
    if filename:
        plt.savefig("graphs/"+filename)