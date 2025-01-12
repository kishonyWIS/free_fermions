import numpy as np
from scipy.linalg import expm
from scipy.linalg import eig
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection

from plot_utils import edit_graph
from plot_utils import *
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

mpl.use('TkAgg')  # Or 'Qt5Agg' if you have PyQt5 installed
set_latex_params()
# sns.set_style("whitegrid")

J_factor_list = [0.5,1.0001,1.5,1.85,1.9,1.5]
pulse_length_list = [0.2,0.2,0.2,0.42,0.63,0.75]

# make a 2 by 3 grid of plots
fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)

for J_factor, pulse_length, ax in tqdm(zip(J_factor_list, pulse_length_list, axs.flatten())):
    marker_size = 5
    J = np.pi / 4 * J_factor
    J /= pulse_length
    num_sites_y = 60
    kx_list = np.linspace(0.001, 2*np.pi-0.001, 101)

    def get_Jx_hamiltonian(J, num_sites_y, kx):
        s = -2*J*np.exp(1j*kx/2)
        upper_diagonal = np.zeros(2*num_sites_y-1, dtype=complex)
        upper_diagonal[::2] = 1j*s
        lower_diagonal = np.zeros(2*num_sites_y-1, dtype=complex)
        lower_diagonal[::2] = -1j*np.conj(s)
        return np.diag(upper_diagonal, 1) + np.diag(lower_diagonal, -1)

    def get_Jy_hamiltonian(J, num_sites_y, kx):
        return get_Jx_hamiltonian(J, num_sites_y, -kx)

    def get_Jz_hamiltonian(J, num_sites_y, kx):
        r = 2*J
        upper_diagonal = np.zeros(2*num_sites_y-1, dtype=complex)
        upper_diagonal[1::2] = 1j*r
        lower_diagonal = np.zeros(2*num_sites_y-1, dtype=complex)
        lower_diagonal[1::2] = -1j*np.conj(r)
        return np.diag(upper_diagonal, 1) + np.diag(lower_diagonal, -1)

    pulse_intervals = {'x': [0, pulse_length], 'y': [1/3, 1/3 + pulse_length], 'z': [2/3, 2/3 + pulse_length]}

    # find all overlapping intervals
    all_times_sorted = np.concatenate(list(pulse_intervals.values())) % 1
    all_times_sorted = np.concatenate((all_times_sorted, [0,1]))
    all_times_sorted = np.unique(all_times_sorted)
    all_times_sorted.sort()

    active_pulses = []

    def intervals_have_overlap_mod_1(interval1, interval2):
        end1 = (interval1[1] - interval1[0]) % 1
        start2 = (interval2[0] - interval1[0]) % 1
        end2 = (interval2[1] - interval1[0]) % 1
        return start2 < end1 and end2 > 0



    for i in range(len(all_times_sorted)-1):
        start_time = all_times_sorted[i]
        end_time = all_times_sorted[i+1]
        active_pulses.append([pulse for pulse, interval in pulse_intervals.items() if
                              intervals_have_overlap_mod_1(interval, [start_time, end_time])])

    Y = np.zeros((2*num_sites_y, 2*num_sites_y))
    Y[::2,:] = 1.5*np.arange(num_sites_y).reshape(-1,1)
    Y[1::2,:] = 1.5*np.arange(num_sites_y).reshape(-1,1) + 0.5


    def truncate_colormap(cmap, min_val=0.0, max_val=1.0, n=100):
        new_cmap = LinearSegmentedColormap.from_list(
            f'trunc({cmap.name},{min_val:.2f},{max_val:.2f})',
            cmap(np.linspace(min_val, max_val, n))
        )
        return new_cmap


    colormap = truncate_colormap(plt.get_cmap('jet'), 0.2, 0.8)

    # Initialize data storage
    all_kx = []
    all_energies = []
    all_colors = []

    # Iterate over kx values
    for kx in kx_list:
        pulse_hamiltonians = {
            'x': get_Jx_hamiltonian(J, num_sites_y, kx),
            'y': get_Jy_hamiltonian(J, num_sites_y, kx),
            'z': get_Jz_hamiltonian(J, num_sites_y, kx)
        }
        # Calculate the unitary for the full cycle
        unitary = np.eye(2 * num_sites_y, dtype=complex)
        for i_time in range(len(all_times_sorted) - 1):
            delta_t = all_times_sorted[i_time + 1] - all_times_sorted[i_time]
            active_pulses_i = active_pulses[i_time]
            if len(active_pulses_i) == 0:
                continue
            hamiltonian = sum([pulse_hamiltonians[pulse] for pulse in active_pulses_i])
            unitary = expm(-1j * delta_t * hamiltonian) @ unitary
        phases, states = eig(unitary)
        energies = np.angle(phases)

        # Sort the energies and states according to the energies
        sort_indices = np.argsort(energies)
        energies = energies[sort_indices]
        states = states[:, sort_indices]

        # Colors according to the mean location of the states in the y direction
        colors = np.sum(np.abs(states) ** 2 * Y, axis=0)
        colors = colors / np.max(Y)

        # Store data for plotting
        all_kx.append(kx * np.ones_like(energies))
        all_energies.append(energies)
        all_colors.append(colors)

    # Convert lists to arrays for easier manipulation
    all_kx = np.array(all_kx)
    all_energies = np.array(all_energies)
    all_colors = np.array(all_colors)

    # Create line segments and assign colors to each segment
    segments = []
    segment_colors = []
    for i in range(all_energies.shape[1]):  # Loop over energy levels
        for j in range(len(kx_list) - 1):  # Loop over segments along kx
            x_start, x_end = all_kx[j, i], all_kx[j + 1, i]
            y_start, y_end = all_energies[j, i], all_energies[j + 1, i]
            segments.append([(x_start, y_start), (x_end, y_end)])
            segment_colors.append(all_colors[j, i])  # Assign color to each segment

    # Use LineCollection to plot the lines with segment-wise colors
    line_collection = LineCollection(
        segments,
        cmap=colormap,
        norm=plt.Normalize(vmin=0, vmax=1),
        linewidths=3
    )
    line_collection.set_array(np.array(segment_colors))
    ax.add_collection(line_collection)

    # # increase fontsize
    # plt.xticks(fontname='Times New Roman', fontsize=30)
    # plt.yticks(fontname='Times New Roman', fontsize=30)
    # # increase label fontsize
    # plt.xlabel('$k_xa$', fontname='Times New Roman', fontsize=45, labelpad=-2)
    # plt.ylabel('$\\varepsilon T$', fontname='Times New Roman', fontsize=45, labelpad=-30)
    plt.xlim([0,2*np.pi])
    plt.ylim([-np.pi,np.pi])
    cmap = mpl.cm.ScalarMappable(norm=None, cmap=colormap)
    cmap.set_array([])
    # cbar = plt.colorbar(cmap, ax=ax, ticks= [0,1])
    # cbar.ax.set_yticklabels(['0', '$N_y$'])
    # cbar.set_label('$y$ center of mass', labelpad=-15)
    # cbar.ax.tick_params(labelsize=30)

    x_ticks = [0, np.pi, 2*np.pi]
    y_ticks = [-np.pi, 0, np.pi]
    x_ticks_labels = ['0', '$\\pi$', '$2\\pi$']
    y_ticks_labels = ['$-\\pi$', '0', '$\\pi$']

    edit_graph(None, None, ax=ax,#'$k_xa$', '$\\varepsilon T$',
               tight=True,
               xticks=x_ticks, yticks=y_ticks,
               xticklabels=x_ticks_labels, yticklabels=y_ticks_labels, scale=6)
    #plt.xticks([0,np.pi,2*np.pi], ['0', '$\\pi$', '$2\\pi$'], fontname='Times New Roman')
    #plt.yticks([-np.pi,0,np.pi], ['$-\\pi$', '0', '$\\pi$'], fontname='Times New Roman')
    #turn off grid
    ax.grid(False)
    # plt.tight_layout()
plt.subplots_adjust(wspace=0.15, hspace=0.15)
plt.savefig(f'graphs/time_vortex/FloquetKSLcylinder_multiple.pdf', bbox_inches='tight')
plt.show()