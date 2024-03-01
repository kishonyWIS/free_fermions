import numpy as np
from scipy.linalg import expm
from scipy.linalg import eig
from matplotlib import pyplot as plt
import matplotlib as mpl
from FloquetKSL import edit_graph

J = np.pi / 4 * 0.9
pulse_length = 1/2
J /= pulse_length
num_sites_y = 40
kx_list = np.linspace(0.001, 2*np.pi-0.001, 401)
fig, ax = plt.subplots()

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

colormap = plt.get_cmap('jet')

for kx in kx_list:
    pulse_hamiltonians = {
        'x': get_Jx_hamiltonian(J, num_sites_y, kx),
        'y': get_Jy_hamiltonian(J, num_sites_y, kx),
        'z': get_Jz_hamiltonian(J, num_sites_y, kx)
    }
    # calculate the unitary for the full cycle
    unitary = np.eye(2*num_sites_y, dtype=complex)
    for i_time in range(len(all_times_sorted)-1):
        delta_t = all_times_sorted[i_time+1] - all_times_sorted[i_time]
        active_pulses_i = active_pulses[i_time]
        if len(active_pulses_i) == 0:
            continue
        hamiltonian = sum([pulse_hamiltonians[pulse] for pulse in active_pulses_i])
        unitary = expm(-1j*delta_t*hamiltonian) @ unitary
    phases, states = eig(unitary)
    energies = np.angle(phases)
    # sort the energies and phases and states according to the energies
    sort_indices = np.argsort(energies)
    energies = energies[sort_indices]
    states = states[:,sort_indices]
    # colors according to the mean location of the states in the y direction

    colors = np.sum(np.abs(states)**2 * Y, axis=0)
    colors = colors/np.max(Y)
    plt.scatter(kx*np.ones_like(energies), energies, color=colormap(colors))

edit_graph('$k_x$', '$\\varepsilon$')
# increase fontsize
plt.xticks(fontname='Times New Roman', fontsize=22)
plt.yticks(fontname='Times New Roman', fontsize=22)
# increase label fontsize
plt.xlabel('$k_x$', fontname='Times New Roman', fontsize=30, labelpad=-2)
plt.ylabel('$\\varepsilon$', fontname='Times New Roman', fontsize=30, labelpad=-10)
plt.xlim([0,2*np.pi])
plt.ylim([-np.pi,np.pi])
plt.xticks([0,np.pi,2*np.pi], ['0', '$\\pi$', '$2\\pi$'], fontname='Times New Roman')
plt.yticks([-np.pi,0,np.pi], ['$-\\pi$', '0', '$\\pi$'], fontname='Times New Roman')
cmap = mpl.cm.ScalarMappable(norm=None, cmap=colormap)
cmap.set_array([])
cbar = plt.colorbar(cmap, ax=ax, ticks= [0,1])
cbar.ax.set_yticklabels(['0', '$N_y$'], fontname='Times New Roman', fontsize='22')
cbar.set_label('$y$ center of mass', fontsize='25', fontname='Times New Roman', labelpad=-15)
cbar.ax.tick_params(labelsize=22)
plt.savefig('graphs/time_vortex/FloquetKSLcylinder.pdf')
plt.show()