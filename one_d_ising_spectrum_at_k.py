import itertools

import numpy as np
from matplotlib import pyplot as plt
from time_dependence_functions import get_g, get_B
from one_d_ising import get_TFI_model
import seaborn as sns
import matplotlib as mpl
from scipy.linalg import eigh


B0 = 5.
B1 = 0.
g0 = 0.5

T = 50.
t1 = T / 4

smoothed_B = lambda t: get_B(t, B0, B1, T)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])
linestyles = itertools.cycle([':','-'])
linewidths = itertools.cycle([1,None])

for smoothed_g in [lambda t:0, lambda t: get_g(t, g0, T, t1)]:

    t_list = np.linspace(0,T,100)
    spectrum = np.zeros((len(t_list),4))
    for it, t in enumerate(t_list):

        E_k = 1

        B = smoothed_B(t)
        g = smoothed_g(t)
        hamiltonian = np.array([[-E_k-B, 0, 0, -g],
                                [0, -E_k+B, -g, 0],
                                [0, -g, E_k-B, 0],
                                [-g, 0, 0, E_k+B]])

        spectrum[it,:], _ = eigh(hamiltonian)

    with sns.axes_style("whitegrid"):
        rc = {"font.family": "serif",
              "mathtext.fontset": "stix"}
        plt.rcParams.update(rc)
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams['legend.title_fontsize'] = 20
        plt.rcParams["font.family"] = "Times New Roman"
        color = next(colors)
        linewidth = next(linewidths)
        linestyle = next(linestyles)
        for ii in range(4):
            plt.plot(t_list/T, spectrum[:, ii], 'b', linestyle=linestyle, linewidth=linewidth)
    plt.xlabel('$t/T$', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.xlim([0,1])
    plt.yticks([0])
    plt.tight_layout()
    plt.savefig(f'graphs/0d_ising_spectrum.pdf')

plt.show()