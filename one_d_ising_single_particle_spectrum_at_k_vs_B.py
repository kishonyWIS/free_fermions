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
g1 = 0.5
E_k = 1

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])
linestyles = itertools.cycle([':','-'])
linewidths = itertools.cycle([1,None])
handles = []

plt.figure(figsize=(6,6))

for g in [0, g1]:

    B_list = np.linspace(B1, B0, 100)
    spectrum = np.zeros((len(B_list),4))
    for iB, B in enumerate(B_list):

        hamiltonian = np.array([[0, -1j*E_k, 1j*g, 0],
                                [1j*E_k, 0, 0, 0],
                                [-1j*g, 0, 0, -1j*B],
                                [0, 0, 1j*B, 0]])

        spectrum[iB,:], _ = eigh(hamiltonian)

    with sns.axes_style("whitegrid"):
        rc = {"font.family": "serif",
              "mathtext.fontset": "stix"}
        plt.rcParams.update(rc)
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams['legend.title_fontsize'] = 40
        plt.rcParams["font.family"] = "Times New Roman"
        color = next(colors)
        linewidth = next(linewidths)
        linestyle = next(linestyles)
        # for ii in range(4):
        handles.append(plt.plot(B_list, spectrum, 'b', linestyle=linestyle, linewidth=linewidth)[0])
plt.xlabel('$B$', fontsize='40', fontname='Times New Roman')#, fontweight='bold')
plt.ylabel('Energy', fontsize='40', fontname='Times New Roman')#, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=30)
plt.xlim([B1,B0])
plt.xticks([B1,B0], labels=['$0$','$B_0$'])
plt.yticks([0])
plt.legend(handles,['$g=0$',f'$g={g1}$'], prop=mpl.font_manager.FontProperties(family='Times New Roman', size=30), loc='upper left', ncol=2, columnspacing=0.3, handletextpad=0.3, borderpad=0, borderaxespad=0.1, handlelength=1)
plt.tight_layout()
plt.savefig(f'graphs/0d_ising_single_particle_spectrum.pdf')

plt.show()