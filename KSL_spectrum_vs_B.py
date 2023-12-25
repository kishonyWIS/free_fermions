import itertools

import numpy as np
from matplotlib import pyplot as plt
from translational_invariant_KSL import get_KSL_model, get_Delta, get_f
from time_dependence_functions import get_g, get_B
import seaborn as sns
import matplotlib as mpl


B0 = 7.
B1 = 0.
g0 = 0.5

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])
linestyles = itertools.cycle(['-',':'])
labels = ['$g_A=0.5, g_B=0.5$', '$g_A=0.5, g_B=0$']

for i_num_cooling_sublattices, num_cooling_sublattices in enumerate([2,1]):

    B_list = np.linspace(B1, B0, 100)
    spectrum = np.zeros((len(B_list),6))
    for iB, B in enumerate(B_list):
        kappa = 1

        Jx = 1
        Jy = 1
        Jz = 1

        n_k_points = 1+6*2


        kx = 2/3*np.pi # 2/3*np.pi, 0
        ky = -2/3*np.pi # -2/3*np.pi, np.pi


        f = get_f(kx, ky, Jx, Jy, Jz)
        f_real = np.real(f)
        f_imag = np.imag(f)

        Delta = get_Delta(kx, ky, kappa)

        hamiltonian, S, E_gs = \
            get_KSL_model(f=f, Delta=Delta, g=lambda tt:g0, B=lambda tt:B, initial_state='random', num_cooling_sublattices=num_cooling_sublattices)

        spectrum[iB,:] = hamiltonian.get_excitation_spectrum(0) # double because k and -k are the same

    with sns.axes_style("whitegrid"):
        rc = {"font.family": "serif",
              "mathtext.fontset": "stix"}
        plt.rcParams.update(rc)
        plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
        plt.rcParams['legend.title_fontsize'] = 20
        plt.rcParams["font.family"] = "Times New Roman"
        color = next(colors)
        linestyle = next(linestyles)
        for ii in range(6):
            if ii == 0:
                plt.plot(B_list, spectrum[:, ii], color, linestyle=linestyle, label=labels[i_num_cooling_sublattices])
            else:
                plt.plot(B_list, spectrum[:, ii], color, linestyle=linestyle)
        plt.xlabel('$B$', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
        plt.ylabel('Energy', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=15)
        l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15),ncols=1)
        # l.set_title(title='auxiliary sites per unit cell',
        #             prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
        plt.tight_layout()
        plt.savefig(f'graphs/KSL_spectrum.pdf')

plt.show()