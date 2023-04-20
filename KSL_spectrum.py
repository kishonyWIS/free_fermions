import itertools

import numpy as np
from matplotlib import pyplot as plt
from translational_invariant_KSL import get_KSL_model, get_Delta, get_f
from one_d_ising import get_smoothed_func, get_g, get_B
import seaborn as sns
import matplotlib as mpl


B0 = 5.
B1 = 0.
g0 = 0.5

T = 30.
t1 = T / 4

smoothed_g_before_zeroing = lambda t: get_smoothed_func(t, lambda tt: get_g(tt, g0, T, t1), T / 10)
smoothed_B_before_zeroing = lambda t: get_smoothed_func(t, lambda tt: get_B(tt, B0, B1, T, t1), T / 10)
smoothed_g = lambda t: smoothed_g_before_zeroing(t) - smoothed_g_before_zeroing(T)
smoothed_B = lambda t: smoothed_B_before_zeroing(t) - smoothed_B_before_zeroing(T)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])
linestyles = itertools.cycle(['-',':'])

for num_cooling_sublattices in [2,1]:

    t_list = np.linspace(0,T,100)
    spectrum = np.zeros((len(t_list),6))
    for it, t in enumerate(t_list):
        kappa = 0.1

        Jx = 1
        Jy = 1
        Jz = 1

        n_k_points = 1+6*2


        kx = 2/3*np.pi
        ky = -2/3*np.pi


        f = get_f(kx, ky, Jx, Jy, Jz)
        f_real = np.real(f)
        f_imag = np.imag(f)

        Delta = get_Delta(kx, ky, kappa)

        hamiltonian, S, E_gs = \
            get_KSL_model(f=f, Delta=Delta, g=smoothed_g, B=smoothed_B, initial_state='random', num_cooling_sublattices=num_cooling_sublattices)

        spectrum[it,:] = hamiltonian.get_excitation_spectrum(t)

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
                plt.plot(t_list/T, spectrum[:, ii], color, linestyle=linestyle, label=str(num_cooling_sublattices))
            else:
                plt.plot(t_list/T, spectrum[:, ii], color, linestyle=linestyle)
        plt.xlabel('$t/T$', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
        plt.ylabel('Energy', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
        plt.tick_params(axis='both', which='major', labelsize=15)
        l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
        l.set_title(title='auxiliary sites per unit cell',
                    prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
        plt.tight_layout()
        plt.savefig(f'graphs/KSL_spectrum.pdf')

plt.show()