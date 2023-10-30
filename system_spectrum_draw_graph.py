import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

Ns = 16
num_states = 41
periodic_bc = False

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']


results_df = pd.read_csv("/Users/giladkishony/Dropbox/GILAD/Keva/phd/quantum computation/Periodic Unitaries and Measurements/TestCode/results_system_spectrum.csv")

results_df = results_df.query(f"Ns == {Ns} & periodic_bc == {periodic_bc}")
results_df['h_minus_J'] = results_df.h-results_df.J
results_df = results_df[['h_minus_J', 'energies', 'V']]



with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(6, 6))
    # plt.rcParams["font.family"] = "Times New Roman"

    groups_V = results_df.groupby(["V"])
    for i_V, (V, group_V) in enumerate(groups_V):
        ax = axs[i_V]
        # fig, ax = plt.subplots(num=f"V={V}")
        energies_groups = group_V.groupby('h_minus_J').energies.apply(np.array)
        # ax.plot(energies_groups.index, np.stack(energies_groups.values)[:,:num_states], linestyle='--', marker=marker, color=color)
        energies_mat = np.stack(energies_groups.values)[:,:num_states].T
        for i_energies, energies in enumerate(energies_mat[::-1]):
            if i_energies == len(energies_mat)-2:
                ax.plot(energies_groups.index, energies[:num_states], linestyle='--', marker='s', color='orange', markersize=8)
            else:
                ax.plot(energies_groups.index, energies[:num_states], linestyle='--', marker='o', color='b')
        fig.supxlabel('$h-J$', fontsize='20', fontname='Times New Roman')
        fig.supylabel('Energy', fontsize='20', fontname='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.xticks([-1.5,-1,-0.5,0,0.5,1,1.5])
        # ax2 = ax.twinx()
        # plt.yticks([])
        # ax2.set_ylabel(f'V={V}', fontsize='20', fontname='Times New Roman', rotation=270)
        l = ax.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), handles=[],
                       labels=[])
        l.set_title(title=f'$V={V}$',
                    prop=mpl.font_manager.FontProperties(family='Times New Roman', size=17))
    plt.tight_layout()
    plt.savefig(f'graphs/system_spectrum_periodic_bc_{periodic_bc}_Ns_{Ns}.pdf')#_V_{V}
    plt.show()
