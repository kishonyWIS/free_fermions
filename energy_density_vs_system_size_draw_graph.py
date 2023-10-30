import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

trotter_steps = 100
cycles = 1000

with sns.axes_style("whitegrid"):
# make a 2 by 2 subplots
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(6,6.2))

    for i_J_j,(J,h) in enumerate([(0.5,1.0), (1.0,0.5)]):
        for i_V,V in enumerate([0.0, 0.1]):
            ax = axs[i_J_j, i_V]
            # results_df = pd.read_csv("/Users/giladkishony/Dropbox/GILAD/Keva/phd/quantum computation/Periodic Unitaries and Measurements/TestCode/results_energy_density_vs_system_size_first_excitation.csv")
            results_df = pd.read_csv("/Users/giladkishony/Dropbox/GILAD/Keva/phd/quantum computation/Periodic Unitaries and Measurements/TestCode/results_energy_density_vs_system_size.csv")

            results_df = results_df.query(f"Nt == {trotter_steps} & N_iter == {cycles} & V == {V} & h == {h} & J == {J}")


            markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = itertools.cycle(prop_cycle.by_key()['color'])

            plt.rcParams["font.family"] = "Times New Roman"
            groups = results_df.groupby(["errors_per_cycle_per_qubit"])
            handles = []
            group_labels = []
            for errors_per_cycle_per_qubit, group in groups:
                marker = next(markers)
                color = next(colors)
                # plt.plot(group.errors_per_cycle_per_qubit, group.energy_density, linestyle='None', marker=marker, label=f'J = {J}, h = {h}')
                yerr = np.array(group.energy_density_std)
                yerr[yerr > group.energy_density] = 0.9999 * group.energy_density[yerr > group.energy_density]
                handles.append(ax.errorbar(group.Ns, group.energy_density, yerr=yerr, linestyle='-', marker=marker, color=color,
                                           label=errors_per_cycle_per_qubit))
                group_labels.append(f'{errors_per_cycle_per_qubit}')
                yerr = np.array(group.energy_density_std_first_excited_state)
                yerr[yerr > group.energy_density_first_excited_state] = 0.9999 * group.energy_density_first_excited_state[yerr > group.energy_density_first_excited_state]
                if J>h:
                    ax.errorbar(group.Ns, group.energy_density_first_excited_state, yerr=yerr, linestyle='--', marker=marker, color=color)
            ax.set_yscale('log')
            ax.tick_params(axis='both', which='major', labelsize=15)
            # plt.tick_params(axis='y', which='minor')
            # ax.yaxis.set_minor_formatter(mpl.ticker.FormatStrFormatter("%.1f"))
            plt.xticks([2,4,6,8,10])
            # plt.yticks([2e-3,1e-2,0.5e-1,0.7e-1])
            plt.ylim([2e-3,0.7e-1])
            plt.ylim([1e-3, 1e-1])
            fig.supxlabel('System size', fontsize='20', fontname='Times New Roman')
            fig.supylabel('Energy density', fontsize='20', fontname='Times New Roman')
            l = fig.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), handles=handles, labels=group_labels, ncol=4, loc='upper center')
            l.set_title(title='errors per cycle per qubit',
                        prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
    plt.tight_layout()
    plt.subplots_adjust(top=0.84)
    plt.savefig(f'graphs/energy_vs_system_size_steps_{trotter_steps}_cycles_{cycles}.pdf')#_J_{J}_h_{h}_V_{V}
    plt.show()