import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

trotter_steps = 400
num_sites_x = [10]
num_sites_y = [10]
cooling_half_life = 2
max_errors_per_cycle_per_qubit = 0.002
periodic_bc = True

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])

with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure()

    for accepted_fraction in [0.1,0.3,0.5,1.]:
        results_df = pd.read_csv("KSL_results_from_cluster.csv")
        results_df = results_df.query(
            f"num_sites_x in {num_sites_x} & num_sites_y in {num_sites_y} & Nt == {trotter_steps}")
        results_df = results_df[results_df.periodic_bc == periodic_bc]
        results_df = results_df[results_df.errors_per_cycle_per_qubit <= max_errors_per_cycle_per_qubit]
        marker = next(markers)
        color = next(colors)
        # aggregate energy_density and energy_density_std entries with same errors_per_cycle_per_qubit
        errors_per_cycle_per_qubit_list = []
        energy_density_list = []
        energy_density_std_list = []
        for group_name, group in results_df.groupby('errors_per_cycle_per_qubit'):
            # keep only entries with errors_per_cycle_per_qubit <= post_select_max_num_errors in the last cycles_averaging_buffer cycles
            # group['max_num_corrections_in_last_cycles'] = group['num_corrections'].transform(lambda x: x.rolling(cycles_averaging_buffer, min_periods=1).max())
            # calculate num_corrections averaged with exponential decay
            group.loc[group['N_iter'] == 0,'num_corrections'] = 1
            group['num_corrections_with_decay'] = group['num_corrections'].transform(lambda x: x.ewm(halflife=cooling_half_life,adjust=False).mean())
            # group = group[group.N_iter >= cycles_averaging_buffer]
            # keep accepted_fraction of entries with the lowest max_num_corrections_in_last_cycles
            group.sort_values(by='num_corrections_with_decay', inplace=True)
            group = group.iloc[:int(len(group)*accepted_fraction)]
            errors_per_cycle_per_qubit_list.append(group_name)
            energy_density_list.append(np.average(group.energy_density))
            energy_density_std_list.append(np.std(group.energy_density)/np.sqrt(len(group.energy_density)/(cooling_half_life/np.log(2))))
        plt.errorbar(errors_per_cycle_per_qubit_list, energy_density_list, yerr=energy_density_std_list, linestyle='None', marker=marker, color=color, label=f'{accepted_fraction}')
        b, a = np.polyfit(errors_per_cycle_per_qubit_list, energy_density_list, deg=1)
        xseq = np.linspace(0, max(errors_per_cycle_per_qubit_list), num=2)
        plt.plot(xseq, a + b * xseq, linestyle='--', color=color, lw=1)
    # plt.xscale('log')
    # plt.yscale('log')
    plt.xlabel('Errors per cycle per qubit', fontsize=str(20), fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('$e_\mathrm{steady}$', fontsize=str(20), fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().set_ylim(bottom=0.)
    leg = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), ncol=2)
    leg.set_title('Accepted fraction', prop={'size': 15, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig(f'graphs/KSL_energy_vs_error_rate_steps_{trotter_steps}_sites_x_{num_sites_x}_sites_y_{num_sites_y}.pdf')
    plt.show()
    print()