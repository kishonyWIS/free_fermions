import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

trotter_steps = 400
cycles = [50, 200]
num_sites_x = [10]
num_sites_y = [10]

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])

with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure()
    periodic_bc_labels = {'True': 'Periodic', 'False': 'Open', '(True, False)': 'Mixed'}

    for periodic_bc in ['True']:
        results_df = pd.read_csv("KSL_results_from_cluster_larger_B.csv")
        results_df = results_df.query(
            f"num_sites_x in {num_sites_x} & num_sites_y in {num_sites_y} & Nt == {trotter_steps} & N_iter in {cycles}")
        results_df = results_df[results_df.periodic_bc == periodic_bc]
        marker = next(markers)
        color = next(colors)
        # aggregate energy_density and energy_density_std entries with same errors_per_cycle_per_qubit
        errors_per_cycle_per_qubit_list = []
        energy_density_list = []
        energy_density_std_list = []
        for group_name, group in results_df.groupby('errors_per_cycle_per_qubit'):
            errors_per_cycle_per_qubit_list.append(group_name)
            energy_density_list.append(np.average(group.energy_density, weights=group.N_iter))
            energy_density_std_list.append(np.sqrt(np.average(group.energy_density_std**2, weights=group.N_iter)))
        plt.errorbar(errors_per_cycle_per_qubit_list, energy_density_list, yerr=energy_density_std_list, linestyle='None', marker=marker, color=color, label=periodic_bc_labels[periodic_bc])
        b, a = np.polyfit(errors_per_cycle_per_qubit_list, energy_density_list, deg=1)
        xseq = np.linspace(0, max(errors_per_cycle_per_qubit_list), num=2)
        plt.plot(xseq, a + b * xseq, linestyle='--', color=color, lw=1);
    plt.xlabel('Errors per cycle per qubit', fontsize=str(20), fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('$e_\mathrm{steady}$', fontsize=str(20), fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().set_ylim(bottom=0.)
    # leg = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
    # leg.set_title('Boundary conditions', prop={'size': 15, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig(f'graphs/KSL_energy_vs_error_rate_steps_{trotter_steps}_sites_x_{num_sites_x}_sites_y_{num_sites_y}.pdf')
    plt.show()
    print()