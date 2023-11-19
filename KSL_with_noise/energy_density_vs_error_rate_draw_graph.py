import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

trotter_steps = 100
cycles = 50
num_sites_x = 10
num_sites_y = 10

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

    for periodic_bc in ['True', 'False', '(True, False)']:
        results_df = pd.read_csv("KSL_results_averaged.csv")
        results_df = results_df.query(
            f"num_sites_x == {num_sites_x} & num_sites_y == {num_sites_y} & Nt == {trotter_steps} & N_iter == {cycles}")
        results_df = results_df[results_df.periodic_bc == periodic_bc]
        marker = next(markers)
        color = next(colors)
        # plt.plot(results_df.errors_per_cycle_per_qubit, results_df.energy_density, linestyle='None', marker=marker, color=color)
        plt.errorbar(results_df.errors_per_cycle_per_qubit, results_df.energy_density, yerr=results_df.energy_density_std, linestyle='None', marker=marker, color=color, label=periodic_bc_labels[periodic_bc])

        b, a = np.polyfit(results_df.errors_per_cycle_per_qubit, results_df.energy_density, deg=1)
        xseq = np.linspace(0, max(results_df.errors_per_cycle_per_qubit), num=2)
        plt.plot(xseq, a + b * xseq, linestyle='--', color=color, lw=1);
    plt.xlabel('Errors per cycle per qubit', fontsize=str(20), fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('$e_\mathrm{steady}$', fontsize=str(20), fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.gca().set_ylim(bottom=0.)
    leg = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
    leg.set_title('Boundary conditions', prop={'size': 15, 'family': 'Times New Roman'})
    plt.tight_layout()
    plt.savefig(f'graphs/KSL_energy_vs_error_rate_steps_{trotter_steps}_cycles_{cycles}_sites_x_{num_sites_x}_sites_y_{num_sites_y}.pdf')
    plt.show()
    print()