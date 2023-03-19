import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

trotter_steps = 100
cycles = 1000
J = 0.5
h = 1.0
V = 0.1

results_df = pd.read_csv("/Users/giladkishony/Dropbox/GILAD/Keva/phd/quantum computation/Periodic Unitaries and Measurements/TestCode/results_energy_density_vs_system_size.csv")

results_df = results_df.query(f"Nt == {trotter_steps} & N_iter == {cycles} & V == {V} & h == {h} & J == {J}")


markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])

with sns.axes_style("whitegrid"):
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    groups = results_df.groupby(["errors_per_cycle_per_qubit"])
    for errors_per_cycle_per_qubit, group in groups:
        marker = next(markers)
        # plt.plot(group.errors_per_cycle_per_qubit, group.energy_density, linestyle='None', marker=marker, label=f'J = {J}, h = {h}')
        yerr = np.array(group.energy_density_std)
        yerr[yerr>group.energy_density] = 0.9999*group.energy_density[yerr>group.energy_density]
        plt.errorbar(group.Ns, group.energy_density, yerr=yerr, linestyle='--', marker=marker, label=f'{errors_per_cycle_per_qubit} errors per cycle per qubit')
    ax.set_yscale('log')
    plt.xlabel('System size', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
    plt.tight_layout()
    plt.savefig(f'graphs/energy_vs_system_size_steps_{trotter_steps}_cycles_{cycles}_J_{J}_h_{h}_V_{V}.pdf')
    plt.show()