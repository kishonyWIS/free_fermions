import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

simulation_mode = "single_particle" #"stochastic_schrodinger" "single_particle"
trotter_steps = 100
cycles = 1000
num_sites = 100
V = 0.0

if simulation_mode == "stochastic_schrodinger":
    results_df = pd.read_csv("/Users/giladkishony/Dropbox/GILAD/Keva/phd/quantum computation/Periodic Unitaries and Measurements/TestCode/results_energy_density_vs_error_rate.csv")
else:
    results_df = pd.read_csv("results_python_energy_density_vs_error_rate.csv")

results_df = results_df.query(f"Ns == {num_sites} & Nt == {trotter_steps} & N_iter == {cycles} & V == {V}")

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])

with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure()
    groups = results_df.groupby(["J", "h"])
    for (J,h), group in groups:
        marker = next(markers)
        color = next(colors)
        plt.plot(group.errors_per_cycle_per_qubit, group.energy_density, linestyle='None', marker=marker, color=color, label=f'J = {J}, h = {h}')
        # plt.errorbar(group.errors_per_cycle_per_qubit, group.energy_density, yerr=group.energy_density_std, linestyle='None', marker=marker, label=f'J = {J}, h = {h}')

        b, a = np.polyfit(group.errors_per_cycle_per_qubit, group.energy_density, deg=1)
        xseq = np.linspace(0, max(group.errors_per_cycle_per_qubit), num=2)
        plt.plot(xseq, a + b * xseq, linestyle='--', color=color, lw=1);
    plt.xlabel('Errors per cycle per qubit', fontsize=str(20), fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('$e_\mathrm{steady}$', fontsize=str(20), fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
    plt.tight_layout()
    plt.savefig(f'graphs/energy_vs_error_rate_{simulation_mode}_steps_{trotter_steps}_cycles_{cycles}_sites_{num_sites}_V_{V}.pdf')
    plt.show()