import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

simulation_mode = "stochastic_schrodinger" #"stochastic_schrodinger" "single_particle"
trotter_steps = 100
cycles = 1000
num_sites = 8
V = 0.0

if simulation_mode == "stochastic_schrodinger":
    results_df = pd.read_csv("/Users/giladkishony/Dropbox/GILAD/Keva/phd/quantum computation/Periodic Unitaries and Measurements/TestCode/results_energy_density_vs_error_rate.csv")
else:
    results_df = pd.read_csv("results_python_energy_density_vs_error_rate.csv")

results_df = results_df.query(f"Ns == {num_sites} & Nt == {trotter_steps} & N_iter == {cycles} & V == {V}")

font_factor = 1.8

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])

with sns.axes_style("whitegrid"):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure()
    groups = results_df.groupby(["J", "h"])
    for (J,h), group in groups:
        marker = next(markers)
        # plt.plot(group.errors_per_cycle_per_qubit, group.energy_density, linestyle='None', marker=marker, label=f'J = {J}, h = {h}')
        plt.errorbar(group.errors_per_cycle_per_qubit, group.energy_density, yerr=group.energy_density_std, linestyle='None', marker=marker, label=f'J = {J}, h = {h}')
    plt.xlabel('Errors per cycle per qubit', fontsize=str(20*font_factor), fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize=str(20*font_factor), fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15*font_factor)
    plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15*font_factor))
    plt.tight_layout()
    plt.savefig(f'graphs/energy_vs_error_rate_{simulation_mode}_steps_{trotter_steps}_cycles_{cycles}_sites_{num_sites}_V_{V}.pdf')
    plt.show()