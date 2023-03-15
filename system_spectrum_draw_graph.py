import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

trotter_steps = 100
cycles = 1000
J = 1.0
V = 0.0

results_df = pd.read_csv("/Users/giladkishony/Dropbox/GILAD/Keva/phd/quantum computation/Periodic Unitaries and Measurements/TestCode/results_energy_density_vs_error_rate.csv")

results_df = results_df.query(f"Nt == {trotter_steps} & N_iter == {cycles} & V == {V} & J == {J} & Ns <= 6 & h > 0")


markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])

with sns.axes_style("whitegrid"):
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    groups = results_df.groupby(["Ns"])
    for Ns, group in groups:
        marker = next(markers)
        # plt.plot(group.errors_per_cycle_per_qubit, group.energy_density, linestyle='None', marker=marker, label=f'J = {J}, h = {h}')
        yerr = np.array(group.energy_density_std)
        yerr[yerr>group.energy_density] = 0.9999*group.energy_density[yerr>group.energy_density]
        plt.errorbar(group.h, group.energy_density, yerr=yerr, linestyle='--', marker=marker, label=f'N = {Ns}')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel('h', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
    plt.tight_layout()
    plt.savefig(f'graphs/energy_vs_h_steps_{trotter_steps}_cycles_{cycles}_J_{J}_V_{V}.pdf')
    plt.show()