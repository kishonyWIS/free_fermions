import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

trotter_steps = 100
num_sites = 4
V = 0.0

with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.family"] = "Times New Roman"
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey='row', figsize=(6,6))

    for i_J_j,(J,h) in enumerate([(0.5,1.0), (1.0,0.5)]):
        results_df = pd.read_csv(
            "/Users/giladkishony/Dropbox/GILAD/Keva/phd/quantum computation/Periodic Unitaries and Measurements/TestCode/results_energy_density_single_time_trace.csv")

        results_df = results_df.query(f"Ns == {num_sites} & Nt == {trotter_steps} & V == {V} & h == {h} & J == {J}")

        ax2 = axes[0, i_J_j]
        # ax2 = ax.twinx()
        ax2.plot(results_df.cycle, (-1)**results_df.bath_parity, linestyle='--', label=f'Bath parity', color="red")
        if i_J_j==0:
            ax2.set_ylabel("$\prod_j\\tau^z_j$", fontsize='20', fontname='Times New Roman', math_fontfamily='stixsans')
        ax2.set_title(f'J={J}, h={h}', fontsize='20', fontname='Times New Roman')
        # ax2.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), loc='upper right')
        ax2.set_yticks([-1.0,1.0])
        ax2.set_ylim([-1.1, 1.1])

        ax = axes[1, i_J_j]
        ax.semilogy(results_df.cycle, results_df.energy_density, linestyle='-', color='blue', label=f'Energy density')
        if i_J_j==0:
            ax.set_ylabel("Energy density", fontsize='20', fontname='Times New Roman')
        ax.set_xlabel('$n$ (cycle no.)', fontsize='20', fontname='Times New Roman')
        # ax.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), loc='upper left')
        # ax.set_xticks([])

        ax.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(f'graphs/single_time_trace_steps_{trotter_steps}_sites_{num_sites}_V_{V}.pdf')
    plt.show()