import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

trotter_steps = 100
num_sites = 8
V = 0.0
J = 1.0
h = 0.5

results_df = pd.read_csv("/Users/giladkishony/Dropbox/GILAD/Keva/phd/quantum computation/Periodic Unitaries and Measurements/TestCode/results_energy_density_single_time_trace.csv")

results_df = results_df.query(f"Ns == {num_sites} & Nt == {trotter_steps} & V == {V} & h == {h} & J == {J}")

with sns.axes_style("whitegrid"):
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots()
    ax.semilogy(results_df.cycle, results_df.energy_density, linestyle='-', color='blue', label=f'Energy density')
    ax.set_xlabel('Cycle', fontsize='20', fontname='Times New Roman')
    # ax.set_ylabel("Energy density", color="blue", fontsize='20', fontname='Times New Roman')
    ax.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), loc='upper left')
    plt.ylim([5e-4, 1e0])

    ax2 = ax.twinx()
    ax2.plot(results_df.cycle, results_df.bath_parity, linestyle='--', label=f'Bath parity', color="red")
    # ax2.set_ylabel("Bath parity", color="red", fontsize='20', fontname='Times New Roman')
    ax2.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), loc='upper right')
    plt.yticks([0,1])
    plt.ylim([-0.1, 2.5])

    ax.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(f'graphs/single_time_trace_steps_{trotter_steps}_sites_{num_sites}_V_{V}_h_{h}_J_{J}.pdf')
    plt.show()