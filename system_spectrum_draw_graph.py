import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

Ns = 16
periodic_bc = False

results_df = pd.read_csv("/Users/giladkishony/Dropbox/GILAD/Keva/phd/quantum computation/Periodic Unitaries and Measurements/TestCode/results_system_spectrum.csv")

results_df = results_df.query(f"Ns == {Ns} & periodic_bc == {periodic_bc}")
results_df['h_minus_J'] = results_df.h-results_df.J
results_df = results_df[['h_minus_J', 'energies', 'V']]

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
colors = itertools.cycle(['b', 'r'])

with sns.axes_style("whitegrid"):
    plt.rcParams["font.family"] = "Times New Roman"

    groups_V = results_df.groupby(["V"])
    for V, group_V in groups_V:
        fig, ax = plt.subplots()
        marker = next(markers)
        color = next(colors)
        energies_groups = group_V.groupby('h_minus_J').energies.apply(np.array)
        ax.plot(energies_groups.index, np.stack(energies_groups.values), linestyle='--', marker=marker, color=color)
        plt.xlabel('h-J', fontsize='20', fontname='Times New Roman')
        plt.ylabel('Energy', fontsize='20', fontname='Times New Roman')
        ax.tick_params(axis='both', which='major', labelsize=15)
        # ax.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
        plt.tight_layout()
        plt.savefig(f'graphs/system_spectrum_periodic_bc_{periodic_bc}_Ns_{Ns}_V_{V}.pdf')
    plt.show()
