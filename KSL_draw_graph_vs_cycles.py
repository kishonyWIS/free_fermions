import itertools
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

results_df = pd.read_csv("KSL_complex_vs_cycles.csv")

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])

with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.figure()
    groups_T = results_df.groupby(["T"])
    for T, group_T in groups_T:
        marker = next(markers)
        color = next(colors)
        groups_system_size = group_T.groupby(["n_k_points"])
        for n_k_points, group_system_size in groups_system_size:
            plt.semilogy(group_system_size['cycles'][:8], group_system_size.energy_density[:8] - min(group_system_size.energy_density), marker=marker, color=color, label=f'{T}, {n_k_points}')
    plt.xlabel('$n$ (cycle no.)', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('$e - e_\mathrm{steady}$', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(f'graphs/KSL_energy_density_vs_cycles.pdf')
    plt.show()