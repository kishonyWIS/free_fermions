import itertools
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

results_df = pd.read_csv("KSL_complex_vs_cycles.csv")

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])

with sns.axes_style("whitegrid"):
    plt.rcParams['legend.title_fontsize'] = 20
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure()
    groups_sublattices = results_df.groupby(["num_cooling_sublattices"])
    for num_cooling_sublattices, group_sublattices in groups_sublattices:
        marker = next(markers)
        groups_system_size = group_sublattices.groupby(["n_k_points"])
        colors = itertools.cycle(['b', 'r', 'g'])
        for n_k_points, group_system_size in groups_system_size:
            color = next(colors)
            plt.semilogy(group_system_size['cycles'], group_system_size.energy_density, linestyle='None', marker=marker, color=color, label=f'{num_cooling_sublattices}, {n_k_points}')
    plt.xlabel('Cycles', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
    l.set_title(title='auxiliary sites per unit cell, system size',
                prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
    plt.tight_layout()
    plt.savefig(f'graphs/KSL_energy_density_vs_cycles.pdf')
    plt.show()