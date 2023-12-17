import itertools
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

results_df = pd.read_csv("KSL_complex_chern.csv")

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])

# marker = next(markers)
# color = next(colors)

with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots()
    plt.rcParams['legend.title_fontsize'] = 50
    plt.rcParams["font.family"] = "Times New Roman"
    groups = results_df.groupby(["num_cooling_sublattices"])
    for num_cooling_sublattices, group in groups:
        marker = next(markers)
        color = next(colors)
        # sort by T
        group = group.sort_values(by=['T'])
        ax.semilogy(group['T'], group.energy_density, linestyle='-', color=color, marker=marker, markersize=10, label=f'{num_cooling_sublattices}')
    plt.xlabel('$T$', fontsize='50', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize='50', fontname='Times New Roman')#, fontweight='bold')
    ax.yaxis.set_label_coords(-0.25, 0.4)
    plt.tick_params(axis='both', which='major', labelsize=38)
    # l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
    # l.set_title(title='auxiliary sites per unit cell',
    #             prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
    fig.tight_layout(pad=0.5)
    plt.savefig(f'graphs/KSL_energy_density_vs_T.pdf', transparent=True)
    plt.show()