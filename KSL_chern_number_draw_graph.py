import itertools
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

results_df = pd.read_csv("KSL_complex_chern.csv")

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])

with sns.axes_style("whitegrid"):
    plt.rcParams['legend.title_fontsize'] = 25
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure()
    groups = results_df.groupby(["n_k_points"])
    for n_k_points, group in groups:
        marker = next(markers)
        color = next(colors)
        plt.plot(group['T'], group.system_chern_number, linestyle='-', marker=marker, color=color, label=f'{n_k_points}')
        plt.plot(group['T'], -group.bath_chern_number, linestyle='--', marker=marker, color=color)
    plt.xlabel('$T$', fontsize='27', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('$\\nu$', fontsize='27', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=20)
    l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=20), loc='lower right', ncol=2, columnspacing=0.5, handletextpad=0.5)
    l.set_title(title='system size',
                prop=mpl.font_manager.FontProperties(family='Times New Roman', size=22))
    plt.tight_layout()
    plt.savefig(f'graphs/KSL_chern_number_vs_T.pdf')
    plt.show()