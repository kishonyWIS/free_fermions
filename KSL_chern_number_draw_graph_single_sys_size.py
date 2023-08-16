import itertools
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

results_df = pd.read_csv("KSL_complex_chern.csv")
n_k_points = 85
results_df = results_df.query(f"n_k_points == {n_k_points}")

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])

with sns.axes_style("whitegrid"):
    plt.rcParams['legend.title_fontsize'] = 25
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure()
    marker = next(markers)
    color = next(colors)
    plt.plot(results_df['T'], results_df.system_chern_number, linestyle='-', marker=marker, color=color, label=f'system')
    plt.plot(results_df['T'], results_df.bath_chern_number, linestyle='--', marker=marker, color=color, label=f'bath')
    plt.xlabel('$T$', fontsize='27', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('$\\nu$', fontsize='27', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=20)
    l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=20), loc='upper left', ncol=1, columnspacing=0.3, handletextpad=0.3, borderpad=0.3)
    # l.set_title(title='system size',
    #             prop=mpl.font_manager.FontProperties(family='Times New Roman', size=22))
    plt.tight_layout()
    plt.savefig(f'graphs/KSL_chern_number_vs_T.pdf')
    plt.show()