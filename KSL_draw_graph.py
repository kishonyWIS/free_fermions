import itertools
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

trotter_steps = 100

results_df = pd.read_csv("KSL.csv")

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])


with sns.axes_style("whitegrid"):
    plt.rcParams['legend.title_fontsize'] = 20
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure()
    groups = results_df.groupby(["num_cooling_sublattices"])
    for num_cooling_sublattices, group in groups:
        marker = next(markers)
        plt.semilogy(group['T'], group.energy_density, linestyle='None', marker=marker, label=f'{num_cooling_sublattices}')
    plt.xlabel('T', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
    l.set_title(title='auxiliary sites per unit cell',
                prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
    plt.tight_layout()
    plt.savefig(f'graphs/KSL_energy_density_vs_T.pdf')
    plt.show()