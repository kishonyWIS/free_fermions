import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns


with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['legend.title_fontsize'] = 20
    plt.rcParams["font.family"] = "Times New Roman"

    labels = {True: 'With flux correction', False: 'Without flux correction'}
    for flux_correction in [True, False]:
        results_df = pd.read_csv(f"KSL_results_single_time_trace{'' if flux_correction else '_no_flux_correction'}.csv").query('N_iter <= 45')
        plt.plot(results_df.N_iter, results_df.energy_density, label=labels[flux_correction])
    plt.xlabel('Cycle', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), loc='upper left', columnspacing=0.3, handletextpad=0.3, borderpad=0.3)
    plt.tight_layout()
    plt.savefig(f'../graphs/KSL_single_time_trace.pdf')

plt.show()