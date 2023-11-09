import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

J = 1.0
h = 0.5
V = 0.0
Ns = 100
Nt = None#100,None

results_df = pd.read_csv("results_python_energy_density_vs_cycle.csv")

if Nt is None:
    results_df = results_df.query(f"V == {V} & h == {h} & J == {J} & Ns == {Ns}")
else:
    results_df = results_df.query(f"V == {V} & h == {h} & J == {J} & Ns == {Ns} & Nt == {Nt}")


# results_df = results_df[results_df['T'].isin([12.5,25.,50.,100.,200.,400.,800.])]
# results_df = results_df[results_df['T'].isin(np.arange(10.,110.,10))]
# results_df = results_df[results_df['T'].isin([10.,20.,30.,40.,50.,60.])]
results_df = results_df[results_df['T'].isin([30.,50.,70.,90.])]
# results_df = results_df[results_df['T'].isin([10.,20.,40.,100.])]


markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])


steady_state_energy_density = {}


with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    fig, ax = plt.subplots()
    groups = results_df.groupby("T")
    for T, group in groups:
        marker = next(markers)
        color = next(colors)
        group = group[group.Nt == group.Nt.max()]
        group_up_to_10_cycles = group.query("N_iter<=10")
        plt.plot(group_up_to_10_cycles.N_iter, group_up_to_10_cycles.energy_density_spatial, linestyle='-', marker=marker,
                 color=color, label=f'{T}')
        steady_state_energy_density[T] = group.energy_density_spatial.iloc[-1]
    ax.set_yscale('log')
    plt.xlabel('Cycle', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), ncol=len(groups)//2)
    l.set_title(title='$T$',
                prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
    plt.tight_layout()
    plt.savefig(f'graphs/energy_vs_cycle_Ns_{Ns}_J_{J}_h_{h}_V_{V}_Nt_{Nt}.pdf')
