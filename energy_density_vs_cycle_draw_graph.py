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

results_df = pd.read_csv("results_python_energy_density_vs_cycle.csv")

results_df = results_df.query(f"V == {V} & h == {h} & J == {J} & Ns == {Ns} & N_iter<=10")


# results_df = results_df[results_df['T'].isin([12.5,25.,50.,100.,200.,400.,800.])]
results_df = results_df[results_df['T'].isin([12.5,50.,200.,800.])]


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
        plt.plot(group.N_iter, group.energy_density, linestyle='-', marker=marker, color=color,
                     label=f'{T}')
        steady_state_energy_density[T] = group.energy_density.iloc[-1]
    ax.set_yscale('log')
    plt.xlabel('Cycle', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), ncol=len(groups)//2)
    l.set_title(title='$T$',
                prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
    plt.tight_layout()
    plt.savefig(f'graphs/energy_vs_cycle_Ns_{Ns}_J_{J}_h_{h}_V_{V}.pdf')



    # fig, ax = plt.subplots()
    # plt.plot(steady_state_energy_density.keys(), steady_state_energy_density.values(), marker='o', color='k', linestyle='None', markersize=10)
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # plt.xlabel('T', fontsize='40', fontname='Times New Roman')#, fontweight='bold')
    # plt.ylabel('Energy density', fontsize='40', fontname='Times New Roman')#, fontweight='bold')
    # plt.tick_params(axis='both', which='major', labelsize=30)
    # plt.tight_layout()
    # plt.savefig(f'graphs/steady_state_energy_vs_T_Ns_{Ns}_J_{J}_h_{h}_V_{V}.pdf')

    plt.show()
