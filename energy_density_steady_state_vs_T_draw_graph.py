import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

J = 0.75
h = 0.75
V = 0.0
Ns_list = [4,10,40,100]

results_df = pd.read_csv("results_python_energy_density_vs_cycle.csv")

results_df = results_df.query(f"V == {V} & h == {h} & J == {J}")

results_df = results_df[results_df['T'].isin(np.arange(10.,110.,10))]
results_df = results_df[results_df['Ns'].isin(Ns_list)]

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])


with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    fig, ax = plt.subplots()
    # for each Ns, plot the minimal energy density as a function of T
    groups_Ns = results_df.groupby("Ns")
    for Ns, group_Ns in groups_Ns:
        marker = next(markers)
        color = next(colors)
        groups_T = group_Ns.groupby("T")
        T_list = []
        energy_density_list = []
        for T, group_T in groups_T:
            group_T = group_T[group_T.Nt == group_T.Nt.max()]
            T_list.append(T)
            energy_density_list.append(group_T.energy_density.min())
        plt.semilogy(T_list, energy_density_list, linestyle='-', marker=marker, color=color, label=f'{Ns}')
    plt.xlabel('$T$', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15), ncol=len(groups_Ns)//2)
    l.set_title(title='system size', prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
    plt.tight_layout()
    plt.xlim([0,105])
    plt.savefig(f'graphs/steady_state_energy_vs_T_J_{J}_h_{h}_V_{V}.pdf')

    plt.show()
