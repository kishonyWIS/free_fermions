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


T_list = [30.,50.,70.,90.]# [10.,20.,30.,40.,50.,60.]
results_df = results_df[results_df['T'].isin(T_list)]
# results_df = results_df[results_df['T'].isin([30.,50.,70.,90.])]


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
    fig_inset, ax_inset = plt.subplots()
    groups = results_df.groupby("T")
    for T, group in groups:
        marker = next(markers)
        color = next(colors)
        group = group[group.Nt == group.Nt.max()]
        group_up_to_5_cycles = group.query("N_iter<=5")
        steady_state_energy_density[T] = group.energy_density.min()
        energy_above_steady_state = group_up_to_5_cycles.energy_density-steady_state_energy_density[T]

        # remove zeros
        N_iter = group_up_to_5_cycles.N_iter[energy_above_steady_state>0]
        energy_above_steady_state = energy_above_steady_state[energy_above_steady_state>0]

        ax.plot(N_iter, energy_above_steady_state, linestyle='-', marker=marker,
                 color=color, label=f'{T}')

        ax_inset.plot(T,steady_state_energy_density[T], linestyle='None', marker=marker, color=color, markersize=15)
    ax.set_yscale('log')
    plt.sca(ax)
    plt.xlabel('$n$ (cycle no.)', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('$e - e_\mathrm{steady}$', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.savefig(f'graphs/energy_convergence_vs_cycle_Ns_{Ns}_J_{J}_h_{h}_V_{V}_Nt_{Nt}.pdf')

    ax_inset.set_yscale('log')
    ax_inset.set_xticks(T_list)
    # ax_inset.set_yticks([1e-3, 1e-2, 1e-1])

    plt.sca(ax_inset)
    plt.xlabel('$T$', fontsize='40', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('$e_\mathrm{steady}$', fontsize='40', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=30)
    plt.tight_layout()
    plt.savefig(f'graphs/energy_convergence_vs_cycle_Ns_{Ns}_J_{J}_h_{h}_V_{V}_Nt_{Nt}_inset.pdf', transparent=True)