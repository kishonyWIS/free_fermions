import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np

results_df = pd.read_csv("/Users/giladkishony/PycharmProjects/free_fermions/KSL_complex_vs_cycles.csv")


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
    groups_T = results_df.groupby("T")
    T_list = []
    energy_density_list = []
    for T, group_T in groups_T:
        color = next(colors)
        marker = next(markers)
        T_list.append(T)
        energy_density_list.append(group_T.energy_density.min())
        plt.semilogy(T_list[-1], energy_density_list[-1], linestyle='-', marker=marker, color=color)
    plt.xlabel('$T$', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.ylabel('Energy density', fontsize='20', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    plt.xlim([0,55])
    plt.savefig(f'graphs/KSL_steady_state_energy_vs_T.pdf')

    plt.show()
