import itertools
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

num_sites_x_list = [None]
num_sites_y_list = list(range(2,20,2))
min_ny_fit = 10
g0 = 0.5
B1 = 0.
B0 = 5.
J = 1.
kappa = 1.
periodic_bc = "'(True, False)'" # 'TRUE', "'(True, False)'", True
initial_state = "ground"
draw_spatial_energy = None
errors_per_cycle_per_qubit = 0.

results_df = pd.read_csv("KSL_results_from_cluster.csv")

T_list = np.arange(1,25,1)
markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.family"] = "Times New Roman"
    for num_sites_x in num_sites_x_list:
        marker = next(markers)
        colors = itertools.cycle(prop_cycle.by_key()['color'])
        for T in T_list:
            color = next(colors)
            trotter_steps = int(T * 40)
            results_df_T = results_df.query(f"Nt == {trotter_steps} & T == {T} & kappa == {kappa} & g == {g0} & B == {B0}")
            if num_sites_x is not None:
                results_df_T = results_df_T.query(f"num_sites_x == {num_sites_x}")
            else:
                results_df_T = results_df_T.query(f"num_sites_x == num_sites_y")
            results_df_T = results_df_T.query(f"periodic_bc == {periodic_bc} & errors_per_cycle_per_qubit == {errors_per_cycle_per_qubit}")
            results_df_T = results_df_T.sort_values(by=['num_sites_y'])
            energy = results_df_T.energy_density * results_df_T.num_sites_x * results_df_T.num_sites_y
            energy_per_circumference = energy / results_df_T.num_sites_x

            plt.figure(1)
            plt.plot(results_df_T.num_sites_y, energy_per_circumference, linestyle=None, marker=marker, color=color, label=f'T={T}, num_sites_x={num_sites_x}')

            b, a = np.polyfit(results_df_T.num_sites_y[results_df_T.num_sites_y >= min_ny_fit], energy_per_circumference[results_df_T.num_sites_y >= min_ny_fit], deg=1)
            xseq = np.linspace(0, max(results_df_T.num_sites_y), num=2)
            plt.plot(xseq, a + b * xseq, linestyle='--', lw=1, color=color)

            plt.figure(2)
            plt.scatter(T, a)

            plt.figure(3)
            plt.scatter(T, b)

    plt.figure(1)
    plt.legend(title='T')
    plt.xlabel('num_sites_y', fontsize=str(20), fontname='Times New Roman')
    plt.ylabel('$E/W$', fontsize=str(20), fontname='Times New Roman')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=5))
    plt.tight_layout()

    plt.figure(2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('T', fontsize=str(20), fontname='Times New Roman')
    plt.ylabel('$e_\mathrm{edge}$', fontsize=str(20), fontname='Times New Roman')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()

    plt.figure(3)
    plt.yscale('log')
    plt.xlabel('T', fontsize=str(20), fontname='Times New Roman')
    plt.ylabel('$e_\mathrm{bulk}$', fontsize=str(20), fontname='Times New Roman')
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.tight_layout()
    print()