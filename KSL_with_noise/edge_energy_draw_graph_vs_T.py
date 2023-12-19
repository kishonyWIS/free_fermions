import itertools
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl

g0 = 0.5
B1 = 0.
B0 = 7.
J = 1.
kappa = 1.
periodic_bc = "'(True, False)'" # 'TRUE', "'(True, False)'", True
initial_state = "ground"
draw_spatial_energy = None
errors_per_cycle_per_qubit = 0.

results_df = pd.read_csv("KSL_results_from_cluster_larger_B.csv").query(f"kappa == {kappa} & g == {g0} & B == {B0} & T*40 == Nt & errors_per_cycle_per_qubit == {errors_per_cycle_per_qubit} & initial_state == '{initial_state}' & periodic_bc == {periodic_bc}")
results_df['energy_per_circumference'] = results_df.energy_density * results_df.num_sites_y
results_df['energy'] = results_df.energy_density * results_df.num_sites_x * results_df.num_sites_y

# for each value of T, plot energy vs num_sites_y at num_sites_x==10, and also energy vs num_sites_x at num_sites_y==10
T_list = sorted(results_df['T'].unique())
markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])
with sns.axes_style("whitegrid"):
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.family"] = "Times New Roman"
    for T in T_list:
        color = next(colors)
        marker = next(markers)
        results_df_T = results_df.query(f"T == {T}")
        results_df_T_vs_x = results_df_T.query(f"num_sites_y == 14")
        results_df_T_vs_x = results_df_T_vs_x.sort_values(by=['num_sites_x'])
        results_df_T_vs_y = results_df_T.query(f"num_sites_x == 14")
        results_df_T_vs_y = results_df_T_vs_y.sort_values(by=['num_sites_y'])
        plt.plot(results_df_T_vs_x["num_sites_x"], results_df_T_vs_x["energy"], linestyle=None, marker=marker, color=color, label=f'T={T}, num_sites_y=10')
        plt.plot(results_df_T_vs_y["num_sites_y"], results_df_T_vs_y["energy"], linestyle='--', marker=marker, color=color, label=f'T={T}, num_sites_x=10')
plt.legend()
plt.xscale('log')
plt.yscale('log')
print()

# plot energy per circumference vs num_sites_x for different T and num_sites_y == 10
# results_df = results_df.query("num_sites_y == 10")
# plt.figure()
# for T in sorted(results_df['T'].unique()):
#     results_df_T = results_df.query(f"T == {T}")
#     results_df_T = results_df_T.sort_values(by=['num_sites_x'])
#     plt.plot(results_df_T["num_sites_x"], results_df_T["energy_per_circumference"], label=f"T = {T}")
# plt.yscale('log')
# plt.xlabel("num_sites_x")
# plt.ylabel("Energy per circumference")
# plt.legend()
# print()

# results_df = results_df.query("num_sites_x == 10")
# plt.figure()
# for num_sites_y in sorted(results_df.num_sites_y.unique()):
#     results_df_y = results_df.query(f"num_sites_y == {num_sites_y}")
#     results_df_y = results_df_y.sort_values(by=['T'])
#     plt.plot(results_df_y["T"], results_df_y["energy_per_circumference"], label=f"num_sites_y = {num_sites_y}")
# plt.yscale('log')
# plt.xlabel("T")
# plt.ylabel("Energy per circumference")
# plt.legend()
# print()