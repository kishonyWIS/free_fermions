import itertools

import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit

J = 1.0
h = 0.5
V = 0.1
Ns = 8
T_list = [30.,50.,70.,90.]
periodic_bc = False
drop_bath_parity_not_0 = True

sns.set_style("whitegrid")
rc = {"font.family": "serif",
      "mathtext.fontset": "stix",
      "figure.autolayout": True}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]

def edit_graph(xlabel, ylabel, legend_title=None, colorbar_title=None, colormap=None, colorbar_args={}, tight=True, ylabelpad=None, text_scale=1.0):
    plt.xlabel(xlabel, fontsize=str(int(20*text_scale)), fontname='Times New Roman')
    plt.ylabel(ylabel, fontsize=str(int(20*text_scale)), fontname='Times New Roman', labelpad=ylabelpad)
    plt.tick_params(axis='both', which='major', labelsize=int(15*text_scale))
    if legend_title:
        l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=int(15*text_scale)))
        l.set_title(title='$T$', prop=mpl.font_manager.FontProperties(family='Times New Roman', size=int(18*text_scale)))
    if colorbar_title:
        if colormap:
            cmap = mpl.cm.ScalarMappable(norm=None, cmap=colormap)
            cmap.set_array([])
            cbar = plt.colorbar(cmap,**colorbar_args)
        else:
            cbar = plt.colorbar(**colorbar_args)
        cbar.set_label(colorbar_title, fontsize=str(int(20*text_scale)), fontname='Times New Roman')
        cbar.ax.tick_params(labelsize=int(15*text_scale))
    if tight:
        plt.tight_layout()



if V == 0.0:
    results_df = pd.read_csv("results_python_energy_density_vs_cycle.csv")
else:
    results_df = pd.read_csv("results_energy_density_vs_cycle.csv")#

results_df = results_df.query(f"V == {V} & h == {h} & J == {J} & Ns == {Ns} & periodic_bc == {periodic_bc} & T in {T_list}")
if drop_bath_parity_not_0:
    results_df = results_df.query("N_iter <= 6")

# group by seed and drop all rows if any bath parity is not 0
if drop_bath_parity_not_0:
    results_df = results_df.groupby('seed').filter(lambda x: (x.bath_parity == 0).all())

Nt = results_df.Nt.max()
results_df = results_df.query(f"Nt == {Nt}")

results_df['infidelity'] = 1 - results_df['ground_subspace_fidelity']

results_df = results_df.groupby(['T', 'N_iter']).agg(
    energy_density=('energy_density', 'mean'),
    energy_density_std=('energy_density', lambda x: np.std(x) / np.sqrt(len(x))),
    infidelity=('infidelity', 'mean'),
    infidelity_std=('infidelity', lambda x: np.std(x) / np.sqrt(len(x))),
).reset_index()





# same for fidelity
fig, ax = plt.subplots()
fig_inset, ax_inset = plt.subplots()
groups = results_df.groupby("T")

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])

for T, group in groups:
    marker = next(markers)
    color = next(colors)

    steady_state_infidelity = group.infidelity.min()
    steady_state_infidelity_std = group.infidelity_std.min()

    # steady_state_infidelity = group.infidelity[group.N_iter==max(group.N_iter)].values
    # steady_state_infidelity_std = group.infidelity_std[group.N_iter==max(group.N_iter)].values

    infidelity_above_steady_state = group.infidelity.values - steady_state_infidelity

    # remove zeros
    N_iter = group.N_iter[infidelity_above_steady_state>0].values
    infidelity_std = group.infidelity_std[infidelity_above_steady_state>0].values
    infidelity_above_steady_state = infidelity_above_steady_state[infidelity_above_steady_state>0]

    # draw the infidelity_above_steady_state vs N_iter with error bars
    ax.errorbar(group.N_iter, group.infidelity, yerr=group.infidelity_std, linestyle='-', marker=marker,
                color=color, label=f'{T}', markersize=22)
    ax_inset.errorbar(T, steady_state_infidelity, yerr=steady_state_infidelity_std, linestyle='None', marker=marker, color=color, markersize=15)
ax.set_yscale('log')
plt.sca(ax)
edit_graph('$n$ (cycle no.)', '$1-F$', text_scale=3)
plt.savefig(f'graphs/fidelity_convergence_vs_cycle_Ns_{Ns}_J_{J}_h_{h}_V_{V}_Nt_{Nt}_postselectevenbath_{drop_bath_parity_not_0}.pdf',transparent=True)

ax_inset.set_yscale('log')
ax_inset.set_xticks(T_list)

plt.sca(ax_inset)
edit_graph('$T$', '$1-F_\mathrm{steady}$', text_scale=1.5)
plt.savefig(f'graphs/fidelity_convergence_vs_cycle_Ns_{Ns}_J_{J}_h_{h}_V_{V}_Nt_{Nt}_postselectevenbath_{drop_bath_parity_not_0}_inset.pdf', transparent=True)



fig, ax = plt.subplots()
fig_inset, ax_inset = plt.subplots()
groups = results_df.groupby("T")

markers = itertools.cycle(['o', 's', '^', '*', '8', 'p', 'd', 'v'])
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = itertools.cycle(prop_cycle.by_key()['color'])

for T, group in groups:
    marker = next(markers)
    color = next(colors)

    steady_state_energy_density = group.energy_density.min()
    steady_state_energy_density_std = group.energy_density_std.min()

    # steady_state_energy_density = group.energy_density[group.N_iter==max(group.N_iter)].values
    # steady_state_energy_density_std = group.energy_density_std[group.N_iter==max(group.N_iter)].values

    energy_above_steady_state = group.energy_density.values - steady_state_energy_density

    # remove zeros
    N_iter = group.N_iter[energy_above_steady_state>0].values
    energy_density_std = group.energy_density_std[energy_above_steady_state>0].values
    energy_above_steady_state = energy_above_steady_state[energy_above_steady_state>0]

    # draw the energy_density_above_steady_state vs N_iter with error bars
    ax.errorbar(group.N_iter, group.energy_density, yerr=group.energy_density_std, linestyle='-', marker=marker,
                color=color, label=f'{T}', markersize=22)
    ax_inset.errorbar(T, steady_state_energy_density, yerr=steady_state_energy_density_std, linestyle='None', marker=marker, color=color, markersize=15)
ax.set_yscale('log')
plt.sca(ax)
edit_graph('$n$ (cycle no.)', '$e$', text_scale=3)
plt.savefig(f'graphs/energy_convergence_vs_cycle_Ns_{Ns}_J_{J}_h_{h}_V_{V}_Nt_{Nt}_postselectevenbath_{drop_bath_parity_not_0}.pdf',transparent=True)

ax_inset.set_yscale('log')
ax_inset.set_xticks(T_list)
# ax_inset.set_yticks([1e-3, 1e-2, 1e-1])

plt.sca(ax_inset)
edit_graph('$T$', '$e_\mathrm{steady}$', text_scale=1.5)
plt.savefig(f'graphs/energy_convergence_vs_cycle_Ns_{Ns}_J_{J}_h_{h}_V_{V}_Nt_{Nt}_postselectevenbath_{drop_bath_parity_not_0}_inset.pdf', transparent=True)
plt.show()
