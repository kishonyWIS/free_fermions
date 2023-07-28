import numpy as np
from matplotlib import pyplot as plt
from floquet_honeycomb_evolution import diagonalize_unitary_at_k_theta_time, get_topological_invariant
from interpolation import interpolate_hyperplane
from find_phase_band_singularities import plot_singularities_3d
from scipy.interpolate import LinearNDInterpolator
import matplotlib
import seaborn as sns


def edit_graph(xlabel, ylabel, zlabel, ax):
    sns.set_style("whitegrid")
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix",
          "figure.autolayout": True}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    ax.set_xlabel(xlabel, fontsize='20', fontname='Times New Roman')
    ax.set_ylabel(ylabel, fontsize='20', fontname='Times New Roman')
    ax.set_zlabel(zlabel, fontsize='20', fontname='Times New Roman')
    plt.tick_params(axis='x', which='major', labelsize=15)
    plt.tick_params(axis='y', which='major', labelsize=15)
    plt.tick_params(axis='z', which='major', labelsize=15)
    plt.tight_layout()


kx_list = np.linspace(0, np.pi, 101)
ky = 0
pulse_length = 1/3
theta_list = np.linspace(0, 2*np.pi, 101)
KX, THETA = np.meshgrid(kx_list, theta_list, indexing='ij')

u = np.zeros((len(kx_list), len(theta_list), 2, 2), dtype=np.complex128)
angles = np.zeros((len(kx_list), len(theta_list), 2))
states = np.zeros((len(kx_list), len(theta_list), 2, 2), dtype=np.complex128)

# anchors define the hyperplane
# start time
# anchors = [np.array([0,0,0]),
#            np.array([0,2*np.pi,0]),
#            np.array([np.pi,0,0]),
#            np.array([np.pi,2*np.pi,0])]

# end time
# anchors = [np.array([0,0,1]),
#            np.array([0,2*np.pi,1]),
#            np.array([np.pi,0,1]),
#            np.array([np.pi,2*np.pi,1])]

# between the zero and pi singularities for ky=0

anchors = [np.array([0,0,0.5]),
           np.array([0,2*np.pi,0.5]),
           np.array([np.pi,0,5/6]),
           np.array([np.pi,2*np.pi*2/3,5/6]),
           np.array([np.pi,2*np.pi,5/6]),
           np.array([np.pi,2*np.pi/3.,0.5])]

# between the zero and pi singularities for ky=np.pi

# anchors = [np.array([0,2*np.pi,0.5]),
#            np.array([0,0,0.5]),
#            np.array([np.pi,2*np.pi*2/3,0.5]),
#            np.array([0,2*np.pi*2/3,5/6]),
#            np.array([0,2*np.pi/3,5/6]),
#            np.array([np.pi,2*np.pi,5/6]),
#            np.array([np.pi,2*np.pi/3,5/6]),
#            np.array([np.pi,0,5/6])]

interp = LinearNDInterpolator(np.array(anchors)[:,:-1], np.array(anchors)[:,-1])
TIME = interp(KX, THETA)

for i_kx, kx in enumerate(kx_list):
    print(i_kx)
    for i_theta, theta in enumerate(theta_list):
        point = np.array([kx, theta])
        time = TIME[i_kx, i_theta]
        # ang, stat = diagonalize_unitary_at_k_theta_time(0.4, 0.7, 0, 1, pulse_length=pulse_length)
        angles[i_kx, i_theta, :], states[i_kx, i_theta, :, :] = diagonalize_unitary_at_k_theta_time(kx, ky, theta, time, pulse_length = pulse_length)


top_band_phases = np.abs(angles.max(axis=-1))
top_band_states = states[:, :, :, 0]

plt.pcolor(KX,THETA,top_band_phases)
plt.xlabel('kx')
plt.ylabel('theta')
plt.colorbar()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(KX, THETA, TIME, cmap=matplotlib.cm.plasma, alpha=0.5, linewidth=0, antialiased=False)

# plot the singularities
plot_singularities_3d(ky, 22, 0.05, ax, pulse_length=pulse_length)

ax.invert_zaxis()
ax.view_init(elev=10., azim=108, vertical_axis='y') # for ky=0
ax.set_xticks([0, np.pi])
ax.set_xticklabels([0, '$\pi$'])
ax.set_yticks([0, 2*np.pi/3, 2*np.pi*2/3, 2*np.pi])
ax.set_yticklabels([0, '$2\pi/3$', '$4\pi/3$', '$2\pi$'])
ax.set_zticks([0, 1/3, 2/3, 1])
ax.set_zticklabels([0, '$T/3$', '$2T/3$', '$T$'])
edit_graph('$k_x$', '$\\theta$', '$t$', ax)
plt.savefig(f'graphs/time_vortex/kx_theta_t_space_ky_{ky:.2f}.pdf')

get_topological_invariant(top_band_states)
plt.show()

print()