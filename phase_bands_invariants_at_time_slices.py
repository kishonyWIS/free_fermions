import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay

from floquet_honeycomb_evolution import diagonalize_unitary_at_k_theta_time, get_topological_invariant
from find_phase_band_singularities import plot_singularities_3d
from scipy.interpolate import LinearNDInterpolator
import matplotlib
import seaborn as sns
import plotly.graph_objects as go


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



scale = 1.0
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 10 * scale,  # Base font size
    'axes.labelsize': 10 * scale,
    'axes.titlesize': 10 * scale,
    'xtick.labelsize': 8 * scale,
    'ytick.labelsize': 8 * scale,
    'legend.fontsize': 8 * scale,
    'figure.titlesize': 12 * scale,
    'text.latex.preamble': r'''
        \usepackage{subfigure}
        \usepackage{amsmath, amssymb, amsfonts}
        \usepackage{xcolor}
        \definecolor{darkblue}{RGB}{0,0,150}
        \definecolor{nightblue}{RGB}{0,0,100}
        \usepackage{graphicx,mathtools,bm,bbm}
        \usepackage{MnSymbol}
        \usepackage[colorlinks,citecolor=darkblue,linkcolor=darkblue,urlcolor=nightblue]{hyperref}
        \usepackage[english]{babel}
        \usepackage[babel,kerning=true,spacing=true]{microtype}
        \usepackage[utf8]{inputenc}
        \usepackage{soul}
    '''
})

kx_list = np.linspace(0, np.pi, 101)
ky = np.pi#np.pi#0.#
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

if ky == 0.:
    anchors = [np.array([0,0,0.5]),
               np.array([0,2*np.pi,0.5]),
               np.array([np.pi,0,5/6]),
               np.array([np.pi,2*np.pi*2/3,5/6]),
               np.array([np.pi,2*np.pi,5/6]),
               np.array([np.pi,2*np.pi/3.,0.5])]

# between the zero and pi singularities for ky=np.pi

if ky == np.pi:
    anchors = [np.array([0,2*np.pi,0.5]),
               np.array([0,0,0.5]),
               np.array([np.pi,2*np.pi*2/3,0.5]),
               np.array([0,2*np.pi*2/3,5/6]),
               np.array([0,2*np.pi/3,5/6]),
               np.array([np.pi,2*np.pi,5/6]),
               np.array([np.pi,2*np.pi/3,5/6]),
               np.array([np.pi,0,5/6])]

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
surf = ax.plot_surface(KX, THETA, TIME, cmap=matplotlib.cm.plasma.reversed(), alpha=0.7, linewidth=0, antialiased=True, zorder=3)

anchors = np.array(anchors)
tri = Delaunay(anchors[:, :-1])
# Plot the edges of the triangles
for simplex in tri.simplices:
    ax.plot_trisurf(anchors[simplex, 0], anchors[simplex, 1], anchors[simplex, 2], color='none', edgecolor='k')

# ax.plot_trisurf(np.array(anchors)[:, 0], np.array(anchors)[:, 1], np.array(anchors)[:, 2], color='none', edgecolor='k')


# plot the singularities
point_singularities_0, point_singularities_pi, line_singularities_0, line_singularities_pi =\
    plot_singularities_3d(ky, 22, 0.05, ax, pulse_length=pulse_length, plot=False)

zorder_0 = 1 if ky == np.pi else 30
for point in point_singularities_0:
    # ax.scatter(point[0], point[1], point[2], c='r', s=size, alpha=1, zorder=zorder_0)
    ax.plot([point[0]] * 2, [point[1]] * 2, [point[2], point[2] - 0.002], c='r', alpha=1, linewidth=5, zorder=zorder_0)
for line in line_singularities_0:
    ax.plot(line[0], line[1], line[2], c='r', alpha=1, linewidth=5, zorder=zorder_0)
for point in point_singularities_pi:
    # ax.scatter(point[0], point[1], point[2], c='b', s=size, alpha=1, zorder=30)
    ax.plot([point[0]] * 2, [point[1]] * 2, [point[2], point[2] - 0.002], c='b', alpha=1, linewidth=5, zorder=30)
for line in line_singularities_pi:
    ax.plot(line[0], line[1], line[2], c='b', alpha=1, linewidth=5, zorder=30)

ax.invert_zaxis()
elev = 12.
azim = 105
ax.view_init(elev=elev, azim=azim, vertical_axis='y') # for ky=0, elev=10., azim=108, ky=pi, elev=12., azim=105
ax.set_xticks([0, np.pi])
ax.set_xticklabels([0, '$\pi/a$'])
ax.set_yticks([0, 2*np.pi/3, 2*np.pi*2/3, 2*np.pi])
ax.set_yticklabels([0, '$\\frac{2\pi}{3}$', '$\\frac{4\pi}{3}$', '$2\pi$'])
ax.set_zticks([0, 1/3, 2/3, 1])
ax.set_zticklabels([0, '$T/3$', '$2T/3$', '$T$'])
edit_graph('$k_x$', '$\\theta$', '$t$', ax)
plt.savefig(f'graphs/time_vortex/kx_theta_t_space_ky_{ky:.2f}.pdf')




# fig = go.Figure()
# fig.add_trace(go.Scatter(
#     x=[1, 2, 3, 4],
#     y=[1, 4, 9, 16],
#     name=r'$\alpha_{1c} = 352 \pm 11 \text{ km s}^{-1}$'
# ))
# fig.add_trace(go.Scatter(
#     x=[1, 2, 3, 4],
#     y=[0.5, 2, 4.5, 8],
#     name=r'$\beta_{1c} = 25 \pm 11 \text{ km s}^{-1}$'
# ))
# fig.update_layout(
#     xaxis_title=r'$\sqrt{(n_\text{c}(t|{T_\text{early}}))}$',
#     yaxis_title=r'$d, r \text{ (solar radius)}$'
# )
# fig.show()




# Create 3D surface plot
fig = go.Figure()

fig.add_trace(go.Surface(
    x=KX,
    y=THETA,
    z=TIME,
    colorscale='Plasma',
    reversescale=True,
    opacity=0.7,
    showscale=False
))

# Plot lines for the edges of the triangles
for simplex in tri.simplices:
    fig.add_trace(go.Scatter3d(
        x=anchors[simplex, 0],
        y=anchors[simplex, 1],
        z=anchors[simplex, 2],
        mode='lines',
        line=dict(color='black', width=3)
    ))

width = 20

# Add the vertical line for the point
for point in point_singularities_0:
    fig.add_trace(go.Scatter3d(
        x=[point[0], point[0]],
        y=[point[1], point[1]],
        z=[point[2], point[2] - 0.02],
        mode='lines',
        line=dict(color='red', width=width)
    ))
for point in point_singularities_pi:
    fig.add_trace(go.Scatter3d(
        x=[point[0], point[0]],
        y=[point[1], point[1]],
        z=[point[2], point[2] - 0.02],
        mode='lines',
        line=dict(color='blue', width=width)
    ))
for line in line_singularities_0:
    fig.add_trace(go.Scatter3d(
        x=line[0],
        y=line[1],
        z=line[2],
        mode='lines',
        line=dict(color='red', width=width)
    ))
for line in line_singularities_pi:
    fig.add_trace(go.Scatter3d(
        x=line[0],
        y=line[1],
        z=line[2],
        mode='lines',
        line=dict(color='blue', width=width)
    ))

fig.update_layout(scene=dict(
    xaxis_title=dict(text=u'k\u2093', font=dict(size=50)),
    yaxis_title=dict(text=r'θ', font=dict(size=50)),
    zaxis_title=dict(text=r't', font=dict(size=50)),
    camera=dict(
        eye=dict(x=3*np.cos(np.deg2rad(elev))*np.sin(np.deg2rad(azim)),
                 y=3*np.cos(np.deg2rad(elev))*np.cos(np.deg2rad(azim)),
                 z=3*np.sin(np.deg2rad(elev))),  # Adjust based on desired azimuth and elevation
        # eye=dict(x=1.2, y=-0.6, z=0.6),  # Adjust based on desired azimuth and elevation
        up=dict(x=0, y=-1, z=0)  # Adjust to control vertical orientation
    ),
    xaxis=dict(
        tickvals=[0, np.pi],
        ticktext=['0', 'π/a'],
        tickfont=dict(size=20)
    ),
    yaxis=dict(
        tickvals=[0, 2*np.pi/3, 4*np.pi/3, 2*np.pi],
        ticktext=['0', '2π/3', '4π/3', '2π'],
        tickfont=dict(size=20)
    ),
    zaxis=dict(
        tickvals=[0, 1/3, 2/3, 1],
        ticktext=['0', 'T/3', '2T/3', 'T'],
        range=[0, 1],
        tickfont=dict(size=20)
    )
))
fig.update_scenes(yaxis_autorange="reversed")

fig.write_html(f'graphs/time_vortex/kx_theta_t_space_ky_{ky:.2f}.html')
fig.show()




get_topological_invariant(top_band_states)
plt.show()

print()