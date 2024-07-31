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

ky_list = np.linspace(0, np.pi, 101)
kx = 0.#np.pi#0.#
pulse_length = 1/3
theta_list = np.linspace(0, 2*np.pi, 101)
KY, THETA = np.meshgrid(ky_list, theta_list, indexing='ij')

u = np.zeros((len(ky_list), len(theta_list), 2, 2), dtype=np.complex128)
angles = np.zeros((len(ky_list), len(theta_list), 2))
states = np.zeros((len(ky_list), len(theta_list), 2, 2), dtype=np.complex128)



if kx == 0.:
    anchors = [np.array([0,0,0.5]),
               np.array([0,2*np.pi,0.5]),
               np.array([np.pi, 0, 0.5]),
               np.array([np.pi, 2 * np.pi, 0.5]),
               np.array([np.pi,2*np.pi/3,5/6]),
               np.array([np.pi,4*np.pi/3,5/6])]

    triangulation_mode = 'manual'  # 'manual' #'auto'
    anchor_pairs_to_draw_lines = [[0,1], [0,2], [1,3], [0,4], [2,4], [1,5], [3,5], [4,5]]

    # before changing kx and ky:
    # anchors = [np.array([0,0,0.5]),
    #            np.array([0,2*np.pi,0.5]),
    #            np.array([np.pi,0,5/6]),
    #            np.array([np.pi,2*np.pi*2/3,5/6]),
    #            np.array([np.pi,2*np.pi,5/6]),
    #            np.array([np.pi,2*np.pi/3.,0.5])]

# between the zero and pi singularities for ky=np.pi

if kx == np.pi:
    anchors = [np.array([0,0,5/6]),
               np.array([0,2*np.pi/3+0.001,0.5]),
               np.array([0,4*np.pi/3,5/6]),
               np.array([0,2*np.pi,5/6]),
               np.array([np.pi, 0, 5/6]),
               np.array([np.pi, 2*np.pi/3, 5/6]),
               np.array([np.pi, 4*np.pi/3-0.001, 0.5]),
               np.array([np.pi, 2*np.pi, 5/6])
               ]

    triangulation_mode = 'manual'  # 'manual' #'auto'
    anchor_pairs_to_draw_lines = [[0,1], [1,2], [2,3], [4,5], [5,6], [6,7], [0,4], [3,7], [1,6], [0,5], [2,7]]

    # before changing kx and ky:
    # anchors = [np.array([0,2*np.pi,0.5]),
    #            np.array([0,0,0.5]),
    #            np.array([np.pi,2*np.pi*2/3,0.5]),
    #            np.array([0,2*np.pi*2/3,5/6]),
    #            np.array([0,2*np.pi/3,5/6]),
    #            np.array([np.pi,2*np.pi,5/6]),
    #            np.array([np.pi,2*np.pi/3,5/6]),
    #            np.array([np.pi,0,5/6])]

interp = LinearNDInterpolator(np.array(anchors)[:,:-1], np.array(anchors)[:,-1])
TIME = interp(KY, THETA)

for i_ky, ky in enumerate(ky_list):
    print(i_ky)
    for i_theta, theta in enumerate(theta_list):
        point = np.array([ky, theta])
        time = TIME[i_ky, i_theta]
        # ang, stat = diagonalize_unitary_at_k_theta_time(0.4, 0.7, 0, 1, pulse_length=pulse_length)
        angles[i_ky, i_theta, :], states[i_ky, i_theta, :, :] = diagonalize_unitary_at_k_theta_time(kx, ky, theta, time, pulse_length = pulse_length)


top_band_phases = np.abs(angles.max(axis=-1))
top_band_states = states[:, :, :, 0]

plt.pcolor(KY,THETA,top_band_phases)
plt.xlabel('ky')
plt.ylabel('theta')
plt.colorbar()



# plot the singularities

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.invert_zaxis()
elev = 17.
azim = 110
ax.view_init(elev=elev, azim=azim, vertical_axis='y')  # for ky=0, elev=10., azim=108, ky=pi, elev=12., azim=105
ax.set_xticks([0, np.pi])
ax.set_xticklabels([0, '$\pi/a$'])
ax.set_yticks([0, 2 * np.pi / 3, 2 * np.pi * 2 / 3, 2 * np.pi])
ax.set_yticklabels(['$0$', '$\\frac{1}{3}$', '$\\frac{2}{3}$', '$1$'])
ax.set_zticks([0, 1 / 3, 2 / 3, 1])
ax.set_zticklabels([0, '$T/3$', '$2T/3$', '$T$'])
ax.set_zlim(1, 0)
ax.set_xlim(0, np.pi)
ax.set_ylim(0, 2 * np.pi)
edit_graph('$k_y$', '$\\lambda$', '$t$', ax)

mode = 'manual' #'manual' #'both' #'numerical'

point_singularities_0, point_singularities_pi, line_singularities_0, line_singularities_pi =\
    plot_singularities_3d(kx, 22, 0.05, ax, pulse_length=pulse_length, plot=False,
                          plot_numerical_singularities=(mode == 'numerical' or mode == 'both'))


if mode == 'manual' or mode == 'both':
    surf = ax.plot_surface(KY, THETA, TIME, cmap=matplotlib.cm.plasma.reversed(), alpha=0.7, linewidth=0,
                           antialiased=True, zorder=3)

    anchors = np.array(anchors)
    tri = Delaunay(anchors[:, :-1])
    # Plot the edges of the triangles
    if triangulation_mode == 'auto':
        for simplex in tri.simplices:
            ax.plot_trisurf(anchors[simplex, 0], anchors[simplex, 1], anchors[simplex, 2], color='none', edgecolor='k')
    if triangulation_mode == 'manual':
        for pair in anchor_pairs_to_draw_lines:
            ax.plot([anchors[pair[0], 0], anchors[pair[1], 0]],
                    [anchors[pair[0], 1], anchors[pair[1], 1]],
                    [anchors[pair[0], 2], anchors[pair[1], 2]], c='k', linewidth=2, zorder=2)

    zorder_0 = 1 if kx == np.pi else 30
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


    plt.savefig(f'graphs/time_vortex/ky_theta_t_space_kx_{kx:.2f}.pdf')






    # Create 3D surface plot
    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=KY,
        y=THETA,
        z=TIME,
        colorscale='Plasma',
        reversescale=True,
        opacity=0.7,
        showscale=False
    ))

    # Plot lines for the edges of the triangles
    if triangulation_mode == 'auto':
        for simplex in tri.simplices:
            fig.add_trace(go.Scatter3d(
                x=anchors[simplex, 0],
                y=anchors[simplex, 1],
                z=anchors[simplex, 2],
                mode='lines',
                line=dict(color='black', width=3)
            ))
    if triangulation_mode == 'manual':
        for pair in anchor_pairs_to_draw_lines:
            fig.add_trace(go.Scatter3d(
                x=[anchors[pair[0], 0], anchors[pair[1], 0]],
                y=[anchors[pair[0], 1], anchors[pair[1], 1]],
                z=[anchors[pair[0], 2], anchors[pair[1], 2]],
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
        xaxis_title=dict(text=u'kᵧ', font=dict(size=50)),
        yaxis_title=dict(text=r'λ', font=dict(size=50)),
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
            ticktext=['0', '1/3', '2/3', '1'],
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

    fig.write_html(f'graphs/time_vortex/ky_theta_t_space_kx_{kx:.2f}.html')
    fig.show()




    get_topological_invariant(top_band_states)
plt.show()

print()