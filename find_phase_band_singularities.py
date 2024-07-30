import numpy as np
from scipy.linalg import eig
from matplotlib import pyplot as plt
from floquet_honeycomb_evolution import get_unitary_evolution, diagonalize_unitary_at_k_theta_time, get_topological_invariant
from interpolation import interpolate_hyperplane
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.interpolate import LinearNDInterpolator


def plot_singularities_3d(kx, res_grid, res_energy=0.1, ax=None, pulse_length = 1/3, plot=False, plot_numerical_singularities=False):
    ky_list = np.linspace(0, np.pi, 101)
    theta_list = np.linspace(0, 2 * np.pi, 101)
    KY, THETA = np.meshgrid(ky_list, theta_list, indexing='ij')

    if kx == 0.:
        point_singularities_0 = [[np.pi, 0, 2/3]]
        point_singularities_pi = []
        line_singularities_0 = [[[np.pi, np.pi], [2*np.pi/3, 4*np.pi/3], [0, 2 / 3]],
                                [[np.pi, np.pi], [4*np.pi/3, 2*np.pi], [0, 2 / 3]],
                                [[np.pi, np.pi], [4*np.pi/3, 2*np.pi], [2/3, 2 / 3]]]
        line_singularities_pi = [[[0, np.pi], [2 / 3 * np.pi, 2 / 3 * np.pi], [2 / 3, 2 / 3]],
                                 [[0, 0],[0, 2*np.pi],[2/3, 2/3]]]
        plane_singularities_0 = []
        plane_singularities_pi = []
    elif kx == np.pi:
        point_singularities_0 = []
        point_singularities_pi = [[0,0,2/3], [0,2*np.pi,2/3], [np.pi,4/3*np.pi,2/3]]
        line_singularities_0 = [[[np.pi, np.pi], [0, 2 * np.pi / 3], [2/3, 2 / 3]],
                                [[0, 0], [2 * np.pi / 3, 4 * np.pi / 3], [2/3, 2 / 3]],
                                [[0, 0], [2 * np.pi / 3, 4 * np.pi / 3], [0, 2 / 3]],
                                [[np.pi, np.pi], [4 * np.pi / 3, 2 * np.pi], [0, 2 / 3]]]
        for ky in np.linspace(0,np.pi,100):
            line_singularities_0.append([[ky,ky],[0,2/3*np.pi],[0,2/3]])
        line_singularities_pi = []
        plane_singularities_0 = [[[0,0,0],
                                 [np.pi,0,0],
                                 [0,2*np.pi/3,2/3],
                                 [np.pi,2*np.pi/3,2/3]]]
        plane_singularities_pi = []


    ky_list = np.linspace(0, np.pi, res_grid//2)
    theta_list = np.linspace(0, 2 * np.pi, res_grid)
    times = np.linspace(0, 1, res_grid)

    u = np.zeros((len(ky_list), len(theta_list), len(times), 2, 2), dtype=np.complex128)
    angles = np.zeros((len(ky_list), len(theta_list), len(times), 2))
    states = np.zeros((len(ky_list), len(theta_list), len(times), 2, 2), dtype=np.complex128)


    for i_ky, ky in enumerate(ky_list):
        print(i_ky)
        for i_theta, theta in enumerate(theta_list):
            for i_t, time in enumerate(times):
                angles[i_ky, i_theta, i_t], states[i_ky, i_theta, i_t, :] = diagonalize_unitary_at_k_theta_time(kx, ky, theta, time, pulse_length=pulse_length)

    top_band_phases = np.abs(angles.max(axis=-1))
    topological_singularities_pi = top_band_phases > np.pi - res_energy
    topological_singularities_0 = top_band_phases < res_energy
    topological_singularities_0[:,:,0] = False

    # plot in 3D space of kx, theta, time where singularities are

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    size = 21
    pos = np.where(topological_singularities_0)
    zorder_0 = 1 if kx == np.pi else 30
    if plot_numerical_singularities:
        s = ax.scatter(ky_list[pos[0]], theta_list[pos[1]], times[pos[2]], c='r', s=20, alpha=1, zorder=zorder_0)
    if plot:
        for point in point_singularities_0:
            # ax.scatter(point[0], point[1], point[2], c='r', s=size, alpha=1, zorder=zorder_0)
            ax.plot([point[0]]*2, [point[1]]*2, [point[2],point[2]-0.002], c='r', alpha=1, linewidth=5,zorder=zorder_0)
        for line in line_singularities_0:
            ax.plot(line[0], line[1], line[2], c='r', alpha=1, linewidth=5, zorder=zorder_0)
    # for plane in plane_singularities_0:
    #     interp = LinearNDInterpolator(np.array(plane)[:, :-1], np.array(plane)[:, -1])
    #     TIME = interp(KX, THETA)
    #     surf = ax.plot_surface(KX, THETA, TIME, color='r', alpha=1, linewidth=0,
    #                            antialiased=True, zorder=1)# , cmap=matplotlib.cm.get_cmap("Reds")

    pos = np.where(topological_singularities_pi)

    if plot_numerical_singularities:
        s = ax.scatter(ky_list[pos[0]], theta_list[pos[1]], times[pos[2]], c='b', s=20, alpha=1, zorder=30)
    # ax.plot([0, np.pi], [2 / 3 * np.pi, 2 / 3 * np.pi], [2 / 3, 2 / 3], c='b', alpha=1, linewidth=5,zorder=30)
    if plot:
        for point in point_singularities_pi:
            # ax.scatter(point[0], point[1], point[2], c='b', s=size, alpha=1, zorder=30)
            ax.plot([point[0]]*2, [point[1]]*2, [point[2],point[2]-0.002], c='b', alpha=1, linewidth=5,zorder=30)
        for line in line_singularities_pi:
            ax.plot(line[0], line[1], line[2], c='b', alpha=1, linewidth=5,zorder=30)
    # for plane in plane_singularities_pi:
    #     interp = LinearNDInterpolator(np.array(plane)[:, :-1], np.array(plane)[:, -1])
    #     TIME = interp(KX, THETA)
    #     surf = ax.plot_surface(KX, THETA, TIME, color='b', alpha=1, linewidth=0,
    #                            antialiased=True, zorder=30) #, cmap=matplotlib.cm.get_cmap("Blues")
    if plot:
        plt.show()

    return point_singularities_0, point_singularities_pi, line_singularities_0, line_singularities_pi