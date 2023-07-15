import numpy as np
from scipy.linalg import eig
from matplotlib import pyplot as plt
from floquet_honeycomb_evolution import get_unitary_evolution, diagonalize_unitary_at_k_theta_time, get_topological_invariant
from interpolation import interpolate_hyperplane
from mpl_toolkits.mplot3d import Axes3D

def plot_singularities_3d(ky, res_grid, res_energy=0.1, ax=None):

    kx_list = np.linspace(0, np.pi, res_grid)
    theta_list = np.linspace(0, 2 * np.pi, res_grid)
    times = np.linspace(0, 1, res_grid)

    u = np.zeros((len(kx_list), len(theta_list), len(times), 2, 2), dtype=np.complex128)
    angles = np.zeros((len(kx_list), len(theta_list), len(times), 2))
    states = np.zeros((len(kx_list), len(theta_list), len(times), 2, 2), dtype=np.complex128)


    for i_kx, kx in enumerate(kx_list):
        print(i_kx)
        for i_theta, theta in enumerate(theta_list):
            for i_t, time in enumerate(times):
                angles[i_kx, i_theta, i_t], states[i_kx, i_theta, i_t, :] = diagonalize_unitary_at_k_theta_time(kx, ky, theta, time)

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
    ax.scatter(kx_list[pos[0]], theta_list[pos[1]], times[pos[2]], c='r')

    pos = np.where(topological_singularities_pi)

    ax.scatter(kx_list[pos[0]], theta_list[pos[1]], times[pos[2]], c='b')

    ax.set_xlabel('kx')
    ax.set_ylabel('theta')
    ax.set_zlabel('time')

    plt.show()
