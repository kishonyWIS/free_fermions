import numpy as np
from scipy.linalg import eig
from matplotlib import pyplot as plt
from floquet_honeycomb_evolution import get_unitary_evolution, diagonalize_unitary_at_k_theta_time, get_topological_invariant
from interpolation import interpolate_hyperplane, find_nearest_index

# define the pauli matrices
SIGMA_X = np.array([[0,1],[1,0]])
SIGMA_Y = np.array([[0,-1j],[1j,0]])
SIGMA_Z = np.array([[1,0],[0,-1]])


nsteps = 2
nsteps = 301

kx_list = np.linspace(0, np.pi, 101)
ky = 0.
times = np.linspace(0, 1, nsteps)
theta_list = np.linspace(0, 2*np.pi, 201)
lamb = 0.0

u = np.zeros((len(kx_list), len(theta_list), len(times), 2, 2), dtype=np.complex128)
phases = np.zeros((len(kx_list), len(theta_list), len(times), 2), dtype=np.complex128)
angles = np.zeros((len(kx_list), len(theta_list), len(times), 2))
states = np.zeros((len(kx_list), len(theta_list), len(times), 2, 2), dtype=np.complex128)

# anchors define the hyperplane
anchors = [np.array([0,0,0.5]),
           np.array([0,2*np.pi,0.5]),
           np.array([np.pi,0,5./6.]),
           np.array([np.pi,2*np.pi,5./6.]),
           np.array([np.pi,2*np.pi*2/3.,0.6])]

# times_matrix = np.zeros((len(kx_list), len(theta_list)))
# for i_kx, kx in enumerate(kx_list):
#     for i_theta, theta in enumerate(theta_list):
#         # get the index of the time slice at kx, theta by interpolating points in 3d space
#         point = np.array([kx, theta])
#         times_matrix[i_kx, i_theta] = interpolate_hyperplane(anchors, point)[2]

# iterate over all the values of ky, theta
# and calculate the unitary for each
# for i_kx, kx in enumerate(kx_list):
#     print(i_kx)
#     for i_theta, theta in enumerate(theta_list):
#         # get the index of the time slice at kx, theta by interpolating points in 3d space
#
#         i_t = nsteps // 2 + i_kx
#         time = times[i_t]
#         # for i_t, time in enumerate(times):
#         angles[i_kx, i_theta, i_t, :], states[i_kx, i_theta, i_t, :, :] = diagonalize_unitary_at_k_theta_time(kx, ky, theta, time)


for i_kx, kx in enumerate(kx_list):
    print(i_kx)
    for i_theta, theta in enumerate(theta_list):
        point = np.array([kx, theta])
        time = interpolate_hyperplane(anchors, point)[2]
        i_t = 0
        angles[i_kx, i_theta, i_t, :], states[i_kx, i_theta, i_t, :, :] = diagonalize_unitary_at_k_theta_time(kx, ky,
                                                                                                              theta,
                                                                                                              time)


top_band_phases = np.abs(angles.max(axis=-1))
topological_singularities_pi = top_band_phases > 3.1415
topological_singularities_0 = top_band_phases < 0.0001
topological_singularities_0[:,:,0] = False
top_band_states = states[:, :, :, :, 0]

# plt.plot(top_band_phases.reshape(len(kx_list)*len(theta_list), len(times)).T)


top_band_states_on_plane = np.zeros((len(kx_list), len(theta_list), 2), dtype=np.complex128)
top_band_phases_on_plane = np.zeros((len(kx_list), len(theta_list)), dtype=np.complex128)
for i_kx in range(len(kx_list)):
    for i_theta in range(len(theta_list)):
        top_band_states_on_plane[i_kx, i_theta, :] = top_band_states[i_kx, i_theta, nsteps//2 + i_kx, :]
        top_band_phases_on_plane[i_kx, i_theta] = top_band_phases[i_kx, i_theta, nsteps//2 + i_kx]

# calculate the overlap of states in consecutive parameter points
# for i_kx in range(top_band_states_on_plane.shape[0]):
#     for i_theta in range(top_band_states_on_plane.shape[1]):
#         current_state = top_band_states_on_plane[i_kx, i_theta, :]
#         prev_state = top_band_states_on_plane[i_kx-1, i_theta, :]
#         overlap = current_state.T.conj() @ prev_state
#         if np.abs(np.abs(overlap)-1) > 0.01:
#             print(i_kx, i_theta, overlap)

print()