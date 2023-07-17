import numpy as np
from matplotlib import pyplot as plt
from floquet_honeycomb_evolution import diagonalize_unitary_at_k_theta_time, get_topological_invariant
from interpolation import interpolate_hyperplane
from find_phase_band_singularities import plot_singularities_3d
from scipy.interpolate import LinearNDInterpolator


kx_list = np.linspace(0, np.pi, 101)
ky = 0.
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
           np.array([np.pi,2*np.pi/3,5/6]),
           np.array([np.pi,2*np.pi,5/6]),
           np.array([np.pi,2*np.pi*2/3.,0.5])]

# between the zero and pi singularities for ky=np.pi
# anchors = [np.array([0,0,0.5]),
#            np.array([0,2*np.pi,0.5]),
#            np.array([np.pi,2*np.pi/3,0.5]),
#            np.array([0,2*np.pi/3,5/6]),
#            np.array([0,2*np.pi*2/3,5/6]),
#            np.array([np.pi,0,5/6]),
#            np.array([np.pi,2*np.pi*2/3,5/6]),
#            np.array([np.pi,2*np.pi,5/6])]

interp = LinearNDInterpolator(np.array(anchors)[:,:-1], np.array(anchors)[:,-1])
TIME = interp(KX, THETA)

for i_kx, kx in enumerate(kx_list):
    print(i_kx)
    for i_theta, theta in enumerate(theta_list):
        point = np.array([kx, theta])
        time = TIME[i_kx, i_theta]
        i_t = 0
        angles[i_kx, i_theta, :], states[i_kx, i_theta, :, :] = diagonalize_unitary_at_k_theta_time(kx, ky, theta, time)


top_band_phases = np.abs(angles.max(axis=-1))
topological_singularities_pi = top_band_phases > 3.1415
topological_singularities_0 = top_band_phases < 0.0001
# topological_singularities_0[:,:,0] = False
top_band_states = states[:, :, :, 0]

plt.pcolor(KX,THETA,top_band_phases)
plt.xlabel('kx')
plt.ylabel('theta')
plt.colorbar()

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(KX, THETA, TIME)

# plot the singularities
plot_singularities_3d(ky, 41, 0.1, ax)

ax.set_xlabel('kx')
ax.set_ylabel('theta')
ax.set_zlabel('time')

get_topological_invariant(top_band_states)
plt.show()

print()