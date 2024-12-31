import numpy as np
from matplotlib import pyplot as plt
from floquet_honeycomb_evolution import diagonalize_unitary_at_k_theta_time
import matplotlib
from tqdm import tqdm
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter, sobel
from plot_utils import edit_graph

matplotlib.use('MacOSX')  # Or 'Qt5Agg' if you have PyQt5 installed

res_grid = 120
T = 1
theta = 0




# kx = 4.084070449666731
# ky = 0.9424777960769379
# pulse_length = 0.4
# pulse_integral_factor = 0.2
# J_factor = 0.16666666666666666
# diagonalize_unitary_at_k_theta_time(kx, ky, theta, T, pulse_length=pulse_length, J_factor=J_factor)

load=True
save=False

kx_list = np.linspace(0, 2*np.pi, 2*res_grid+1)
ky_list = np.linspace(0, np.pi, res_grid+1)
# kx_list = [0, np.pi]
# ky_list = [0, np.pi]
pulse_length_list = np.linspace(0, 1, res_grid+1)[1:-1]
pulse_integral_factor_list = np.linspace(0, 2, res_grid+1)[1:-1]

PULSE_LENGTH, PULSE_INTEGRAL_FACTOR = np.meshgrid(pulse_length_list, pulse_integral_factor_list, indexing='ij')

topological_singularities_0 = np.zeros_like(PULSE_LENGTH)
topological_singularities_pi = np.zeros_like(PULSE_LENGTH)
topological_singularities_both = np.zeros_like(PULSE_LENGTH)
topological_singularities_0[:] = np.nan
topological_singularities_pi[:] = np.nan
topological_singularities_both[:] = np.nan

if load:
    topological_singularities_0 = np.load('topological_singularities_0.npy')
    topological_singularities_pi = np.load('topological_singularities_pi.npy')

else:
    for i_pulse_length, pulse_length in tqdm(enumerate(pulse_length_list)):
        for i_pulse_integral_factor, pulse_integral_factor in enumerate(pulse_integral_factor_list):
            J_factor = pulse_integral_factor/(pulse_length/(1/3))
            angles = np.zeros((len(kx_list), len(ky_list), 2))
            for i_kx, kx in enumerate(kx_list):
                for i_ky, ky in enumerate(ky_list):
                    angles[i_kx, i_ky, :], _ = (
                        diagonalize_unitary_at_k_theta_time(kx, ky, theta, T, pulse_length=pulse_length, J_factor=J_factor))
            top_band_phases = np.abs(angles.max(axis=-1))
            topological_singularities_0[i_pulse_length, i_pulse_integral_factor] = np.min(top_band_phases) + 10**-10
            topological_singularities_pi[i_pulse_length, i_pulse_integral_factor] = np.min(np.pi - top_band_phases) + 10**-10

# save the data
if save:
    np.save('topological_singularities_0.npy', topological_singularities_0)
    np.save('topological_singularities_pi.npy', topological_singularities_pi)

topological_singularities_both = np.minimum(topological_singularities_0, topological_singularities_pi)


def detect_saddle_points(img):
    epsilon = 0.05
    # gaussian filter with width of 3 pixels
    # img = gaussian_filter(img, sigma=1)
    # check if each point is higher than its two neighbors along x
    saddle_mask_x = np.logical_and(img > np.roll(img, 1, axis=0) + epsilon, img > np.roll(img, -1, axis=0) + epsilon)
    # remove the borders
    saddle_mask_x[0, :] = False
    saddle_mask_x[-1, :] = False
    # check if each point is higher than its two neighbors along y
    saddle_mask_y = np.logical_and(img > np.roll(img, 1, axis=1) + epsilon, img > np.roll(img, -1, axis=1) + epsilon)
    # remove the borders
    saddle_mask_y[:, 0] = False
    saddle_mask_y[:, -1] = False

    return np.logical_or(saddle_mask_x, saddle_mask_y)

# draw some points:
J_factor_list = [1.9,1.85,1.5,1.5,0.5,1.001]
pulse_length_list = [0.63,0.42,0.75,0.2,0.2,0.2]

x_ticks = [0, 1/3, 2/3, 1]
y_ticks = [0, np.pi/4, np.pi/2]
x_tick_labels = ['0', '1/3', '2/3', '1']
y_tick_labels = ['0', '$\pi/4$', '$\pi/2$']
scale = 2

for name, quasienergy_singularities in zip(['0', 'pi', 'both'],[topological_singularities_0, topological_singularities_pi, topological_singularities_both]):
    fig, ax = plt.subplots()
    plt.pcolor(PULSE_LENGTH, PULSE_INTEGRAL_FACTOR*np.pi/4, np.log10(quasienergy_singularities))
    plt.colorbar()
    edit_graph('$\Delta t$', '$\mathcal{J}_0^a \Delta t$', ax=ax, scale=scale, xticks=x_ticks, yticks=y_ticks, xticklabels=x_tick_labels, yticklabels=y_tick_labels)

    threshold = 0.1  # Replace with your desired threshold
    smoothed_data = quasienergy_singularities#gaussian_filter(topological_singularities_pi, sigma=1)
    binary_mask = smoothed_data < threshold
    skeleton = skeletonize(binary_mask)
    skeleton_x, skeleton_y = np.where(skeleton)
    plt.scatter(PULSE_LENGTH[skeleton_x, skeleton_y], (PULSE_INTEGRAL_FACTOR*np.pi/4)[skeleton_x, skeleton_y], color='red', s=2, label="Skeleton")
    plt.xlim([0, 1])
    plt.ylim([0, np.pi/2])
    plt.savefig(f'graphs/time_vortex/phase_diagram_quasienergy_{name}_with_gap.pdf')

    # make another plot without the pcolor, only the skeleton
    fig, ax = plt.subplots()
    plt.scatter(PULSE_LENGTH[skeleton_x, skeleton_y], (PULSE_INTEGRAL_FACTOR*np.pi/4)[skeleton_x, skeleton_y], color='red', s=2, label="Skeleton")
    for J_factor, pulse_length in zip(J_factor_list, pulse_length_list):
        plt.scatter(pulse_length, J_factor*np.pi/4, color='red', s=10)
    # set axis limits
    plt.xlim([0, 1])
    plt.ylim([0, np.pi/2])
    edit_graph('$\Delta t$', '$\mathcal{J}_0^a \Delta t$', ax=ax, scale=scale, xticks=x_ticks, yticks=y_ticks, xticklabels=x_tick_labels, yticklabels=y_tick_labels)
    plt.savefig(f'graphs/time_vortex/phase_diagram_quasienergy_{name}.pdf')

    # Detect saddle points
    saddle_mask = detect_saddle_points(-np.log10(quasienergy_singularities))
    plt.pcolor(PULSE_LENGTH, PULSE_INTEGRAL_FACTOR*np.pi/4, np.log10(quasienergy_singularities))
    saddle_x, saddle_y = np.where(saddle_mask)
    for J_factor, pulse_length in zip(J_factor_list, pulse_length_list):
        plt.scatter(pulse_length, J_factor*np.pi/4, color='red', s=10)
    plt.scatter(PULSE_LENGTH[saddle_x, saddle_y], (PULSE_INTEGRAL_FACTOR*np.pi/4)[saddle_x, saddle_y], color='blue', s=2, label="Saddle points")

    # draw only the saddle mask as an image
    plt.figure()
    # make black and white
    plt.pcolor(PULSE_LENGTH, PULSE_INTEGRAL_FACTOR*np.pi/4, np.logical_not(saddle_mask), cmap='gray')
    for J_factor, pulse_length in zip(J_factor_list, pulse_length_list):
        plt.scatter(pulse_length, J_factor*np.pi/4, color='red', s=10)
    edit_graph('$\Delta t$', '$\mathcal{J}_0^a \Delta t$', scale=scale, xticks=x_ticks, yticks=y_ticks, xticklabels=x_tick_labels, yticklabels=y_tick_labels)
    plt.savefig(f'graphs/time_vortex/phase_diagram_quasienergy_{name}_saddle_points.pdf')

    # do the same with imshow
    plt.figure()
    plt.imshow(np.logical_not(saddle_mask.T), cmap='gray', origin='lower', aspect='auto', extent=[0, 1, 0, np.pi/2])
    for J_factor, pulse_length in zip(J_factor_list, pulse_length_list):
        plt.scatter(pulse_length, J_factor*np.pi/4, color='red', s=10)
    edit_graph('$\Delta t$', '$\mathcal{J}_0^a \Delta t$', scale=scale, xticks=x_ticks, yticks=y_ticks, xticklabels=x_tick_labels, yticklabels=y_tick_labels)
    plt.savefig(f'graphs/time_vortex/phase_diagram_quasienergy_{name}_saddle_points_imshow.pdf')


plt.show()