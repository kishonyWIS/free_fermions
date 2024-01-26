import os

import matplotlib.pyplot as plt

from FloquetKSL import *
import numpy as np
from fft import calculate_fourier_transform

class LocalSpectralFunction():
    def __init__(self, hamiltonian: MajoranaFreeFermionHamiltonian, steps:int):
        self.hamiltonian = hamiltonian
        self.matrix_shape = np.prod(hamiltonian.system_shape[:len(hamiltonian.system_shape) // 2])
        self.hamiltonian.dt = 1 / steps
        self.steps = steps
        self.small_unitaries = self._get_small_unitaries()

    def _get_small_unitaries(self):
        small_unitaries = []
        for t in np.arange(0, int(1. / self.hamiltonian.dt)) * self.hamiltonian.dt:
            Ud = np.eye(self.matrix_shape)
            Ud = self.hamiltonian._unitary_trotterize_run_step(Ud, t)
            small_unitaries.append(Ud)
        return small_unitaries

    def get_unitary_vs_delta_t(self, step_start, cycles=1):
        unitary_vs_delta_t = []
        unitary = np.eye(self.matrix_shape)
        for i in range(step_start, step_start + cycles*self.steps):
            unitary = self.small_unitaries[i%self.steps] @ unitary
            unitary_vs_delta_t.append(np.diag(unitary).copy())
        return np.array(unitary_vs_delta_t)

    def get_time_averaged_unitary(self, cycles=1, start_time_steps=None, fourier=False):
        unitary_vs_delta_t_averaged = np.zeros((cycles*self.steps, self.matrix_shape))
        if start_time_steps is None:
            start_time_steps = self.steps
        for i in range(0, self.steps, start_time_steps):
            print(i)
            unitary_vs_delta_t = self.get_unitary_vs_delta_t(i, cycles=cycles)
            unitary_vs_delta_t_averaged += unitary_vs_delta_t
        unitary_vs_delta_t_averaged /= start_time_steps
        if fourier:
            angular_frequencies, unitary_vs_omega = calculate_fourier_transform(unitary_vs_delta_t_averaged, dt=self.hamiltonian.dt, axis=0)
            return angular_frequencies, unitary_vs_omega
        else:
            return unitary_vs_delta_t_averaged


if __name__ == '__main__':
    J = np.pi/4*0.9#np.pi / 4 * 1#
    pulse_length = 1/2#1/6#
    vortex_location = 'plaquette'#None#
    num_sites_x = 6
    num_sites_y = 6

    hamiltonian, location_dependent_delay, vortex_center = get_floquet_KSL_model(num_sites_x, num_sites_y, J=J,
                                                                                 pulse_length=pulse_length,
                                                                                 vortex_location=vortex_location)
    steps = 300
    start_time_steps = 30
    cycles = 100

    # load if already calculated
    # check if file exists
    filename = f'graphs/time_vortex/unitary_vs_omega_num_sites_x_{num_sites_x}_num_sites_y_{num_sites_y}_J_{J:.2f}_pulse_length_{pulse_length:.2f}.npz'
    if os.path.isfile(filename):
        data = np.load(filename)
        unitary_vs_omega = data['unitary_vs_omega']
        angular_frequencies = data['angular_frequencies']
    else:
        local_spectral_function = LocalSpectralFunction(hamiltonian, steps)
        angular_frequencies, unitary_vs_omega = local_spectral_function.get_time_averaged_unitary(cycles=cycles, start_time_steps=start_time_steps, fourier=True)
        unitary_vs_omega = unitary_vs_omega.reshape(steps*cycles, *hamiltonian.system_shape[:len(hamiltonian.system_shape)//2])
        # save the data including steps, cycles, start_time_steps, num_sites_x, num_sites_y, J, pulse_length
        np.savez(filename, unitary_vs_omega=unitary_vs_omega, angular_frequencies=angular_frequencies)


    # average over sites around the vortex
    sites_distance_from_vortex = np.zeros(hamiltonian.system_shape[:len(hamiltonian.system_shape)//2])
    for site in np.ndindex(hamiltonian.system_shape[:len(hamiltonian.system_shape)//2]):
        x, y = hexagonal_lattice_site_to_x_y(site)
        sites_distance_from_vortex[site] = np.sqrt((vortex_center[0]-x)**2 + (vortex_center[1]-y)**2)
    closest_distance = np.min(sites_distance_from_vortex)
    closest_sites_to_vortex = np.where(np.abs(sites_distance_from_vortex - closest_distance)<0.01)
    unitary_vs_omega_close_to_vortex = unitary_vs_omega[:, closest_sites_to_vortex[0], closest_sites_to_vortex[1], closest_sites_to_vortex[2]]
    unitary_vs_omega_close_to_vortex = np.mean(unitary_vs_omega_close_to_vortex, axis=1)
    plt.plot(angular_frequencies, np.abs(unitary_vs_omega_close_to_vortex)/np.max(np.abs(unitary_vs_omega_close_to_vortex)))
    plt.xlim(-np.pi*1.05, np.pi*1.05)
    plt.xticks(np.arange(-2, 3) * np.pi / 2, [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    edit_graph('Energy', 'Time-averaged density of states')
    plt.savefig(f'graphs/time_vortex/dos_Nx_{num_sites_x}_Ny_{num_sites_y}_J_{J:.2f}_pulse_length_{pulse_length:.2f}.pdf')
    plt.show()

    index_frequency_pi = np.argmin(np.abs(angular_frequencies - np.pi))
    # is this the right normalization?
    unitary_at_pi = unitary_vs_omega[index_frequency_pi, :, :, :].copy()
    draw_lattice(hamiltonian.system_shape, np.abs(unitary_at_pi), hamiltonian, location_dependent_delay=location_dependent_delay,
                 color_bonds_by='delay', add_colorbar=False)
    edit_graph(None, None)
    plt.savefig(
        f'graphs/time_vortex/pi_mode_Nx_{num_sites_x}_Ny_{num_sites_y}_J_{J:.2f}_pulse_length_{pulse_length:.2f}.pdf')
    plt.show()