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
            unitary_vs_delta_t.append(unitary.copy())
        return np.array(unitary_vs_delta_t)

    def get_time_averaged_unitary(self, cycles=1, fourier=False):
        unitary_vs_delta_t_averaged = np.zeros((cycles*self.steps, self.matrix_shape, self.matrix_shape))
        for i in range(self.steps):
            print(i)
            unitary_vs_delta_t = self.get_unitary_vs_delta_t(i, cycles=cycles)
            unitary_vs_delta_t_averaged += unitary_vs_delta_t
        unitary_vs_delta_t_averaged /= self.steps
        if fourier:
            angular_frequencies, unitary_vs_omega = calculate_fourier_transform(unitary_vs_delta_t_averaged, dt=self.hamiltonian.dt, axis=0)
            return angular_frequencies, unitary_vs_omega
        else:
            return unitary_vs_delta_t_averaged


if __name__ == '__main__':
    J = np.pi / 4 * 1#np.pi/4*0.9#
    pulse_length = 0.05#1/2#
    vortex_location = 'plaquette'#None#
    num_sites_x = 6
    num_sites_y = 6

    hamiltonian, location_dependent_delay, vortex_center = get_floquet_KSL_model(num_sites_x, num_sites_y, J=J,
                                                                                 pulse_length=pulse_length,
                                                                                 vortex_location=vortex_location)
    steps = 100
    cycles = 100
    local_spectral_function = LocalSpectralFunction(hamiltonian, steps)
    angular_frequencies, unitary_vs_omega = local_spectral_function.get_time_averaged_unitary(cycles=cycles, fourier=True)
    unitary_vs_omega_diag = np.diagonal(unitary_vs_omega, axis1=1, axis2=2)
    unitary_vs_omega_diag = unitary_vs_omega_diag.reshape(steps*cycles, *hamiltonian.system_shape[:len(hamiltonian.system_shape)//2])
    unitary_vs_omega = unitary_vs_omega.reshape(steps*cycles, *hamiltonian.system_shape)
    plt.plot(angular_frequencies, np.abs(unitary_vs_omega[:, 3, 3, 0, 3, 3, 0]))
    plt.show()

    index_frequency_pi = np.argmin(np.abs(angular_frequencies - np.pi))
    draw_lattice(hamiltonian.system_shape, np.abs(unitary_vs_omega_diag[index_frequency_pi, :, :, :]), hamiltonian, location_dependent_delay=location_dependent_delay,
                 color_bonds_by='delay', add_colorbar=False)
    edit_graph(None, None)
    plt.savefig(
        f'graphs/time_vortex/pi_mode_Nx_{num_sites_x}_Ny_{num_sites_y}_J_{J:.2f}_pulse_length_{pulse_length:.2f}.pdf')
    plt.show()