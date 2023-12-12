from __future__ import annotations

from itertools import product

import numpy as np
from KSL_with_noise.correcting_fluxes import KSL_flux_corrector
import pandas as pd
from time_dependence_functions import get_g, get_B
from matplotlib import pyplot as plt
from KSL_model import cool_KSL
np.random.seed(0)

def run():
    num_sites_x_list = [4]
    num_sites_y_list = [4]
    g0 = 0.5
    B1 = 0.
    B0 = 12.
    J = 1.
    kappa = 1.
    periodic_bc = (True, False)
    cycles_averaging_buffer = 98
    initial_state = "ground"
    draw_spatial_energy = 'last'

    cycles = 100

    T_list = np.arange(1,21,4)
    trotter_steps_per_T = 40
    errors_per_cycle_per_qubit = [0.]  # [1e-99], np.linspace(1e-99, 0.02, 10)

    for num_sites_x, num_sites_y in product(num_sites_x_list, num_sites_y_list):

        for T in T_list:

            trotter_steps = int(T * trotter_steps_per_T)

            t1 = T / 4
            smoothed_g = lambda t: get_g(t, g0, T, t1)
            smoothed_B = lambda t: get_B(t, B0, B1, T)

            flux_corrector = KSL_flux_corrector(num_sites_x, num_sites_y, periodic_bc)

            for error_rate in errors_per_cycle_per_qubit:

                errors_per_cycle = error_rate * num_sites_x * num_sites_y * 4

                energy_above_ground, flux_x, flux_y = cool_KSL(num_sites_x, num_sites_y, J, kappa, smoothed_g, smoothed_B, initial_state=initial_state, periodic_bc=periodic_bc, cycles=cycles, errors_per_cycle=errors_per_cycle, trotter_steps=trotter_steps, T=T, flux_corrector=flux_corrector, g0=g0, B0=B0, cycles_averaging_buffer=cycles_averaging_buffer, draw_spatial_energy=draw_spatial_energy)

                results_df_averaged = pd.Series(
                    {'num_sites_x': num_sites_x, 'num_sites_y': num_sites_y, 'periodic_bc': periodic_bc, 'J': J,
                     'kappa': kappa, 'g': g0, 'B': B0, 'T': T, 'Nt': trotter_steps, 'N_iter': cycles,
                     'errors_per_cycle_per_qubit': error_rate, 'energy_density': np.mean(energy_above_ground[cycles_averaging_buffer:]) / num_sites_x / num_sites_y,
                     'energy_density_std': np.std(energy_above_ground[cycles_averaging_buffer:]) / num_sites_x / num_sites_y / np.sqrt(cycles-cycles_averaging_buffer),
                     'initial_state': initial_state}).to_frame().transpose()


                plt.figure()
                plt.plot(energy_above_ground)

                print(energy_above_ground[-1])


                with open("KSL_results_B_12.csv", 'a') as f:
                    results_df_averaged.to_csv(f, mode='a', header=f.tell()==0, index=False)

if __name__ == '__main__':
    run()