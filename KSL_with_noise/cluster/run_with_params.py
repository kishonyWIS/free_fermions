import numpy as np
from correcting_fluxes import KSL_flux_corrector
import pandas as pd
from time_dependence_functions import get_g, get_B
from matplotlib import pyplot as plt
from KSL_model import cool_KSL
import sys

if __name__ == '__main__':
    np.random.seed(int(sys.argv[1]))
    g0 = float(sys.argv[2])
    B1 = 0.
    B0 = float(sys.argv[3])
    J = float(sys.argv[4])
    kappa = float(sys.argv[5])
    if sys.argv[6] == 'True':
        periodic_bc = True
    elif sys.argv[6] == 'False':
        periodic_bc = False
    elif sys.argv[6] == '(True, False)':
        periodic_bc = (True, False)
    elif sys.argv[6] == '(False, True)':
        periodic_bc = (False, True)
    cycles = int(sys.argv[7])
    cycles_averaging_buffer = int(sys.argv[8])
    initial_state = sys.argv[9]
    draw_spatial_energy = bool(sys.argv[10])
    num_sites_x = int(sys.argv[11])
    num_sites_y = int(sys.argv[12])
    T = float(sys.argv[13])
    trotter_steps = int(sys.argv[14])
    error_rate = float(sys.argv[15])

    t1 = T / 4
    smoothed_g = lambda t: get_g(t, g0, T, t1)
    smoothed_B = lambda t: get_B(t, B0, B1, T)
    flux_corrector = KSL_flux_corrector(num_sites_x, num_sites_y, periodic_bc)


    errors_per_cycle = error_rate * num_sites_x * num_sites_y * 4

    energy_above_ground, flux_x, flux_y = cool_KSL(num_sites_x, num_sites_y, J, kappa, smoothed_g, smoothed_B, initial_state=initial_state, periodic_bc=periodic_bc, cycles=cycles, errors_per_cycle=errors_per_cycle, trotter_steps=trotter_steps, T=T, flux_corrector=flux_corrector, g0=g0, B0=B0, cycles_averaging_buffer=cycles_averaging_buffer, draw_spatial_energy=draw_spatial_energy)

    results_df_averaged = pd.Series(
        {'num_sites_x': num_sites_x, 'num_sites_y': num_sites_y, 'periodic_bc': periodic_bc, 'J': J,
         'kappa': kappa, 'g': g0, 'B': B0, 'T': T, 'Nt': trotter_steps, 'N_iter': cycles,
         'errors_per_cycle_per_qubit': error_rate, 'energy_density': np.mean(energy_above_ground[cycles_averaging_buffer:]) / num_sites_x / num_sites_y,
         'energy_density_std': np.std(energy_above_ground[cycles_averaging_buffer:]) / num_sites_x / num_sites_y / np.sqrt(cycles-cycles_averaging_buffer),
         'initial_state': initial_state}).to_frame().transpose()

    print(energy_above_ground[-1])


    with open("KSL_results.csv", 'a') as f:
        results_df_averaged.to_csv(f, mode='a', header=f.tell()==0, index=False)