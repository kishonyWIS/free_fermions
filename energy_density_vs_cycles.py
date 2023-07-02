import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from one_d_ising import get_TFI_model # get_smoothed_func, get_g, get_B,
from time_dependence_functions import get_g, get_B
np.random.seed(0)

num_sites = 100
g0 = 0.5
B1 = 0.
B0 = 3.
T = 800.#([12.5,25.,50.,100.,200.,400.,800.])]
t1 = T / 4

trotter_steps = 100000
cycles = 50

# smoothed_g_before_zeroing = lambda t: get_smoothed_func(t, lambda tt: get_g(tt, g0, T, t1), T/10)
# smoothed_B_before_zeroing = lambda t: get_smoothed_func(t, lambda tt: get_B(tt, B0, B1, T, t1), T/10)
# smoothed_g = lambda t: smoothed_g_before_zeroing(t) - smoothed_g_before_zeroing(T)
# smoothed_B = lambda t: smoothed_B_before_zeroing(t) - smoothed_B_before_zeroing(T)

smoothed_g = lambda t: get_g(t, g0, T, t1)
smoothed_B = lambda t: get_B(t, B0, B1, T)

# integration_params = dict(name='vode', nsteps=20000, rtol=1e-12, atol=1e-16)


errors_per_cycle_per_qubit = [1e-100] #np.linspace(1e-10, 0.02, 10)
errors_per_cycle = errors_per_cycle_per_qubit * num_sites * 2
hs = [0.5]
Js = [1.]
periodic_bc = False

columns = ["Ns", "periodic_bc", "drop_one_g_for_odd_bath_signs", "J", "h", "V", "T", "Nt", "N_iter", "errors_per_cycle_per_qubit", "energy_density"]
results_df = pd.DataFrame(columns=columns)

for i_h_J, (h, J) in enumerate(zip(hs, Js)):
    hamiltonian, S, decoupled_hamiltonian_with_gauge, E_gs, all_errors_unitaries, errors_effect_gauge = \
        get_TFI_model(num_sites, h, J, smoothed_g, smoothed_B, initial_state='random', periodic_bc=periodic_bc)
    Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
    # Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
    average_Es = []
    for error_rate in errors_per_cycle_per_qubit:
        Es = []
        cycle = 0
        time_to_error = np.random.exponential(T / (error_rate * num_sites * 2))
        time_in_current_cycle = 0.

        Es.append(S.get_energy(decoupled_hamiltonian_with_gauge.get_matrix()))
        new_row = pd.DataFrame(
            {'Ns': num_sites, 'periodic_bc': periodic_bc, 'drop_one_g_for_odd_bath_signs': False, 'J': J,
             'h': h, 'V': 0, 'T': T, 'Nt': trotter_steps, 'N_iter': cycle,
             'errors_per_cycle_per_qubit': error_rate, 'energy_density': (Es[-1] - E_gs) / num_sites},
            index=[0])
        results_df = pd.concat([results_df, new_row], ignore_index=True)


        while True:
            if cycle == cycles:
                # finished all cycles
                break
            elif time_to_error == 0:
                # print('apply error')
                error_name = np.random.choice(list(all_errors_unitaries.keys()))
                S.evolve_with_unitary(all_errors_unitaries[error_name])
                # print(error_name)
                if errors_effect_gauge[error_name]:
                    # print('recalculating Ud')
                    # Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
                    Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
                time_to_error = np.random.exponential(T / (error_rate * num_sites * 2))
            elif time_to_error > T and time_in_current_cycle == 0:
                # print('apply a full cycle unitary')
                S.evolve_with_unitary(Ud)
                time_to_error -= T
                time_in_current_cycle = T
            elif time_to_error < T - time_in_current_cycle:
                # print('apply a partial unitary until error')
                # Ud_temp = hamiltonian.full_cycle_unitary_faster(integration_params,
                #                                                 time_in_current_cycle,
                #                                                 time_in_current_cycle + time_to_error)
                steps = int(trotter_steps * time_to_error / T)
                if steps > 0:
                    Ud_temp = hamiltonian.full_cycle_unitary_trotterize(time_in_current_cycle,
                                                                        time_in_current_cycle + time_to_error, steps=steps)
                    S.evolve_with_unitary(Ud_temp)
                time_in_current_cycle += time_to_error
                time_to_error = 0
            elif time_in_current_cycle == T:
                # print('reset')
                S.reset_all_tau()
                cycle += 1
                print(cycle)
                Es.append(S.get_energy(decoupled_hamiltonian_with_gauge.get_matrix()))
                time_in_current_cycle = 0
                new_row = pd.DataFrame(
                    {'Ns': num_sites, 'periodic_bc': periodic_bc, 'drop_one_g_for_odd_bath_signs': False, 'J': J,
                     'h': h, 'V': 0, 'T': T, 'Nt': trotter_steps, 'N_iter': cycle,
                     'errors_per_cycle_per_qubit': error_rate, 'energy_density': (Es[-1] - E_gs) / num_sites},
                    index=[0])
                results_df = pd.concat([results_df, new_row], ignore_index=True)
            elif time_in_current_cycle > 0:
                # print('finish incomplete cycle')
                # Ud_temp = hamiltonian.full_cycle_unitary_faster(integration_params,
                #                                                 time_in_current_cycle,
                #                                                 T)
                steps = int(trotter_steps * (T - time_in_current_cycle) / T)
                if steps > 0:
                    Ud_temp = hamiltonian.full_cycle_unitary_trotterize(time_in_current_cycle, T, steps=steps)
                    S.evolve_with_unitary(Ud_temp)
                time_to_error -= T - time_in_current_cycle
                time_in_current_cycle = T
            else:
                raise 'invalid cycle state'

        print(Es[-1])
        print('ground state energy = ' + str(E_gs))
        plt.figure(i_h_J)
        plt.semilogy(np.arange(len(Es))+1, Es-E_gs)

with open("results_python_energy_density_vs_cycle.csv", 'a') as f:
    results_df.to_csv(f, mode='a', header=f.tell()==0, index=False)