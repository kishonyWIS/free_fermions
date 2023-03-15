import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

from one_d_ising import get_smoothed_func, get_g, get_B, get_TFI_model
np.random.seed(0)

num_sites = 2
g0 = 0.5
B1 = 0.
B0 = 3.
T = 30.
t1 = T / 4


# Es = []
# Bs = np.linspace(0,3,100)
# for B in Bs:
#     Es.append(get_translationally_invariant_spectrum(J=J,h=h,k=0,g=0,B=B))
# plt.plot(Bs, Es)
# plt.show()

smoothed_g_before_zeroing = lambda t: get_smoothed_func(t, lambda tt: get_g(tt, g0, T, t1), T/10)
smoothed_B_before_zeroing = lambda t: get_smoothed_func(t, lambda tt: get_B(tt, B0, B1, T, t1), T/10)
smoothed_g = lambda t: smoothed_g_before_zeroing(t) - smoothed_g_before_zeroing(T)
smoothed_B = lambda t: smoothed_B_before_zeroing(t) - smoothed_B_before_zeroing(T)

# ts = np.linspace(0,T,1000)
# gs = []
# Bs = []
# for t in ts:
#     gs.append(smoothed_g(t))
#     Bs.append(smoothed_B(t))
# plt.plot(ts, gs, label='g')
# plt.plot(ts, Bs, label='B')
# plt.legend()
# plt.show()


integration_params = dict(name='vode', nsteps=20000, rtol=1e-8, atol=1e-12)


# Ud = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)
# steps_list = np.logspace(0,4,15, dtype=int)
# errors = []
# L1_errors = []
# for steps in steps_list:
#     t0 = time()
#     Ud_trotter = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=steps)
#     tf = time()
#     print(tf - t0)
#     errors.append(np.sum(np.abs(Ud - Ud_trotter)) / np.sum(np.abs(Ud)))
#     L1_errors.append(np.linalg.norm(1j*scipy.linalg.logm(Ud @ Ud_trotter.T.conj()), ord=2))
# plt.loglog(steps_list, errors, '.')
# plt.figure()
# plt.loglog(steps_list, L1_errors, '.')
# plt.show()


trotter_steps = 100
cycles = 50
errors_per_cycle_per_qubit = [1e-10, 2e-2] #np.linspace(1e-10, 0.02, 10)
errors_per_cycle = errors_per_cycle_per_qubit * num_sites * 2
hs = [0.5, 1]
Js = [1, 0.5]
periodic_bc = True

columns = ["Ns", "periodic_bc", "drop_one_g_for_odd_bath_signs", "J", "h", "V", "Nt", "N_iter", "errors_per_cycle_per_qubit", "energy_density", "energy_density_std"];
results_df = pd.DataFrame(columns=columns)

for i_h_J, (h, J) in enumerate(zip(hs, Js)):
    hamiltonian, S, decoupled_hamiltonian_with_gauge, E_gs, all_errors_unitaries, errors_effect_gauge = \
        get_TFI_model(num_sites, h, J, smoothed_g, smoothed_B, initial_state='ground', periodic_bc=periodic_bc)
    Ud = hamiltonian.full_cycle_unitary_trotterize(0, T, steps=trotter_steps)
    average_Es = []
    for error_rate in errors_per_cycle_per_qubit:
        Es = []
        cycle = 0
        time_to_error = np.random.exponential(T / (error_rate * num_sites * 2))
        time_in_current_cycle = 0.
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
                Es.append(S.get_energy(decoupled_hamiltonian_with_gauge.get_matrix()))
                cycle += 1
                print(cycle)
                time_in_current_cycle = 0
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
        plt.plot(Es)
        plt.plot([E_gs]*len(Es))

        new_row = pd.DataFrame({'Ns': num_sites, 'periodic_bc': periodic_bc, 'drop_one_g_for_odd_bath_signs': False, 'J': J,
                                'h': h, 'V': 0, 'Nt': trotter_steps, 'N_iter': cycles,
                                'errors_per_cycle_per_qubit': error_rate, 'energy_density': (np.mean(Es[2:]) - E_gs) / num_sites,
                                'energy_density_std': np.std(Es[2:]) / num_sites / np.sqrt(cycles)},
                               index=[0])
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        average_Es.append(np.mean(Es[2:]))
    plt.figure(100)
    plt.plot(errors_per_cycle_per_qubit, np.array(average_Es - E_gs) / num_sites, linestyle='None', marker='o', label=f'J = {J}, h = {h}')

with open("results_python_energy_density_vs_error_rate.csv", 'a') as f:
    results_df.to_csv(f, mode='a', header=f.tell()==0, index=False)
plt.xlabel('Errors per cycle per qubit', fontsize='20')
plt.ylabel('Energy density', fontsize='20')
plt.tick_params(axis='both', which='major', labelsize=15)
plt.legend(fontsize=15)
plt.tight_layout()
# plt.savefig(f'graphs/energy_vs_error_rate_steps_{trotter_steps}_cycles_{cycles}_sites_{num_sites}.pdf')
plt.show()