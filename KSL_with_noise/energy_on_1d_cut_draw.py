from energy_distribution import EnergyDistribution
from matplotlib import pyplot as plt

num_sites_x = 2
num_sites_y = 50
g0 = 0.5
B1 = 0.
B0 = 5.
J = 1.
kappa = 1.
periodic_bc = (True, False)
cycles_averaging_buffer = 3
initial_state = "ground"
draw_spatial_energy = "last"

cycles = 50

trotter_steps = 800

T = 5.
errors_per_cycle_per_qubit = 1e-99

_, axA = plt.subplots()

for trotter_steps in [100, 200, 400, 600, 800]:
    spatial_energy_filename = f'KSL_spatial_energy_nx_{num_sites_x}_ny_{num_sites_y}_T_{T}_error_rate_{errors_per_cycle_per_qubit}_J_{J}_kappa_{kappa}_g_{g0}_B_{B0}_initial_state_{initial_state}_periodic_bc_{periodic_bc}_cycles_{cycles}_trotter_steps_{trotter_steps}_draw_spatial_energy_{draw_spatial_energy}'
    energy_dist = EnergyDistribution.load(spatial_energy_filename+'.pkl')
    # energy_dist.draw_term_on_1d_cut(['kappa_y_sublattice_A_shift_0', 'kappa_y_sublattice_A_shift_1'], ax=axA)
    energy_dist.draw_term_on_1d_cut(['kappa_z_sublattice_A_shift_0', 'kappa_z_sublattice_A_shift_1'], ax=axA)
    # set label of the last plot
    axA.lines[-1].set_label(f'{trotter_steps}')
plt.legend()
plt.show()
print()