from itertools import product
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from time_dependence_functions import get_g, get_B
from translational_invariant_KSL import get_KSL_model, get_Delta, get_f
from scipy.linalg import expm
from scipy.optimize import minimize

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Parameters from original file
g0 = 0.5
B1 = 0.
B0 = 7.
kappa = 1.
Jx = 1.
Jy = 1.
Jz = 1.

# Variational circuit parameters
T = 50
p = 10  # Number of layers in the variational circuit
n_k_points_train = 1 + 6*1  # Training grid size
n_k_points_test = 1 + 6*3   # Testing grid size

smoothed_g = lambda tt: get_g(tt, g0, T, T/4) #lambda tt: 0#
smoothed_B = lambda tt: get_B(tt, B0, B1, T) #lambda tt: 0#

def get_chern_number_from_single_particle_dm(single_particle_dm):
    """Calculate Chern number from single-particle density matrix"""
    dP_dkx = np.diff(single_particle_dm, axis=0)[:,:-1,:,:]
    dP_dky = np.diff(single_particle_dm, axis=1)[:-1,:,:,:]
    P = single_particle_dm[:-1,:-1,:,:]
    integrand = np.zeros(P.shape[0:2],dtype=complex)
    for i_kx, i_ky in product(range(P.shape[0]), repeat=2):
        integrand[i_kx,i_ky] = np.trace(P[i_kx,i_ky,:,:] @ (dP_dkx[i_kx,i_ky,:,:] @ dP_dky[i_kx,i_ky,:,:] - dP_dky[i_kx,i_ky,:,:] @ dP_dkx[i_kx,i_ky,:,:]))
    return (np.sum(integrand)/(2*np.pi)).imag

def pauli_exponentiation(a_n):
    """
    Compute exp(i * a * (n_vec · σ)) using the formula:
    exp(i * a * (n_vec · σ)) = I * cos(a) + i * (n_vec · σ) * sin(a)
    
    Args:
        a_n: unormalized vector a*n_vec
    
    Returns:
        2x2 unitary matrix
    """

    a = np.linalg.norm(a_n)
    if a == 0:
        return np.eye(2, dtype=complex)
    n_vec = a_n / a
    
    # Identity matrix
    I = np.eye(2, dtype=complex)
    
    # Construct n_vec · σ
    n_dot_sigma = n_vec[0] * sigma_x + n_vec[1] * sigma_y + n_vec[2] * sigma_z
    
    # Apply the formula
    return I * np.cos(a) + 1j * n_dot_sigma * np.sin(a)


def create_variational_circuit(strength_durations, kx, ky):
    """
    Create variational circuit unitary from individual Hamiltonian terms
    
    Args:
        strength_durations: Dictionary with keys ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B'] 
                           and values as arrays of length p (number of layers)
                           Each value represents the combined strength*duration for that term
        kx, ky: momentum values
    
    Returns:
        Ud: 6x6 unitary matrix representing the variational circuit
    """
    Ud = np.eye(6, dtype=complex)
    
    # Apply p layers
    for layer in range(p):
        # Get time for this layer (distributed across the period T)
        t = layer * T / p
        
        # Get the base coefficients at this time
        Delta = get_Delta(kx, ky, kappa)
        g_t = smoothed_g(t)
        B_t = smoothed_B(t)

        # Apply each term with its combined strength*duration for this layer
        for term_name, strength_duration in strength_durations.items():
            strength_duration_val = strength_duration[layer]

            U_6x6 = np.eye(6, dtype=complex)
            
            if term_name in ['Jx', 'Jy', 'Jz', 'kappa']:
                # These terms act on the first 2x2 block [:2, :2]
                if term_name == 'Jx':
                    a_n = np.array([0, -2*Jx * strength_duration_val, 0])
                elif term_name == 'Jy':
                    a_n = 2*Jy * strength_duration_val * np.array([-np.sin(kx), -np.cos(kx), 0])
                elif term_name == 'Jz':
                    a_n = 2*Jz * strength_duration_val * np.array([-np.sin(ky), -np.cos(ky), 0])
                elif term_name == 'kappa':
                    a_n = np.array([0, 0, Delta * strength_duration_val])
                
                U_term = pauli_exponentiation(a_n)
                
                # Start with identity and place the 2x2 unitary in the first block
                U_6x6[:2, :2] = U_term
                
            elif term_name == 'g':
                a_n = np.array([0, -2*g_t * strength_duration_val, 0])
                U_term = pauli_exponentiation(a_n)
                # insert this on the submatrix [[0,2],[0,2]] and on [[1,3],[1,3]]
                U_6x6[np.ix_([0,2],[0,2])] = U_term
                U_6x6[np.ix_([1,3],[1,3])] = U_term

            elif term_name == 'B':
                a_n = np.array([0, 2*B_t * strength_duration_val, 0])
                U_term = pauli_exponentiation(a_n)
                # insert this on the submatrix [[2,4],[2,4]] and on [[3,5],[3,5]]
                U_6x6[np.ix_([2,4],[2,4])] = U_term
                U_6x6[np.ix_([3,5],[3,5])] = U_term
            
            Ud = U_6x6 @ Ud
    
    return Ud

def trotterized_evolution_parameters():
    """
    Create trotterized evolution parameters based on the original adiabatic evolution
    This provides a starting point for the variational optimization
    """
    # Time step for trotterization
    dt = T / p
    
    # Initialize strength*duration for each term and layer
    strength_durations = {
        'Jx': np.ones(p)*dt,
        'Jy': np.ones(p)*dt,
        'Jz': np.ones(p)*dt,
        'kappa': np.ones(p)*dt,
        'g': np.ones(p)*dt,
        'B': np.ones(p)*dt
    }    
    return strength_durations

def single_cooling_cycle_variational(kx, ky, strength_durations):
    """
    Perform a single cooling cycle using the variational circuit
    """
    # Create the variational circuit unitary
    Ud = create_variational_circuit(strength_durations, kx, ky)
    
    # Get the KSL model (we'll use this for the state and ground state energy)
    f = get_f(kx, ky, Jx, Jy, Jz)
    Delta = get_Delta(kx, ky, kappa)
    
    num_cooling_sublattices = 2
    
    # Get the KSL model for state initialization and ground state energy
    hamiltonian, S, E_gs = get_KSL_model(
        f=f, Delta=Delta, g=smoothed_g, B=smoothed_B, 
        initial_state='product', num_cooling_sublattices=num_cooling_sublattices
    )

    # Apply the variational circuit
    S.evolve_with_unitary(Ud)
    
    # Calculate final energy
    E_final = S.get_energy(hamiltonian.get_matrix(T))
    E_diff = E_final - E_gs
    
    return S, E_diff, E_gs

def strength_durations_to_vector(strength_durations):
    """Convert strength_durations dictionary to a flat vector for optimization"""
    vector = []
    for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
        vector.extend(strength_durations[term])
    return np.array(vector)

def vector_to_strength_durations(vector):
    """Convert flat vector back to strength_durations dictionary"""
    strength_durations = {}
    start_idx = 0
    for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
        end_idx = start_idx + p
        strength_durations[term] = vector[start_idx:end_idx]
        start_idx = end_idx
    return strength_durations

def objective_function(vector, kx_list, ky_list, verbose=False):
    """
    Objective function to minimize: energy density
    """
    # Convert vector back to strength_durations
    strength_durations = vector_to_strength_durations(vector)
    
    # Initialize energy difference array
    E_diff = np.zeros((len(kx_list), len(ky_list)))
    
    if verbose:
        print(f"    Evaluating objective function on {len(kx_list)}x{len(ky_list)} grid...")
    
    # Loop over momentum space
    for i_kx, kx in enumerate(kx_list):
        if verbose and i_kx % 2 == 0:  # Print every other kx for brevity
            print(f"      Processing kx={kx:.3f} ({i_kx+1}/{len(kx_list)})")
        for i_ky, ky in enumerate(ky_list):
            # Perform single cooling cycle
            S, E_diff_val, E_gs = single_cooling_cycle_variational(kx, ky, strength_durations)
            E_diff[i_kx, i_ky] = E_diff_val
    
    # Calculate average energy density
    energy_density = np.nanmean(E_diff) / 2  # Divide by 2 because we count k and -k together
    
    if verbose:
        print(f"    Objective function result: {energy_density:.6f}")
    
    return energy_density

def optimize_strength_durations(kx_list, ky_list, initial_strength_durations=None, method='L-BFGS-B'):
    """
    Optimize strength_durations to minimize energy density
    
    Args:
        kx_list, ky_list: momentum space grid
        initial_strength_durations: initial guess (if None, uses trotterized evolution)
        method: optimization method ('L-BFGS-B', 'SLSQP', etc.)
    
    Returns:
        optimized_strength_durations: dictionary with optimized parameters
        optimization_result: scipy optimization result
    """
    print("Starting optimization...")
    
    # Get initial guess
    if initial_strength_durations is None:
        initial_strength_durations = trotterized_evolution_parameters()
    
    # Convert to vector
    initial_vector = strength_durations_to_vector(initial_strength_durations)
    # initial_vector = np.zeros_like(initial_vector)
    
    # Set bounds: allow both positive and negative values for flexibility
    # but keep them reasonable (e.g., -10 to 10)
    bounds = [(-10.0, 10.0)] * len(initial_vector)
    
    print(f"Optimizing {len(initial_vector)} parameters using {method}")
    print(f"Initial energy density: {objective_function(initial_vector, kx_list, ky_list, verbose=True):.6f}")
    
    # Set up optimization options
    options = {'maxiter': 20, 'disp': True, 'ftol': 1e-9, 'gtol': 1e-5}
    max_iter = options['maxiter']
    
    # Create callback function with access to kx_list, ky_list
    iteration_count = [0]  # Use list to allow modification in nested function
    def callback(xk):
        iteration_count[0] += 1
        current_energy = objective_function(xk, kx_list, ky_list, verbose=False)
        print(f"  Iteration {iteration_count[0]}/{max_iter}: Energy density = {current_energy:.6f}")
        return False
    
    # Perform optimization
    result = minimize(
        objective_function,
        initial_vector,
        args=(kx_list, ky_list),
        method=method,
        bounds=bounds,
        callback=callback,
        options=options
    )
    
    # Convert result back to strength_durations
    optimized_strength_durations = vector_to_strength_durations(result.x)
    
    print(f"Optimization completed!")
    print(f"Final energy density: {result.fun:.6f}")
    print(f"Success: {result.success}")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Gradient evaluations: {result.njev}")
    
    return optimized_strength_durations, result

def load_optimized_parameters(filename='optimized_strength_durations.npz'):
    """Load previously optimized parameters from file"""
    data = np.load(filename)
    strength_durations = {}
    for term in ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']:
        strength_durations[term] = data[term]
    print(f"Loaded optimized parameters from {filename}")
    return strength_durations

def plot_results(E_diff, optimized_strength_durations, grid_size, title_suffix="", show_training_points=False):
    """
    Unified plotting function for results visualization
    
    Args:
        E_diff: Energy difference array
        optimized_strength_durations: Dictionary of optimized parameters
        grid_size: Size of the momentum grid used
        title_suffix: Additional text for plot titles
        show_training_points: Whether to highlight training data points (for test grid)
    """
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Energy difference heatmap
    plt.subplot(1, 2, 1)
    
    # Create coordinate arrays for pcolormesh
    kx_coords = np.linspace(-np.pi, np.pi, grid_size)
    ky_coords = np.linspace(-np.pi, np.pi, grid_size)
    
    # Use pcolormesh instead of imshow for better coordinate control
    plt.pcolormesh(kx_coords, ky_coords, E_diff)
    
    # Add training points overlay if requested and this is a test grid
    if show_training_points and grid_size == n_k_points_test:
        # Calculate exact indices for training points
        # Training grid: 7x7, Test grid: 19x19
        # The training points should be evenly spaced in the test grid
        
        # Calculate the step size to get exactly 7 points from 19
        step = (grid_size - 1) // (n_k_points_train - 1)  # (19-1)/(7-1) = 18/6 = 3
        
        # Get the exact indices for training points
        training_indices = np.arange(0, grid_size, step)
        
        # Ensure we have exactly 7 points
        if len(training_indices) > n_k_points_train:
            training_indices = training_indices[:n_k_points_train]
        elif len(training_indices) < n_k_points_train:
            # If we don't have enough points, add the last point
            training_indices = np.append(training_indices, grid_size - 1)
        
        # Get the exact coordinates for training points
        kx_train_coords = kx_coords[training_indices]
        ky_train_coords = ky_coords[training_indices]
        
        # Plot training points as red crosses - these will be perfectly aligned
        for i, kx_coord in enumerate(kx_train_coords):
            for j, ky_coord in enumerate(ky_train_coords):
                plt.scatter(kx_coord, ky_coord, c='red', marker='x', s=50, linewidths=2, 
                           label='Training points' if i==0 and j==0 else "")
    
    plt.colorbar(label='Energy difference')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title(f'Energy difference {title_suffix}(Grid: {grid_size}x{grid_size})')
    if show_training_points and grid_size == n_k_points_test:
        plt.legend()
    
    # Plot 2: Optimized parameters
    plt.subplot(1, 2, 2)
    terms = ['Jx', 'Jy', 'Jz', 'kappa', 'g', 'B']
    for i, term in enumerate(terms):
        plt.plot(optimized_strength_durations[term], label=term, marker='o')
    plt.xlabel('Layer')
    plt.ylabel('Strength*duration')
    plt.title('Optimized strength*duration parameters')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def print_results(energy_density, total_chern_number, system_chern_number, 
                 bath_chern_number, opt_result=None, grid_size=None, phase_name="RESULTS"):
    """
    Unified function to print results
    
    Args:
        energy_density: Average energy density
        total_chern_number: Total Chern number
        system_chern_number: System Chern number
        bath_chern_number: Bath Chern number
        opt_result: Optimization result object (optional)
        grid_size: Size of momentum grid (optional)
        phase_name: Name of the phase (e.g., "TRAINING", "TESTING", "FINAL RESULTS")
    """
    print(f"\n" + "="*60)
    print(phase_name)
    print("="*60)
    
    if grid_size:
        if isinstance(grid_size, str):
            print(f"Grid size: {grid_size}")
        else:
            print(f"Grid size: {grid_size}x{grid_size} = {grid_size**2} points")
    
    print(f"Energy density: {energy_density:.6f}")
    print(f"Total Chern number: {total_chern_number:.6f}")
    print(f"System Chern number: {system_chern_number:.6f}")
    print(f"Bath Chern number: {bath_chern_number:.6f}")
    
    if opt_result:
        print(f"Optimization success: {opt_result.success}")
        print(f"Optimization iterations: {opt_result.nit}")
    
    print("="*60)

def run_simulation_on_grid(kx_list, ky_list, strength_durations, grid_name="", plot=True):
    """
    Run simulation on a given momentum grid
    
    Args:
        kx_list, ky_list: momentum space grids
        strength_durations: circuit parameters
        grid_name: name for display purposes
        plot: whether to plot results
    
    Returns:
        tuple: (E_diff, single_particle_dm, total_chern_number, system_chern_number, bath_chern_number, energy_density)
    """
    grid_size = len(kx_list)
    print(f"{grid_name} grid: {grid_size}x{grid_size} = {grid_size**2} points")
    
    # Initialize arrays for results
    E_diff = np.zeros((grid_size, grid_size))
    single_particle_dm = np.zeros((grid_size, grid_size, 6, 6), dtype=complex)
    
    # Loop over momentum space
    for i_kx, kx in enumerate(kx_list):
        print(f'Processing kx={kx:.3f} ({i_kx+1}/{grid_size})')
        for i_ky, ky in enumerate(ky_list):
            # Perform single cooling cycle with given parameters
            S, E_diff_val, E_gs = single_cooling_cycle_variational(kx, ky, strength_durations)
            
            # Store results
            E_diff[i_kx, i_ky] = E_diff_val
            single_particle_dm[i_kx, i_ky, :, :] = S.matrix
    
    # Calculate Chern numbers
    total_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm)
    system_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,:2,:2])
    bath_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,2:,2:])
    
    # Calculate average energy density
    energy_density = np.nanmean(E_diff) / 2  # Divide by 2 because we count k and -k together
    
    if plot:
        plot_results(E_diff, strength_durations, grid_size, f"{grid_name} ")
    
    return E_diff, single_particle_dm, total_chern_number, system_chern_number, bath_chern_number, energy_density

def train_variational_circuit():
    """
    Train the variational circuit using a smaller momentum grid
    """
    print("="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    # Create training momentum space grid
    kx_list_train = np.linspace(-np.pi, np.pi, n_k_points_train)
    ky_list_train = np.linspace(-np.pi, np.pi, n_k_points_train)
    
    print(f"Training grid: {len(kx_list_train)}x{len(ky_list_train)} = {len(kx_list_train)*len(ky_list_train)} points")
    
    # Get initial trotterized evolution parameters
    initial_strength_durations = trotterized_evolution_parameters()
    print(f"Initial strength*duration parameters shape: {[(k, v.shape) for k, v in initial_strength_durations.items()]}")
    
    # Optimize the strength_durations on training data
    optimized_strength_durations, opt_result = optimize_strength_durations(
        kx_list_train, ky_list_train, 
        initial_strength_durations=initial_strength_durations,
        method='L-BFGS-B'
    )
    
    # Save optimized parameters
    np.savez('optimized_strength_durations.npz', **optimized_strength_durations)
    print("Optimized parameters saved to 'optimized_strength_durations.npz'")
    
    return optimized_strength_durations, opt_result

def test_variational_circuit(optimized_strength_durations):
    """
    Test the variational circuit using a larger momentum grid
    """
    print("\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)
    
    # Create testing momentum space grid
    kx_list_test = np.linspace(-np.pi, np.pi, n_k_points_test)
    ky_list_test = np.linspace(-np.pi, np.pi, n_k_points_test)
    
    # Run simulation using helper function
    return run_simulation_on_grid(kx_list_test, ky_list_test, optimized_strength_durations, "Testing", plot=False)

def main():
    """Main function to run the variational circuit simulation with training and testing"""
    print("Starting variational circuit simulation with training and testing...")
    print(f"Using {p} layers for variational circuit")
    
    # Phase 1: Training
    optimized_strength_durations, opt_result = train_variational_circuit()
    
    # Phase 2: Testing
    E_diff, single_particle_dm, total_chern_number, system_chern_number, bath_chern_number, energy_density = test_variational_circuit(optimized_strength_durations)

    # Plot and print results using helper functions
    print_results(energy_density, total_chern_number, system_chern_number, 
                bath_chern_number, opt_result=opt_result, 
                grid_size=f"{n_k_points_train}x{n_k_points_train} (train), {n_k_points_test}x{n_k_points_test} (test)", 
                phase_name="FINAL RESULTS")
    plot_results(E_diff, optimized_strength_durations, n_k_points_test, "Test ", show_training_points=True)
    
if __name__ == "__main__":
        main()

