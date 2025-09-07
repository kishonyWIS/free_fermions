from itertools import product
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from time_dependence_functions import get_g, get_B
from translational_invariant_KSL import get_KSL_model, get_Delta, get_f
from scipy.linalg import expm
from scipy.optimize import minimize

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
p = 400  # Number of layers in the variational circuit
n_k_points = 1 + 6*1

integration_params = dict(name='vode', nsteps=6000, rtol=1e-6, atol=1e-10)

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

def get_individual_hamiltonian_terms(kx, ky, t):
    """Get individual Hamiltonian term matrices using get_KSL_model for consistency"""
    f = get_f(kx, ky, Jx, Jy, Jz)
    Delta = get_Delta(kx, ky, kappa)
    g_t = smoothed_g(t)
    B_t = smoothed_B(t)
    
    H_terms = {}
    
    # 1. Jx term - only Jx contribution to f
    f_Jx = get_f(kx, ky, Jx, 0, 0)  # Only Jx term
    hamiltonian_Jx, _, _ = get_KSL_model(f=f_Jx, Delta=0, g=0, B=0, 
                                        initial_state='product', num_cooling_sublattices=2)
    H_terms['Jx'] = hamiltonian_Jx.get_matrix()
    
    # 2. Jy term - only Jy contribution to f
    f_Jy = get_f(kx, ky, 0, Jy, 0)  # Only Jy term
    hamiltonian_Jy, _, _ = get_KSL_model(f=f_Jy, Delta=0, g=0, B=0, 
                                        initial_state='product', num_cooling_sublattices=2)
    H_terms['Jy'] = hamiltonian_Jy.get_matrix()
    
    # 3. Jz term - only Jz contribution to f
    f_Jz = get_f(kx, ky, 0, 0, Jz)  # Only Jz term
    hamiltonian_Jz, _, _ = get_KSL_model(f=f_Jz, Delta=0, g=0, B=0, 
                                        initial_state='product', num_cooling_sublattices=2)
    H_terms['Jz'] = hamiltonian_Jz.get_matrix()
    
    # 4. kappa term - only Delta contribution
    hamiltonian_kappa, _, _ = get_KSL_model(f=0, Delta=Delta, g=0, B=0, 
                                           initial_state='product', num_cooling_sublattices=2)
    H_terms['kappa'] = hamiltonian_kappa.get_matrix()
    
    # 5. g term - only g coupling contribution
    hamiltonian_g, _, _ = get_KSL_model(f=0, Delta=0, g=g_t, B=0, 
                                       initial_state='product', num_cooling_sublattices=2)
    H_terms['g'] = hamiltonian_g.get_matrix()
    
    # 6. B term - only B coupling contribution
    hamiltonian_B, _, _ = get_KSL_model(f=0, Delta=0, g=0, B=B_t, 
                                       initial_state='product', num_cooling_sublattices=2)
    H_terms['B'] = hamiltonian_B.get_matrix()
    
    return H_terms

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
        
        # Get individual Hamiltonian terms at this time
        H_terms = get_individual_hamiltonian_terms(kx, ky, t)
        
        # Apply each term with its combined strength*duration for this layer
        for term_name, H_term in H_terms.items():
            strength_duration = strength_durations[term_name][layer]
            U_term = expm(1j * H_term.conj() * strength_duration / 4)
            Ud = U_term @ Ud
    
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

    # ### !!!
    # Ud_old = hamiltonian.full_cycle_unitary_faster(integration_params, 0, T)

    # #compare Ud and Ud_old up to a global phase by minimizing the distance
    # Ud_Ud_old_distance = minimize(lambda x: np.linalg.norm(Ud * np.exp(1j*x) - Ud_old), x0=np.zeros(6))
    # print('='*100)
    # print('Ud and Ud_old distance = ' + str(Ud_Ud_old_distance.fun))
    # print('='*100)
    # #check if both are unitary
    # print('Ud distance from unitary = ' + str(np.linalg.norm(Ud @ Ud.conj().T - np.eye(6))))
    # print('Ud_old distance from unitary = ' + str(np.linalg.norm(Ud_old @ Ud_old.conj().T - np.eye(6))))
    # print('='*100)
    # # round to 5 decimal places
    # print(np.round(Ud, 4))
    # print('='*100)
    # print(np.round(Ud_old, 4))
    # print('='*100)

    # Apply the variational circuit
    S.evolve_with_unitary(Ud)
    
    # Calculate final energy
    E_final = S.get_energy(hamiltonian.get_matrix(T))
    E_diff = E_final - E_gs
    
    return S, E_diff, E_gs

def main():
    """Main function to run the variational circuit simulation"""
    print("Starting variational circuit simulation...")
    
    # Create momentum space grid
    kx_list = np.linspace(-np.pi, np.pi, n_k_points)
    ky_list = np.linspace(-np.pi, np.pi, n_k_points)
    
    # Get trotterized evolution parameters
    strength_durations = trotterized_evolution_parameters()
    
    print(f"Using {p} layers for variational circuit")
    print(f"Strength*duration parameters shape: {[(k, v.shape) for k, v in strength_durations.items()]}")
    
    # Initialize arrays for results
    E_diff = np.zeros((len(kx_list), len(ky_list)))
    single_particle_dm = np.zeros((n_k_points, n_k_points, 6, 6), dtype=complex)
    
    # Loop over momentum space
    for i_kx, kx in enumerate(kx_list):
        print(f'Processing kx={kx:.3f} ({i_kx+1}/{len(kx_list)})')
        for i_ky, ky in enumerate(ky_list):
            try:
                # Perform single cooling cycle
                S, E_diff_val, E_gs = single_cooling_cycle_variational(kx, ky, strength_durations)
                
                # Store results
                E_diff[i_kx, i_ky] = E_diff_val
                single_particle_dm[i_kx, i_ky, :, :] = S.matrix
                
            except Exception as e:
                print(f"Error at kx={kx}, ky={ky}: {e}")
                E_diff[i_kx, i_ky] = np.nan
                single_particle_dm[i_kx, i_ky, :, :] = np.nan
    
    # Calculate Chern numbers
    total_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm)
    system_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,:2,:2])
    bath_chern_number = get_chern_number_from_single_particle_dm(single_particle_dm[:,:,2:,2:])
    
    # Calculate average energy density
    energy_density = np.nanmean(E_diff) / 2  # Divide by 2 because we count k and -k together

    plt.figure()
    plt.imshow(E_diff)
    plt.colorbar()
    plt.show()
    
    print(f"\nResults:")
    print(f"Energy density: {energy_density:.6f}")
    print(f"Total Chern number: {total_chern_number:.6f}")
    print(f"System Chern number: {system_chern_number:.6f}")
    print(f"Bath Chern number: {bath_chern_number:.6f}")
    
if __name__ == "__main__":
    main()
