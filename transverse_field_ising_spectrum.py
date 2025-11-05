import numpy as np
import matplotlib.pyplot as plt

class TransverseFieldIsingSpectrum:
    """
    Compute the spectrum of excitations in the transverse field Ising model
    using the fermionized picture.
    
    The Hamiltonian in the fermionized picture is:
    H = (i/4) * sum_{j,k} A_{jk} C_j C_k
    
    where A is an antisymmetric matrix and C_j are Majorana fermion operators.
    The spectrum is given by the eigenvalues of the matrix iA.
    """
    
    def __init__(self, J: float, h_x: float, N: int):
        """
        Initialize the Transverse Field Ising model with open boundary conditions.
        
        Parameters:
        -----------
        J : float
            Coupling strength between nearest neighbors
        h_x : float
            Transverse field strength
        N : int
            Number of sites
        """
        self.J = J
        self.h_x = h_x
        self.N = N
        
        # Build the antisymmetric matrix A
        self.A_matrix = self._build_A_matrix()
        
        # Compute the spectrum
        self.eigenvalues = self._compute_spectrum()
        self.ground_state_energy = self._compute_ground_state_energy()
    
    def _build_A_matrix(self) -> np.ndarray:
        """
        Build the antisymmetric matrix A for the fermionized Hamiltonian.
        
        The matrix A has the form:
        A_{j,k} = J * (δ_{j+1,k} - δ_{j,k+1}) + h_x * (δ_{j,k} - δ_{j+1,k+1})
        
        Returns:
        --------
        np.ndarray
            The antisymmetric matrix A of size (2N, 2N)
        """
        # The fermionized picture requires 2N Majorana fermions
        size = 2 * self.N
        A = np.zeros((size, size))
        
        for j in range(self.N):
            # Site j corresponds to Majorana fermions 2j and 2j+1
            c_j = 2 * j
            c_j_next = 2 * (j + 1)
            
            A[c_j, c_j + 1] = self.h_x
            A[c_j + 1, c_j] = -self.h_x

            if j < self.N - 1:
                A[c_j+1, c_j_next] = self.J
                A[c_j_next, c_j+1] = -self.J
        
        return A
    
    def _compute_spectrum(self) -> np.ndarray:
        """
        Compute the spectrum by diagonalizing the matrix iA.
        
        Returns:
        --------
        np.ndarray
            Eigenvalues of iA
        """
        # Compute eigenvalues of iA
        iA = 1j * self.A_matrix
        eigenvalues = np.linalg.eigvals(iA)
        
        # Sort eigenvalues by real part
        sorted_indices = np.argsort(eigenvalues.real)
        eigenvalues = eigenvalues[sorted_indices]
        
        return eigenvalues
    
    def _compute_ground_state_energy(self) -> float:
        """
        Compute the ground state energy as the sum of negative eigenvalues.
        
        Returns:
        --------
        float
            Ground state energy
        """
        # Take the real part of eigenvalues and sum the negative ones
        real_eigenvalues = self.eigenvalues.real
        negative_eigenvalues = real_eigenvalues[real_eigenvalues < 0]
        
        return np.sum(negative_eigenvalues)
    
    def get_spectrum(self) -> np.ndarray:
        """Get the spectrum of excitations."""
        return self.eigenvalues
    
    def get_ground_state_energy(self) -> float:
        """Get the ground state energy."""
        return self.ground_state_energy
    
    def get_gap(self) -> float:
        """Get the energy gap (smallest positive eigenvalue)."""
        positive_eigenvalues = self.eigenvalues[self.eigenvalues.real >= 0]
        return np.min(positive_eigenvalues.real)

    def get_second_excited_state_energy(self) -> float:
        """Get the energy of the second excited state."""
        positive_eigenvalues = self.eigenvalues[self.eigenvalues.real >= 0]
        return np.sort(positive_eigenvalues.real)[1]
    
    def plot_spectrum(self, save_path: str = None):
        """
        Plot the spectrum of excitations.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        indices = np.arange(len(self.eigenvalues))
        plt.plot(indices, self.eigenvalues.real, 'o-', alpha=0.7)
        plt.xlabel('Eigenvalue index')
        plt.ylabel('Real part')
        plt.title('Spectrum of excitations')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def print_summary(self):
        """Print a summary of the results."""
        print(f"Transverse Field Ising Model - open boundary conditions")
        print(f"Parameters: J = {self.J}, h_x = {self.h_x}, N = {self.N}")
        print(f"Ground state energy: {self.ground_state_energy:.6f}")
        print(f"Energy gap: {self.get_gap():.6f}")
        print(f"Second excited state energy: {self.get_second_excited_state_energy():.6f}")
        print(f"Number of eigenvalues: {len(self.eigenvalues)}")
        print(f"Eigenvalue range: [{self.eigenvalues.real.min():.6f}, {self.eigenvalues.real.max():.6f}]")


def main():
    """Example usage and demonstration."""
    
    # Example parameters
    J = 1.0
    h_x = 0.5
    N = 20
    
    print("=== Transverse Field Ising Model Spectrum ===")
    print()
    
    # Create model with open boundary conditions
    model = TransverseFieldIsingSpectrum(J, h_x, N)
    model.print_summary()
    print()
    
    # Plot the spectrum
    print("Plotting spectrum...")
    model.plot_spectrum()


if __name__ == "__main__":
    main()
