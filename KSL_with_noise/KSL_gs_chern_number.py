from itertools import product
import numpy as np
from free_fermion_hamiltonian import ComplexFreeFermionHamiltonian
from translational_invariant_KSL import get_Delta, get_f

kappa = 1.
Jx = 1.
Jy = 1.
Jz = 1.


def get_chern_number_from_single_particle_dm(single_particle_dm):
    dP_dkx = np.diff(single_particle_dm, axis=0)[:,:-1,:,:]
    dP_dky = np.diff(single_particle_dm, axis=1)[:-1,:,:,:]
    P = single_particle_dm[:-1,:-1,:,:]
    integrand = np.zeros(P.shape[0:2],dtype=complex)
    for i_kx, i_ky in product(range(P.shape[0]), repeat=2):
        integrand[i_kx,i_ky] = np.trace(P[i_kx,i_ky,:,:] @ (dP_dkx[i_kx,i_ky,:,:] @ dP_dky[i_kx,i_ky,:,:] - dP_dky[i_kx,i_ky,:,:] @ dP_dkx[i_kx,i_ky,:,:]))
    return (np.sum(integrand)/(2*np.pi)).imag


for n_k_points in [1+6*nn for nn in [14]]:

    kx_list = np.linspace(-np.pi, np.pi, n_k_points)
    ky_list = np.linspace(-np.pi, np.pi, n_k_points)

    single_particle_dm = np.zeros((n_k_points, n_k_points, 2, 2), dtype=complex)

    for i_kx, kx in enumerate(kx_list):
        print(f'kx={kx}')
        for i_ky, ky in enumerate(ky_list):

            f = get_f(kx, ky, Jx, Jy, Jz)
            Delta = get_Delta(kx, ky, kappa)

            H = ComplexFreeFermionHamiltonian((1, 2, 1, 2))
            H.add_term(name='Delta0', strength=Delta, sublattice1=0, sublattice2=0, site1=[0], site2=[0])
            H.add_term(name='Delta1', strength=-Delta, sublattice1=1, sublattice2=1, site1=[0], site2=[0])
            H.add_term(name='f01', strength=1j*f, sublattice1=0, sublattice2=1, site1=[0], site2=[0])
            H.add_term(name='f10', strength=-1j*np.conj(f), sublattice1=1, sublattice2=0, site1=[0], site2=[0])

            S = H.get_ground_state()

            single_particle_dm[i_kx,i_ky,:,:] = S.matrix


    chern_number = get_chern_number_from_single_particle_dm(single_particle_dm)
    print(chern_number)
