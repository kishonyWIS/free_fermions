from floquet_defect_invariants import get_unitary, get_Hamiltonian_lambda, get_unitary_odeint, get_unitary_solve_ivp, get_effective_hamiltonian, get_u_eff
import numpy as np
from scipy.linalg import eig, expm
from matplotlib import pyplot as plt
from itertools import permutations


nsteps = 4
integration_params = dict(name='vode', nsteps=nsteps, rtol=1e-8, atol=1e-10)
epsilon = np.pi

kx_list = np.linspace(0, 2*np.pi, nsteps)
ky_list = np.linspace(0, 2*np.pi, nsteps)
times = np.linspace(0, 1, nsteps)
theta_list = np.linspace(0, 2*np.pi, nsteps)
lamd_list = np.linspace(0, 1, nsteps)

u = np.zeros((len(kx_list), len(ky_list), len(theta_list), len(lamd_list), nsteps, 2, 2), dtype=np.complex128)

#iterate over all the values of kx, ky, theta, lamb,
#and calculate the unitary for each
for i, kx in enumerate(kx_list):
    print(i)
    for j, ky in enumerate(ky_list):
        for k, theta in enumerate(theta_list):
            for l, lamb in enumerate(lamd_list):
                u[i,j,k,l,:,:,:] = get_u_eff(kx, ky, theta, lamb, integration_params, t0=0, tf=1, epsilon=epsilon).transpose(2,0,1)

# calculate the inverse of u at each kx, ky, theta, lamb, time
u_inv = np.linalg.inv(u)

# calculate u_inv times the partial derivative of u with respect to kx, ky, theta, lamb, time
u_inv_du_dkx = np.matmul(u_inv, np.gradient(u, axis=0))
u_inv_du_dky = np.matmul(u_inv, np.gradient(u, axis=1))
u_inv_du_dtheta = np.matmul(u_inv, np.gradient(u, axis=2))
u_inv_du_dlamb = np.matmul(u_inv, np.gradient(u, axis=3))
u_inv_du_dt = np.matmul(u_inv, np.gradient(u, axis=4))

W = 0.0

def perm_parity(lst):
    '''\
    Given a permutation of the digits 0..N in order as a list,
    returns its parity (or sign): +1 for even parity; -1 for odd.
    '''
    parity = 1
    for i in range(0,len(lst)-1):
        if lst[i] != i:
            parity *= -1
            mn = min(range(i,len(lst)), key=lst.__getitem__)
            lst[i],lst[mn] = lst[mn],lst[i]
    return parity

permutations_of_four_elements = permutations(range(4))
permutations_of_four_elements_with_parity = [(p, perm_parity(list(p))) for p in permutations_of_four_elements]


# iterate over the values of kx, ky, theta, lamb, and calculate the integrand for the invariant
for i, kx in enumerate(kx_list):
    print(i)
    for j, ky in enumerate(ky_list):
        for k, theta in enumerate(theta_list):
            for l, lamb in enumerate(lamd_list):
                for m, time in enumerate(times):
                    A = np.zeros((5,2,2), dtype=complex)
                    A[0,:,:] = u_inv_du_dkx[i,j,k,l,m,:,:]
                    A[1,:,:] = u_inv_du_dky[i,j,k,l,m,:,:]
                    A[2,:,:] = u_inv_du_dtheta[i,j,k,l,m,:,:]
                    A[3,:,:] = u_inv_du_dlamb[i,j,k,l,m,:,:]
                    A[4,:,:] = u_inv_du_dt[i,j,k,l,m,:,:]
                    # iterate over the permutations of A_ky, A_theta, A_lamb, A_t
                    for p, parity in permutations_of_four_elements_with_parity:
                        W += 5 * parity * np.trace(np.matmul(A[0,:,:], np.matmul(A[p[0],:,:], np.matmul(A[p[1],:,:], np.matmul(A[p[2],:,:], A[p[3],:,:])))))
                        print(W)

print(u.shape)
print(W)