from time import time
from typing import Callable
import cmocean

import numpy as np
import scipy

from free_fermion_hamiltonian import MajoranaFreeFermionHamiltonian, MajoranaSingleParticleDensityMatrix, get_fermion_bilinear_unitary
from scipy.linalg import eigh, eig
from matplotlib import pyplot as plt


def get_J(t):
    # periodic function with period 1 applying a plus between t=0 and t=1/3
    return int(t - np.floor(t) < 1/3)


def get_floquet_KSL_model(num_sites_x, num_sites_y, J):
    num_sublattices = 2
    lattice_shape = (num_sites_x, num_sites_y, num_sublattices)
    system_shape = lattice_shape*2

    hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)

    site_offset_x = (0, 0)
    site_offset_y = (1, 0)
    site_offset_z = (0, 1)
    add_J_term(J, 'x', hamiltonian, site_offset_x, 0, 1, lattice_shape, 0)
    add_J_term(J, 'y', hamiltonian, site_offset_y, 1, 0, lattice_shape, 1/3)
    add_J_term(J, 'z', hamiltonian, site_offset_z, 1, 0, lattice_shape, 2/3)

    return hamiltonian


def add_J_term(J, alpha, hamiltonian, site_offset_x, sublattice1, sublattice2, lattice_shape, t0):
    for site1 in np.ndindex(tuple(lattice_shape[:-1] - np.array(site_offset_x))):
        site2 = tuple(np.array(site1) + np.array(site_offset_x))
        hamiltonian.add_term(name=f'J{alpha}_{site1}', strength=J, sublattice1=sublattice1, sublattice2=sublattice2, site1=site1, site2=site2,
                             time_dependence=lambda t: get_J(t-t0))


def hexagonal_lattice_site_to_x_y(site):
    # site is a tuple of the form (x, y, sublattice)
    x = 1.5*(site[0] + site[1]) + site[2]
    y = (site[1]-site[0])*np.sqrt(3)/2
    return x, y


def draw_state(state, system_shape, hamiltonian:MajoranaFreeFermionHamiltonian = None, circle_radius=1/3):
    # draw all sites as circles with size according to the absolute value of the state and color according to the phase
    # this part draws the hamiltonian terms as lines between the sites with color according to the term name being x, y or z
    # the lines are drawn at the center of the site but are not contained in a circle of radius circle_radius which is also drawn
    arrow_to_circle_scale = 0.9
    fig, ax = plt.subplots()
    xyz_to_color = {'x': 'b', 'y': 'g', 'z': 'r'}
    if hamiltonian is not None:
        for name, term in hamiltonian.terms.items():
            site1 = term.site1
            site2 = term.site2
            sublattice1 = term.sublattice1
            sublattice2 = term.sublattice2
            x1, y1 = hexagonal_lattice_site_to_x_y((*site1, sublattice1))
            x2, y2 = hexagonal_lattice_site_to_x_y((*site2, sublattice2))
            x1_at_circle_edge, y1_at_circle_edge = x1 + circle_radius*(x2-x1), y1 + circle_radius*(y2-y1)
            x2_at_circle_edge, y2_at_circle_edge = x2 - circle_radius*(x2-x1), y2 - circle_radius*(y2-y1)
            color = xyz_to_color[name[1]]
            plt.plot([x1_at_circle_edge, x2_at_circle_edge], [y1_at_circle_edge, y2_at_circle_edge], color=color, linewidth=2)
    # this part draws the state by drawing arrows pointing according to the phase and with size according to the absolute value
    # the arrows are drawn at the center of the site and are contained in a circle of radius 1/2 which is also drawn
    lattice_shape = system_shape[:len(system_shape) // 2]
    state = state.reshape(lattice_shape)
    phase = np.angle(state)
    for site in np.ndindex(lattice_shape):
        x, y = hexagonal_lattice_site_to_x_y(site)
        site_phase = phase[site]
        circle = plt.Circle((x, y), circle_radius, color='k', fill=False)
        ax.add_patch(circle)
        strength = np.array(np.abs(state[site])/np.max(np.abs(state))).reshape(1,1)
        ax.quiver(x, y, arrow_to_circle_scale*2*circle_radius*np.cos(site_phase), arrow_to_circle_scale*2*circle_radius*np.sin(site_phase),
                  # headlength=2*circle_radius,
                  # headaxislength=2*circle_radius,
                  # width=0.01,
                  # headwidth=3*circle_radius,
                  scale=1,
                  units='xy', pivot='middle', alpha=strength)
        plt.axis('equal')


if __name__ == "__main__":
    integration_params = dict(name='vode', nsteps=2000, rtol=1e-10, atol=1e-14)
    J = 3*np.pi/4
    hamiltonian = get_floquet_KSL_model(5,5, J=J)
    unitary = hamiltonian.full_cycle_unitary_faster(integration_params, 0, 1)
    plt.imshow(np.real(unitary))
    plt.colorbar()
    plt.show()
    phases, states = eig(unitary)
    energies = np.angle(phases)
    # plt.figure()
    # plt.plot(energies, 'o')
    # plt.show()
    for state, energy in zip(states.T, energies):
        draw_state(state, hamiltonian.system_shape, hamiltonian)
        plt.title(f'energy/$\pi$ = {energy/np.pi}')
        plt.show()