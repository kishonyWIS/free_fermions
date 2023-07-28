import matplotlib.collections
import numpy as np
from free_fermion_hamiltonian import MajoranaFreeFermionHamiltonian
from scipy.linalg import eig
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Circle
import random
from scipy.stats import gaussian_kde

def edit_graph(xlabel, ylabel, legend_title=None, colorbar_title=None, colormap=None, colorbar_args={}, tight=True, ylabelpad=None):
    sns.set_style("whitegrid")
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix",
          "figure.autolayout": True}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.xlabel(xlabel, fontsize='20', fontname='Times New Roman')
    plt.ylabel(ylabel, fontsize='20', fontname='Times New Roman', labelpad=ylabelpad)
    plt.tick_params(axis='both', which='major', labelsize=15)
    if legend_title:
        l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=15))
        l.set_title(title='$T$', prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
    if colorbar_title:
        if colormap:
            cmap = mpl.cm.ScalarMappable(norm=None, cmap=colormap)
            cmap.set_array([])
            cbar = plt.colorbar(cmap,**colorbar_args)
        else:
            cbar = plt.colorbar(**colorbar_args)
        cbar.set_label(colorbar_title, fontsize='20', fontname='Times New Roman')
        cbar.ax.tick_params(labelsize=15)
    if tight:
        plt.tight_layout()

def get_J(t, pulse_length, delay=0):
    # periodic function with period 1 applying a plus between t=0 and t=pulse_length
    return int(t - delay - np.floor(t - delay) < pulse_length)/pulse_length


def get_J_delayed(pulse_length, delay=0):
    return lambda t: get_J(t, pulse_length=pulse_length, delay=delay)


def get_floquet_KSL_model(num_sites_x, num_sites_y, J, pulse_length=1/20, vortex_location=None):
    num_sublattices = 2
    lattice_shape = (num_sites_x, num_sites_y, num_sublattices)
    system_shape = lattice_shape*2

    hamiltonian = MajoranaFreeFermionHamiltonian(system_shape)

    site_offset_x = (0, 0)
    site_offset_y = (1, 0)
    site_offset_z = (0, 1)
    J_x_strength = np.ones(lattice_shape[:-1] - np.array(site_offset_x)) * J
    J_y_strength = np.ones(lattice_shape[:-1] - np.array(site_offset_y)) * J
    J_z_strength = np.ones(lattice_shape[:-1] - np.array(site_offset_z)) * J
    # J_x_strength[0,0] = 0
    # J_x_strength[-1,-1] = 0
    # J_x_strength[2,2] = -J


    #vortex is given by location_dependent_delay which is the angle/(2*pi) in the x-y plane of the bond location around the vortex
    if vortex_location == 'bond':
        vortex_center = tuple(np.array(hexagonal_lattice_site_to_x_y((num_sites_x // 2, num_sites_y // 2, 0))) + np.array(
            (0.5, 0)))  # on bond
    if vortex_location == 'site':
        vortex_center = tuple(np.array(hexagonal_lattice_site_to_x_y((num_sites_x // 2, num_sites_y // 2, 0))) + np.array(
            (1, 0)))  # on site
    if vortex_location == 'plaquette':
        vortex_center = tuple(np.array(hexagonal_lattice_site_to_x_y((num_sites_x // 2, num_sites_y // 2, 0))) + np.array(
            (-1, 0)))  # on plaquette
    if vortex_location is None:
        location_dependent_delay = None
    else:
        location_dependent_delay = lambda x, y: (np.arctan2(y - vortex_center[1], x - vortex_center[0]) + np.pi/2) / (2*np.pi)

    add_J_term(J_x_strength, 'x', hamiltonian, site_offset_x, 0, 1, lattice_shape, pulse_length=pulse_length, alpha_delay=0, location_dependent_delay=location_dependent_delay)
    add_J_term(J_y_strength, 'y', hamiltonian, site_offset_y, 1, 0, lattice_shape, pulse_length=pulse_length, alpha_delay=1 / 3, location_dependent_delay=location_dependent_delay)
    add_J_term(J_z_strength, 'z', hamiltonian, site_offset_z, 1, 0, lattice_shape, pulse_length=pulse_length, alpha_delay=2 / 3, location_dependent_delay=location_dependent_delay)

    return hamiltonian, location_dependent_delay, vortex_center


def add_J_term(J, alpha, hamiltonian, site_offset, sublattice1, sublattice2, lattice_shape, alpha_delay, pulse_length=1/20, location_dependent_delay=None):
    for site1 in np.ndindex(tuple(lattice_shape[:-1] - np.array(site_offset))):
        site2 = tuple(np.array(site1) + np.array(site_offset))
        J_on_site = J[site1] if isinstance(J, np.ndarray) else J
        delay = get_bond_delay(alpha_delay, location_dependent_delay, site1, site2, sublattice1, sublattice2)
        time_dependence = get_J_delayed(pulse_length=pulse_length, delay=delay)
        hamiltonian.add_term(name=f'J{alpha}_{site1}', strength=J_on_site, sublattice1=sublattice1, sublattice2=sublattice2, site1=site1, site2=site2,
                             time_dependence=time_dependence)


def get_bond_delay(alpha_delay, location_dependent_delay, site1, site2, sublattice1, sublattice2):
    if location_dependent_delay is not None:
        bond_center = get_x_y_of_bond_center_from_site1_site2((*site1, sublattice1), (*site2, sublattice2))
        bond_delay = location_dependent_delay(*bond_center)
    else:
        bond_delay = 0
    delay = alpha_delay + bond_delay
    return delay


def hexagonal_lattice_site_to_x_y(site):
    # site is a tuple of the form (x, y, sublattice)
    x = 1.5*(site[0] + site[1]) + site[2]
    y = (site[1]-site[0])*np.sqrt(3)/2
    return x, y


def get_x_y_of_bond_center_from_site1_site2(site1, site2):
    x1, y1 = hexagonal_lattice_site_to_x_y(site1)
    x2, y2 = hexagonal_lattice_site_to_x_y(site2)
    x, y = (x1+x2)/2, (y1+y2)/2
    return x, y

def draw_lattice(system_shape, state=None, hamiltonian:MajoranaFreeFermionHamiltonian = None, circle_radius=1 / 3, ax=None,
                 location_dependent_delay=None, color_bonds_by='xyz', add_colorbar=True):
    # draw all sites as circles with size according to the absolute value of the state and color according to the phase
    # this part draws the hamiltonian terms as lines between the sites with color according to the term name being x, y or z
    # the lines are drawn at the center of the site but are not contained in a circle of radius circle_radius which is also drawn
    arrow_to_circle_scale = 0.9
    colormap = mpl.colormaps['viridis']
    if ax is None:
        fig, ax = plt.subplots()
    xyz_to_color = {'x': 'b', 'y': 'g', 'z': 'r'}
    xyz_to_delay = {'x': 0, 'y':1/3, 'z':2/3}

    # this part draws the state by drawing arrows pointing according to the phase and with size according to the absolute value
    # the arrows are drawn at the center of the site and are contained in a circle of radius 1/2 which is also drawn

    lattice_shape = system_shape[:len(system_shape) // 2]
    if state is not None:
        state = state.reshape(lattice_shape)
        phase = np.angle(state)

    circles = []
    for site in np.ndindex(lattice_shape):
        x, y = hexagonal_lattice_site_to_x_y(site)
        circle = Circle((x, y), circle_radius, color='gray', fill=state is None)
        circles.append(circle)
        if state is not None:
            site_phase = phase[site]
            strength = np.array(np.abs(state[site])**2/np.max(np.abs(state))**2).reshape(1,1)
            ax.quiver(x, y, arrow_to_circle_scale*2*circle_radius*np.cos(site_phase), arrow_to_circle_scale*2*circle_radius*np.sin(site_phase),
                      scale=1,
                      units='xy', pivot='middle', alpha=strength, zorder=2)
    coll = matplotlib.collections.PatchCollection(circles, zorder=1, match_original=True)
    ax.add_collection(coll)

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
            bond_center = get_x_y_of_bond_center_from_site1_site2((*site1, sublattice1), (*site2, sublattice2))
            delay = xyz_to_delay[name[1]] + (location_dependent_delay(*bond_center) if location_dependent_delay else 0)
            delay = delay % 1
            if color_bonds_by == 'xyz':
                color = xyz_to_color[name[1]]
            if color_bonds_by == 'delay':
                color = colormap(delay)
            ax.plot([x1_at_circle_edge, x2_at_circle_edge], [y1_at_circle_edge, y2_at_circle_edge], color=color, linewidth=2, zorder=0)
            # ax.text(bond_center[0],bond_center[1],'{0:.2f}'.format(delay),horizontalalignment='center',verticalalignment='center',
            #         fontsize=10, fontname='Times New Roman')
    if color_bonds_by == 'delay' and add_colorbar==True:
        edit_graph(None, None, colorbar_title='Pulse Delay', colormap=colormap,
                   colorbar_args={'orientation':'horizontal', 'pad':-0.15, 'aspect':30, 'shrink':0.5}, tight=False)

    ax.axis('equal')
    plt.axis('off')

# Function to calculate jitter amount proportional to the density of points
def calculate_jitter_amount(data, bandwidth=0.5):
    kde = gaussian_kde(data, bw_method=bandwidth)
    density = kde(data)
    max_density = max(density)
    min_density = min(density)
    normalized_density = (density - min_density)/(max_density - min_density)
    return normalized_density

def get_average_distance_of_state_from_vortex(system_shape, state, vortex_center):
    lattice_shape = system_shape[:len(system_shape) // 2]
    state = state.reshape(lattice_shape)

    strengths = []
    distances = []
    for site in np.ndindex(lattice_shape):
        x, y = hexagonal_lattice_site_to_x_y(site)
        strengths.append(abs(state[site])**2)
        distances.append(np.sqrt((vortex_center[0]-x)**2 + (vortex_center[1]-y)**2))
    return np.average(distances, weights=strengths)

def draw_spectrum(energies, distance_from_vortex, ax=None, colormap='plasma'):
    # use a violin plot to plot the density of states as a function of energy
    if ax is None:
        fig, ax = plt.subplots()

    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(['$-\pi$', 0, '$\pi$'])
    ax.grid(axis='y')
    plt.ylim([-np.pi*1.05, np.pi*1.05])
    sns.violinplot(y=energies, color='gray', palette='gray', inner=None, linewidth=1.5, cut=0, bw=0.05)

    ax.collections[0].set_alpha(0.2)
    # Calculate jitter amounts based on density of points for each energy
    jitter_amounts = calculate_jitter_amount(energies, bandwidth=0.03)**(1/4)/4

    # Add jitter to x-axis positions
    jittered_x = np.array([random.uniform(-jitter_amounts[i], jitter_amounts[i]) for i,_ in enumerate(energies)])

    # Plot the points using scatter with jittered x-axis positions
    ax.scatter(jittered_x, energies, c=distance_from_vortex, cmap=colormap, s=30, zorder=4)
    ax.set_aspect(1)

    edit_graph('Density of States', 'Energy', colorbar_title='Distance from Time Vortex', colormap=colormap)


if __name__ == "__main__":
    integration_params = dict(name='vode', nsteps=2000, rtol=1e-10, atol=1e-14)
    J = np.pi/4*0.9
    pulse_length = 1/2
    vortex_location = 'plaquette'
    num_sites_x = 6
    num_sites_y = 6
    hamiltonian, location_dependent_delay, vortex_center = get_floquet_KSL_model(num_sites_x, num_sites_y, J=J, pulse_length=pulse_length, vortex_location=vortex_location)
    # unitary = hamiltonian.full_cycle_unitary_faster(integration_params, 0, 1)
    unitary = hamiltonian.full_cycle_unitary_trotterize(0, 1, 1000)
    # draw the real and imaginary parts of the unitary as subplots with colorbars for each
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(np.real(unitary))
    ax[0].set_title('real part')
    ax[1].imshow(np.imag(unitary))
    ax[1].set_title('imaginary part')
    phases, states = eig(unitary)
    energies = np.angle(phases)
    # sort the energies and phases and states according to the energies
    sort_indices = np.argsort(energies)
    energies = energies[sort_indices]
    phases = phases[sort_indices]
    states = states[:, sort_indices]

    # plot the energies
    distance_from_vortex = np.zeros_like(energies)
    for i in range(len(energies)):
        distance_from_vortex[i] = get_average_distance_of_state_from_vortex(hamiltonian.system_shape, states[:, i], vortex_center)
    draw_spectrum(energies, distance_from_vortex)
    plt.savefig(f'graphs/time_vortex/spectrum_Nx_{num_sites_x}_Ny_{num_sites_y}_J_{J:.2f}_pulse_length_{pulse_length:.2f}.pdf', bbox_inches='tight')
    plt.show()

    # draw the lattice with the delays
    draw_lattice(hamiltonian.system_shape, hamiltonian=hamiltonian, location_dependent_delay=location_dependent_delay, color_bonds_by='delay', circle_radius=0.1)
    plt.savefig(
        f'graphs/time_vortex/pulse_delays_on_the_lattice_Nx_{num_sites_x}_Ny_{num_sites_y}.pdf')
    plt.show()
    # acting with the unitary on an initial state with a localized single fermion excitation
    initial_state = np.zeros_like(states[:, 0])
    initial_site = (num_sites_x//2,num_sites_y//2-1,1)
    initial_site = (1,2,1)
    initial_site_index = np.ravel_multi_index(initial_site, hamiltonian.system_shape[:len(hamiltonian.system_shape)//2])
    initial_state[initial_site_index] = 1
    final_state = unitary @ initial_state
    # draw the initial and final states in subplots
    _, ax = plt.subplots(1, 2)
    draw_lattice(hamiltonian.system_shape, initial_state, hamiltonian, location_dependent_delay=location_dependent_delay, ax=ax[0])
    draw_lattice(hamiltonian.system_shape, final_state, hamiltonian, location_dependent_delay=location_dependent_delay, ax=ax[1])
    ax[0].set_title('initial state')
    ax[1].set_title('final state')


    for i_state, (state, energy) in enumerate(zip(states.T, energies)):
        draw_lattice(hamiltonian.system_shape, state, hamiltonian, location_dependent_delay=location_dependent_delay, color_bonds_by='delay', add_colorbar=False)
        edit_graph(None, None)
        if i_state in [0, len(states)-1]:
            plt.savefig(
                f'graphs/time_vortex/pi_mode_state_number_{i_state}_Nx_{num_sites_x}_Ny_{num_sites_y}_J_{J:.2f}_pulse_length_{pulse_length:.2f}.pdf')
        plt.title(f'energy/$\pi$ = {energy/np.pi}')
        plt.show()