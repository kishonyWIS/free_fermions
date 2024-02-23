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
from scipy.optimize import minimize

def rescale_linewidths(ax):
    plt.tight_layout()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width = abs(xlim[1] - xlim[0])
    height = abs(ylim[1] - ylim[0])
    diagonal_length = np.sqrt(width**2 + height**2)
    trans = ax.transData.transform
    dpi = ax.get_figure().dpi
    for line in ax.lines:
        # linewidth = line.get_linewidth()
        linewidth = ((trans((1, line.get_linewidth())) - trans((0, 0))) * 72./dpi)[1]
        # line.set_linewidth(linewidth * ax.transData.transform((1, 0))[0]/100)
        line.set_linewidth(linewidth)
    plt.tight_layout()

def edit_graph(xlabel, ylabel, ax=None, legend_title=None, colorbar_title=None, colormap=None, colorbar_args={}, tight=True, ylabelpad=None, colorbar_xticklabels=None, colorbar_yticklabels=None):
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
            cbar = plt.colorbar(cmap,ax=ax,ticks=[0, 1/3, 2/3, 1],**colorbar_args)
        else:
            cbar = plt.colorbar(ax=ax,ticks=[0, 1/3, 2/3, 1],**colorbar_args)
        if colorbar_xticklabels:
            cbar.ax.set_xticklabels(colorbar_xticklabels, fontname='Times New Roman')
        if colorbar_yticklabels:
            cbar.ax.set_yticklabels(colorbar_yticklabels, fontname='Times New Roman')
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
    vortex_center = None
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

def draw_lattice(system_shape, shading=None, hamiltonian:MajoranaFreeFermionHamiltonian = None, circle_radius=1 / 3, ax=None,
                 location_dependent_delay=None, color_bonds_by='xyz', add_colorbar=True):
    # draw all sites as circles with size according to the absolute value of the state and color according to the phase
    # this part draws the hamiltonian terms as lines between the sites with color according to the term name being x, y or z
    # the lines are drawn at the center of the site but are not contained in a circle of radius circle_radius which is also drawn
    colormap = mpl.colormaps['viridis']
    if ax is None:
        fig, ax = plt.subplots()
    # xyz_to_color = {'x': 'b', 'y': 'g', 'z': 'r'}
    xyz_to_delay = {'x': 0, 'y':1/3, 'z':2/3}

    # this part draws the state by drawing arrows pointing according to the phase and with size according to the absolute value
    # the arrows are drawn at the center of the site and are contained in a circle of radius 1/2 which is also drawn

    lattice_shape = system_shape[:len(system_shape) // 2]
    if shading is not None:
        shading = shading.reshape(lattice_shape)
        phase = np.angle(shading)

    circles = []
    colors = []
    circles_colormap = matplotlib.colormaps['gray'].reversed()
    max_strength = np.max(np.abs(shading)) if shading is not None else 1
    for site in np.ndindex(lattice_shape):
        x, y = hexagonal_lattice_site_to_x_y(site)
        if shading is not None:
            strength = shading[site] / max_strength
        else:
            strength = 1
        # draw a circle at the center of the site with a radius of circle_radius and color in grayscale according to the strength
        circle = Circle((x, y), circle_radius, color=circles_colormap(strength), fill=True)  # state is None)
        circles.append(circle)
        colors.append(strength)
        circle = Circle((x, y), circle_radius, color='gray', fill=False)#state is None)
        circles.append(circle)
    coll = matplotlib.collections.PatchCollection(circles, cmap=circles_colormap, zorder=1, match_original=True)
    ax.add_collection(coll)
    if shading is not None:
        cbar = plt.colorbar(coll, orientation='horizontal', aspect=30, shrink=0.5, ticks=[0, 0.5, 1])
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.set_xticklabels(list(map('{0:.2f}'.format,[0,max_strength/2,max_strength])), fontname='Times New Roman')


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
                color = colormap(xyz_to_delay[name[1]])#xyz_to_color[name[1]]
            if color_bonds_by == 'delay':
                color = colormap(delay)
            plt.plot([x1_at_circle_edge, x2_at_circle_edge], [y1_at_circle_edge, y2_at_circle_edge], color=color, linewidth=0.08, zorder=0)
    if color_bonds_by == 'delay' and add_colorbar==True:
        edit_graph(None, None, ax=ax, colorbar_title='Pulse Delay', colormap=colormap,
                   colorbar_args={'orientation':'horizontal', 'pad':-0.15, 'aspect':30, 'shrink':0.5}, colorbar_xticklabels=['$0$', '$T/3$', '2T/3', '$T$'], tight=False)

    rescale_linewidths(ax)
    ax.axis('equal')
    if ax is not None:
        ax.axis('off')
    else:
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

    edit_graph('Density of States', 'Energy', colorbar_title='Distance from Time Vortex', ax=ax, colormap=colormap)


def get_vortex_and_edge_state(energies, states, vortex_center, system_shape):
    # sort the states by the absolute value of their energy and take the two highest
    sort_indices = np.argsort(np.abs(energies))
    state1, state2 = states[:, sort_indices[-1]], states[:, sort_indices[-2]]
    # find the normalized linear combination of the two states that has the smallest
    # get_average_distance_of_state_from_vortex
    minimize_result = minimize(lambda x: get_average_distance_of_state_from_vortex(system_shape, np.cos(x[0]/2)*state1+np.sin(x[0]/2)*np.exp( 1j*x[1])*state2, vortex_center), x0=[0.5, 0.5])
    theta = minimize_result.x[0]
    phi = minimize_result.x[1]
    vortex_state = np.cos(theta/2)*state1+np.sin(theta/2)*np.exp(1j*phi)*state2
    # find the state that is orthogonal to the vortex state and has the highest energy
    edge_state = np.cos((np.pi-theta)/2)*state1+np.sin((np.pi-theta)/2)*np.exp(1j*(phi+np.pi))*state2
    states[:, sort_indices[-1]], states[:, sort_indices[-2]] = vortex_state, edge_state
    return states


if __name__ == "__main__":
    J = np.pi/4*0.9
    pulse_length = 1/2
    vortex_location = 'plaquette'
    num_sites_x = 6
    num_sites_y = 6
    hamiltonian, location_dependent_delay, vortex_center = get_floquet_KSL_model(num_sites_x, num_sites_y, J=J, pulse_length=pulse_length, vortex_location=vortex_location)
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
    states = get_vortex_and_edge_state(energies, states, vortex_center, hamiltonian.system_shape)
    for i in range(len(energies)):
        distance_from_vortex[i] = get_average_distance_of_state_from_vortex(hamiltonian.system_shape, states[:, i], vortex_center)
    draw_spectrum(energies, distance_from_vortex)
    plt.savefig(f'graphs/time_vortex/spectrum_Nx_{num_sites_x}_Ny_{num_sites_y}_J_{J:.2f}_pulse_length_{pulse_length:.2f}.pdf', bbox_inches='tight')
    plt.show()

    # draw the lattice with the xyz
    draw_lattice(hamiltonian.system_shape, hamiltonian=hamiltonian, location_dependent_delay=None, color_bonds_by='delay', circle_radius=0.1)
    plt.savefig(
        f'graphs/time_vortex/xyz_on_the_lattice_Nx_{num_sites_x}_Ny_{num_sites_y}.pdf')
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
    draw_lattice(hamiltonian.system_shape, np.abs(initial_state)**2, hamiltonian, location_dependent_delay=location_dependent_delay, ax=ax[0])
    draw_lattice(hamiltonian.system_shape, np.abs(final_state)**2, hamiltonian, location_dependent_delay=location_dependent_delay, ax=ax[1])
    ax[0].set_title('initial state')
    ax[1].set_title('final state')


    for i_state, (state, energy) in enumerate(zip(states.T, energies)):
        draw_lattice(hamiltonian.system_shape, np.abs(state)**2, hamiltonian, location_dependent_delay=location_dependent_delay, color_bonds_by='delay', add_colorbar=False)
        edit_graph(None, None)
        if i_state in [0, len(states)-1]:
            plt.savefig(
                f'graphs/time_vortex/pi_mode_state_number_{i_state}_Nx_{num_sites_x}_Ny_{num_sites_y}_J_{J:.2f}_pulse_length_{pulse_length:.2f}.pdf')
        plt.title(f'energy/$\pi$ = {energy/np.pi}')
        plt.show()