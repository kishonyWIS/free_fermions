import os
import matplotlib.pyplot as plt
from FloquetKSL import *
import numpy as np

# Define the system
# bonds on the hexagonal lattice are assigned a delay according to the following pattern:
# the plaquettes are colored in a checkerboard pattern 0,1,2
# the bonds are assigned a delay according to the plaquettes they connect to 0,1,2

class Site:
    def __init__(self, i_x, i_y, sublattice):
        self.i_x = i_x
        self.i_y = i_y
        self.sublattice = sublattice

    def x_y(self):
        return np.array(hexagonal_lattice_site_to_x_y([self.i_x, self.i_y, self.sublattice]))

class Bond:
    def __init__(self, i_x, i_y, direction):
        self.i_x = i_x
        self.i_y = i_y
        self.direction = direction
        self.site1 = Site(i_x, i_y, 1)
        if direction == 0:
            self.site2 = Site(i_x, i_y, 0)
        elif direction == 1:
            self.site2 = Site(i_x+1, i_y, 0)
        elif direction == 2:
            self.site2 = Site(i_x, i_y+1, 0)
        self.color = ((-i_x + i_y + direction) % 3)/3

    def x_y(self):
        return (self.site1.x_y() + self.site2.x_y())/2

class Lattice:
    def __init__(self, num_sites_x, num_sites_y):
        self.num_sites_x = num_sites_x
        self.num_sites_y = num_sites_y
        self.bonds = []
        for i_x in range(num_sites_x):
            for i_y in range(num_sites_y):
                for direction in range(3):
                    self.bonds.append(Bond(i_x, i_y, direction))

    def add_delay_vortex(self, delay):
        for bond in self.bonds:
            bond.color += delay(bond.x_y()[0], bond.x_y()[1])

    def draw_bond_colors(self):
        for bond in self.bonds:
            colormap = mpl.colormaps['viridis']
            #draw a line from bond.site1 to bond.site2 with color bond.color
            plt.plot([bond.site1.x_y()[0], bond.site2.x_y()[0]], [bond.site1.x_y()[1], bond.site2.x_y()[1]], color=colormap(bond.color), linewidth=4)

if __name__ == '__main__':
    num_sites_x = 8
    num_sites_y = 8
    lattice = Lattice(num_sites_x, num_sites_y)
    vortex_center = tuple(np.array(hexagonal_lattice_site_to_x_y((num_sites_x // 2, num_sites_y // 2, 0))) + np.array((-1, 0)))  # on plaquette
    location_dependent_delay = lambda x, y: (np.arctan2(y - vortex_center[1], x - vortex_center[0]) + np.pi/2) / (2*np.pi)

    plt.figure()
    lattice.draw_bond_colors()
    # draw the vortex center
    plt.scatter(*vortex_center, color='r')
    plt.axis('equal')
    plt.figure()
    lattice.add_delay_vortex(location_dependent_delay)
    lattice.draw_bond_colors()
    # draw the vortex center
    plt.scatter(*vortex_center, color='r')
    plt.axis('equal')
    plt.show()