from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np

colormap = cm.viridis
res = 1000
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

r = np.linspace(0, 1, 100)
theta = np.linspace(0, 2 * np.pi, 100)
R, Theta = np.meshgrid(r, theta)
X, Y = R * np.cos(Theta), R * np.sin(Theta)

for z in np.arange(4)*0.5:
    Z = np.ones_like(X)*z
    delay = ((-np.arctan2(Y, X)+Z) / (2*np.pi)) % 1

    ax.plot_surface(X, Y, Z, facecolors=colormap(delay), linewidth=0, antialiased=False)

# add the winding surface
# r = np.linspace(0, 1, 100)
# theta = np.linspace(0, 1, 10000)
# R, Theta = np.meshgrid(r, theta)
# X, Y = R * np.cos(Theta), R * np.sin(Theta)
# Z = Theta
# ax.plot_surface(X, Y, Z, color='grey', alpha=0.5, linewidth=0, antialiased=False)


# set elevation and azimuth
ax.view_init(26, 18-90)
# remove grid
ax.grid(False)
# remove ticks
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
# set labels
ax.set_xlabel('$x$', labelpad=-10, fontsize=30, fontname='Times New Roman')
ax.set_ylabel('$y$', labelpad=-10, fontsize=30, fontname='Times New Roman')
ax.set_zlabel('$t$', labelpad=-10, fontsize=30, fontname='Times New Roman')
ax.zaxis._axinfo['juggled'] = (1,2,2)

cbar = plt.colorbar(cm.ScalarMappable(cmap=colormap), ax=ax, shrink=0.6, ticks= [0,1], pad=-0.04)
cbar.ax.set_yticklabels([0, '$T$'], fontname='Times New Roman', fontsize='22')
cbar.ax.tick_params(labelsize=22)
plt.tight_layout()

plt.savefig('graphs/time_vortex/time_vortex_spiral.pdf')
plt.show()