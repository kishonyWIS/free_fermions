import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import seaborn as sns
from time_dependence_functions import get_g, get_B

g0 = 0.5
B1 = 0.
B0 = 3.
T = 30.
t1 = T / 4

smoothed_g = lambda t: get_g(t, g0, T, t1)
smoothed_B = lambda t: get_B(t, B0, B1, T)

ts = np.linspace(0,T,1000)
gs = []
Bs = []
for t in ts:
    gs.append(smoothed_g(t))
    Bs.append(smoothed_B(t))

with sns.axes_style("whitegrid"):
    plt.rcParams['legend.title_fontsize'] = 40
    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure()
    plt.plot(ts/T, gs, linewidth=3, label='$g$')
    plt.plot(ts/T, Bs, linewidth=3, label='$B$')
    plt.xlabel('t/T', fontsize='50', fontname='Times New Roman')#, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=40)
    l = plt.legend(prop=mpl.font_manager.FontProperties(family='Times New Roman', size=40))
    # l.set_title(title='auxiliary sites per unit cell',
    #             prop=mpl.font_manager.FontProperties(family='Times New Roman', size=18))
    plt.tight_layout()
    plt.savefig(f'graphs/g_and_B_vs_T.pdf')
    plt.show()


plt.legend()
plt.show()