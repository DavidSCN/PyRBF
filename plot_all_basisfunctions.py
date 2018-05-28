import matplotlib.pyplot as plt
from RBF.basisfunctions import *
from plot_helper import *

set_save_fig_params(rows = 1.3, cols = 1)

plot_mesh = np.linspace(-1, 1, 1000)
in_mesh = np.linspace(-1, 1, 10)

ax = plt.subplot(3, 2, 1)
ax.plot(plot_mesh, Gaussian()(plot_mesh, 8), label = "s = 8" )
ax.plot(plot_mesh, Gaussian()(plot_mesh, 4), label = "s = 6" )
ax.plot(plot_mesh, Gaussian()(plot_mesh, 2), label = "s = 4" )
ax.set_xticks([])
ax.legend(loc = "center right")
ax.set_title("Gaussian")

ax = plt.subplot(3, 2, 2)
ax.plot(plot_mesh, VolumeSplines()(plot_mesh), label = "s = 2")
ax.set_xticks([])
ax.set_title("Volume Splines")

ax = plt.subplot(3, 2, 3)
ax.plot(plot_mesh, ThinPlateSplines()(plot_mesh))
ax.set_xticks([])
ax.set_title("Thin Plate Splines")

ax = plt.subplot(3, 2, 4)
ax.plot(plot_mesh, CompactThinPlateSplineC2()(plot_mesh, 1), label = "r = 1")
ax.plot(plot_mesh, CompactThinPlateSplineC2()(plot_mesh, 0.8), label = "r = 0.8")
ax.plot(plot_mesh, CompactThinPlateSplineC2()(plot_mesh, 0.5), label = "r = 0.5")
ax.set_xticks([])
ax.legend(loc = "center right")
ax.set_title("Compact Thin Plate Spline C2")

ax = plt.subplot(3, 2, 5)
ax.plot(plot_mesh, MultiQuadrics()(plot_mesh, 3), label = "s = 3")
ax.plot(plot_mesh, MultiQuadrics()(plot_mesh, 2), label = "s = 2")
ax.plot(plot_mesh, MultiQuadrics()(plot_mesh, 1), label = "s = 1")
ax.legend(loc = "center right")
ax.set_title("Multi Quadrics")

ax = plt.subplot(3, 2, 6)
ax.plot(plot_mesh, InverseMultiQuadrics()(plot_mesh, 3), label = "s = 3")
ax.plot(plot_mesh, InverseMultiQuadrics()(plot_mesh, 2), label = "s = 2")
ax.plot(plot_mesh, InverseMultiQuadrics()(plot_mesh, 1), label = "s = 1")
ax.legend(loc = "center right")
ax.set_title("Inverse Multi Quadrics")


plt.tight_layout()


# plt.show()
plt.savefig("basisfunctions.pdf")
