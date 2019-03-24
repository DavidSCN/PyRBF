""" Generates data to show the effect of rescaling. Low density basisfunctions used. """

import pandas
from rbf import *
import basisfunctions, testfunctions
import matplotlib.pyplot as plt

    
bf = basisfunctions.Gaussian(shape_parameter = 9)
func = lambda x: (x-0.1)**2 + 1

one_func = lambda x: np.ones_like(x)
in_mesh = np.linspace(0, 1, 6)
plot_mesh = np.linspace(0, 1, 200)
in_vals = func(in_mesh)

interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
resc_interp = NoneConsistent(bf, in_mesh, in_vals, rescale = True)
one_interp = NoneConsistent(bf, in_mesh, one_func(in_mesh), rescale = False)

plt.plot(plot_mesh, func(plot_mesh), label = "Target $f$")
plt.plot(plot_mesh, interp(plot_mesh), "--", label = "Interpolant $S_f$")
plt.plot(plot_mesh, one_interp(plot_mesh), "--", label = "Interpolant $S_r$ of $g(x) = 1$")

plt.tight_layout()
plt.plot(plot_mesh, resc_interp(plot_mesh), label = "Rescaled Interpolant")

print("RMSE no rescale =", interp.RMSE(func, plot_mesh))
print("RMSE rescaled   =", resc_interp.RMSE(func, plot_mesh))
plt.legend()
plt.show()

plt.plot(plot_mesh, interp.error(func, plot_mesh))
plt.plot(plot_mesh, resc_interp.error(func, plot_mesh))
plt.grid()
plt.show()

df = pandas.DataFrame(data = { "Target" : func(plot_mesh),
                               "Interpolant" : interp(plot_mesh),
                               "RescaledInterpolant" : resc_interp(plot_mesh),
                               "OneInterpolant" : one_interp(plot_mesh),
                               "Error" : interp.error(func, plot_mesh),
                               "RescaledError" : resc_interp.error(func, plot_mesh)},
                      index = plot_mesh)

df.to_csv("rescaled_demo.csv", index_label = "x")
