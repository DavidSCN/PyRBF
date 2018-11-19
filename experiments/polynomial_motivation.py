""" 
Plots polynomial, actual basis function and interpolant on a small mesh.
Intention is to demonstrate the effect of the polynomial.
"""

import csv
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import basisfunctions, rbf

from ipdb import set_trace

func = lambda x: 16*np.power(x, 2) + 4

in_mesh = np.linspace(0, 1, 4)
out_mesh = np.linspace(0, 1, 4)
plot_mesh = np.linspace(0, 1, 100)
in_vals = func(in_mesh)

df = pd.DataFrame(index = plot_mesh)
df.index.name = "x"

m = 2
bf = basisfunctions.Gaussian().shaped(m, in_mesh)
print("Shape Parameter =", basisfunctions.Gaussian().shape_param(m, in_mesh))

plt.plot(in_mesh, in_vals, "d")
plt.plot(plot_mesh, func(plot_mesh), "-", label = "Testfunction")
df["f"] = func(plot_mesh)

none_consistent = rbf.NoneConsistent(bf, in_mesh, in_vals, rescale = False)
plt.plot(plot_mesh, none_consistent(plot_mesh), "-", label = "No Polynomial Interpolant")
df["NoneConsistent"] = none_consistent(plot_mesh)

for i, gamma, vertex in zip(range(len(in_mesh)), none_consistent.gamma, in_mesh):
    plt.plot(plot_mesh, bf(plot_mesh-vertex) * gamma, "--")
    df["None_BF_" + str(i)] = bf(plot_mesh-vertex) * gamma
    


sep_consistent = rbf.SeparatedConsistent(bf, in_mesh, in_vals, rescale = False)
plt.plot(plot_mesh, sep_consistent(plot_mesh), "-", label = "Polynomial Interpolant")
df["SeparatedConsistent"] = sep_consistent(plot_mesh)
plt.plot(plot_mesh, sep_consistent.polynomial(plot_mesh), "-", label = "Polynomial")
df["Polynomial"] = sep_consistent.polynomial(plot_mesh)


for i, gamma, vertex in zip(range(len(in_mesh)), sep_consistent.gamma, in_mesh):
    plt.plot(plot_mesh, bf(plot_mesh-vertex) * gamma, "--")
    df["Separated_BF_" + str(i)] = bf(plot_mesh-vertex) * gamma


plt.legend()
plt.show()
print(df)

df.to_csv("polynomial_motivation.csv")

# for x in plot_mesh:
    # print(evaluate(x, none_consistent, sep_consistent))
