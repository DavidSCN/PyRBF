""" Displays how coefficients react to jump and boundaries. """

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import testfunctions, rbf
from basisfunctions import *

in_mesh = np.linspace(0, 1, 21)
plot_mesh = np.linspace(0,1, 1000)

# tf = testfunctions.Constant(1)
tf = testfunctions.Jump()

# bf = basisfunctions.CompactPolynomialC0(4)

bfs  = [Gaussian(Gaussian.shape_param_from_m(4, in_mesh)),
        ThinPlateSplines(),
        VolumeSplines(),
        CompactPolynomialC0(4 * Basisfunction.h_max(in_mesh))]

print("CP0 support radius =", 4 * Basisfunction.h_max(in_mesh))

df = pd.DataFrame()


for bf in bfs:
    interp = rbf.NoneConsistent(bf, in_mesh, tf(in_mesh))

    plt.plot(in_mesh, interp.gamma, "o")
    plt.plot(plot_mesh, interp(plot_mesh))
    plt.plot(plot_mesh, tf(plot_mesh))
    # plt.ylim(-2, 6)
    plt.title(str(bf))
    plt.grid()
    plt.show()

    df1 = pd.DataFrame(index = plot_mesh,
                       data = {"Interpolant_" + str(bf): interp(plot_mesh)})

    df2 = pd.DataFrame(index = in_mesh,
                       data = {"Coefficient_" + str(bf): interp.gamma})

    df = df.join([df1, df2], how = "outer")
    


df.index.name = "x"
df.to_csv("jump_coefficients.csv")
