""" Displays how coefficients react to jump and boundaries. """

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import testfunctions, rbf
from basisfunctions import *

in_mesh = np.linspace(0, 1, 21)
plot_mesh = np.linspace(0,1, 1000)

test_mesh = np.linspace(0, 1, 10000)
# exclude the discontinuity at 0.5
test_mesh_excl = np.concatenate((np.linspace(0, 0.4, 5000), np.linspace(0.6, 1, 5000)))
test_mesh_only = np.linspace(0.45, 0.55, 2000)


# tf = testfunctions.Constant(1)
tf = testfunctions.Jump()

# bf = basisfunctions.CompactPolynomialC0(4)

bfs  = [Gaussian(Gaussian.shape_param_from_m(4, in_mesh)),
        ThinPlateSplines(),
        VolumeSplines(),
        CompactPolynomialC0(CompactPolynomialC0.shape_param_from_m(4, in_mesh))]

print("Gaussian epsilon =", Gaussian.shape_param_from_m(4, in_mesh))
print("CP0 r =", CompactPolynomialC0.shape_param_from_m(4, in_mesh))

df = pd.DataFrame()


for bf in bfs:
    interp = rbf.NoneConsistent(bf, in_mesh, tf(in_mesh))

    plt.plot(in_mesh, interp.gamma, "o")
    plt.plot(plot_mesh, interp(plot_mesh))
    plt.plot(plot_mesh, tf(plot_mesh))
    # plt.ylim(-2, 6)
    plt.title(str(bf))
    plt.grid()
    # plt.show()

    df1 = pd.DataFrame(index = plot_mesh,
                       data = {"Interpolant_" + str(bf): interp(plot_mesh)})

    df2 = pd.DataFrame(index = in_mesh,
                       data = {"Coefficient_" + str(bf): interp.gamma})

    df = df.join([df1, df2], how = "outer")

    print()
    print("Basisfunction:", bf)
    error_mesh = np.linalg.norm(tf(test_mesh) - interp(test_mesh), ord=np.inf)
    error_mesh_excl = np.linalg.norm(tf(test_mesh_excl) - interp(test_mesh_excl), ord=np.inf)
    error_only = np.linalg.norm(tf(test_mesh_only) - interp(test_mesh_only), ord=np.inf)

    print("L_inf error on whole mesh =", error_mesh)
    print("L_inf error on sub mesh   =", error_mesh_excl)
    print("L_inf error on discont    =", error_only)
    print("Ratio sub / whole =", error_mesh_excl / error_mesh)
    print("Ratio whole / sub =", error_mesh / error_mesh_excl)

    


df.index.name = "x"
df.to_csv("jump_coefficients.csv")
