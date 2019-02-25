""" Displays how coefficients react to jump and boundaries. """

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import basisfunctions, testfunctions, rbf

in_mesh = np.linspace(0, 1, 20)
plot_mesh = np.linspace(0,1, 500)

# tf = testfunctions.Constant(1)
tf = testfunctions.Jump()
bf = basisfunctions.Gaussian(basisfunctions.Gaussian.shape_param_from_m(4, in_mesh))
# bf = basisfunctions.ThinPlateSplines()
interp = rbf.NoneConsistent(bf, in_mesh, tf(in_mesh))

plt.plot(in_mesh, interp.gamma, "o")
plt.plot(plot_mesh, interp(plot_mesh))
plt.plot(plot_mesh, tf(plot_mesh))
# plt.ylim(-2, 6)
plt.show()

df1 = pd.DataFrame(index = plot_mesh,
                  data = {"Interpolant": interp(plot_mesh)})

df2 = pd.DataFrame(index = in_mesh,
                   data = {"Coefficient": interp.gamma})

df = df1.join(df2, how = "outer")
df.index.name = "x"
df.to_csv("jump_coefficients.csv")
