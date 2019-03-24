import matplotlib.pyplot as plt, numpy as np, pandas as pd

import mesh, rbf, testfunctions, basisfunctions
from mesh import GaussChebyshev_1D


gc_points = GaussChebyshev_1D(order = 24, element_size = 1, domain_size = 2, domain_start = -1)
test_mesh = np.linspace(-1, 1, 500)

spacing = mesh.spacing(gc_points)

h_max = np.max(spacing)
print("h_max =", h_max)
m = 4

support = h_max * m
print("support =", support)

local_m = support / spacing

tf = testfunctions.Constant(1)
bf = basisfunctions.Gaussian(basisfunctions.Gaussian.shape_param_from_m(m, gc_points))
interp = rbf.NoneConsistent(bf, gc_points, tf(gc_points))

fig, ax1 = plt.subplots()
ax1.plot(gc_points, np.zeros_like(gc_points), "o")
ax1.plot(gc_points, mesh.spacing(gc_points), label = "spacing")
ax1.plot(gc_points, local_m, "o-", label = "local m")

ax2 = plt.gca().twinx()
ax2.plot(test_mesh, interp.error(tf, test_mesh))


df1 = pd.DataFrame(index = gc_points,
                  data = {"Pos" : np.zeros_like(gc_points),
                          "Spacing" : mesh.spacing(gc_points),
                          "LocalM" : local_m})

df2 = pd.DataFrame(index = test_mesh,
                   data = {"Error" : interp.error(tf, test_mesh)})

df = pd.DataFrame()
df = df.join([df1, df2], how = "outer")

df.index.name = "x"



print(df)
df.to_csv("GC_mesh_local_m_demo.csv")

ax1.legend()
plt.grid()
plt.show()

