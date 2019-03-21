""" Gauss-Chebyshev mesh for increasing polynomial degree, constant cell size. """

import itertools
import matplotlib.pyplot as plt, numpy as np, pandas as pd

import mesh, rbf, testfunctions
from mesh import GaussChebyshev_1D
from basisfunctions import Gaussian, CompactPolynomialC0


test_mesh = np.linspace(0, 1, 10000)

tf_const = testfunctions.Constant(1)
tf_hf = testfunctions.Highfreq()

ms = [8, 6, 4]
BF = Gaussian

# ms = [10, 40, 80]
# BF = CompactPolynomialC0

df = pd.DataFrame()

orders = np.arange(4, 56, 1)
element_size = 0.0625

for order, m in itertools.product(orders, ms):
    gc_points = GaussChebyshev_1D(order = order, element_size = element_size, domain_size = 1, domain_start = 0)
    print("Order =", order, ", elementSize =", element_size, "mesh size =", len(gc_points), ", m =", m)
    spacing = mesh.spacing(gc_points)
    h_max = BF.h_max(gc_points)
    support = h_max * m
    local_m = support / spacing

    epsilon = BF.shape_param_from_m(m, gc_points)
    bf = BF(epsilon)
    interp_const = rbf.NoneConsistent(bf, gc_points, tf_const(gc_points))
    interp_hf = rbf.NoneConsistent(bf, gc_points, tf_hf(gc_points))
    
    ss = pd.Series(data = {"BF" : str(bf),
                           "ElementSize" : element_size,
                           "epsilon" : epsilon,
                           "InfError_Constant" : np.linalg.norm(interp_const.error(tf_const, test_mesh), ord = np.inf),
                           "RMSE_Constant" : interp_const.RMSE(tf_const, test_mesh),
                           "InfError_Highfreq" : np.linalg.norm(interp_hf.error(tf_hf, test_mesh), ord = np.inf),
                           "RMSE_Highfreq" : interp_hf.RMSE(tf_hf, test_mesh),
                           "m" : m,
                           "m_max" : np.max(local_m)})
    ss.name = order

    df = df.append(ss)
    

df.index.name = "Order"
print(df)

for name, group in df.groupby("m"):
    group.to_csv("GC_GS_accuracy_over_order_m_" + str(name) + ".csv")

fig, ax1 = plt.subplots()
ax1.semilogy(df.index, df["InfError_Constant"], label = "InfError_const")
ax1.semilogy(df.index, df["RMSE_Constant"], label = "RMSE_const")
ax1.semilogy(df.index, df["InfError_Highfreq"], label = "InfError_hf")
ax1.semilogy(df.index, df["RMSE_Highfreq"], label = "RMSE_hf")
ax1.legend()

ax2 = plt.gca().twinx()
ax2.plot(df.index, df["epsilon"], "r-")
ax2.plot(df.index, df["m_max"], "b-")

plt.grid()
plt.show()


