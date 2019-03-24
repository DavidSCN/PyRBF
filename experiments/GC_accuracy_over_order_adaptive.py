""" Gauss-Chebyshev mesh for increasing polynomial degree, constant cell size. """

import itertools
import matplotlib.pyplot as plt, numpy as np, pandas as pd

import mesh, rbf, testfunctions
from mesh import GaussChebyshev_1D
from basisfunctions import Gaussian, CompactPolynomialC0

from ipdb import set_trace



def create_BFs(bf, m, in_mesh):
    BFs = []
    spaces = mesh.spacing(in_mesh)
    # spaces = np.full_like(spaces, np.max(spaces)) # same as normal
    epses = []
    for s in spaces:
        if bf == Gaussian:
            epsilon = np.sqrt(-np.log(1e-9)) / (m * s)
        else:
            epsilon = m * s
        BFs.append(bf(shape_parameter=epsilon))
        epses.append(epsilon)

    print("max h =", max(spaces), ", min h =", min(spaces), ", max eps =", max(epses), ", min eps =", min(epses))
    return BFs

test_mesh = np.linspace(0, 1, 5000)

ms = [4, 6, 8, 10, 12]
tf_const = testfunctions.Constant(1)
tf_hf = testfunctions.Highfreq()
BF = Gaussian

df = pd.DataFrame()

orders = np.arange(4, 64, 1)
element_size = 0.0625

for order, m in itertools.product(orders, ms):
    gc_points = GaussChebyshev_1D(order = order, element_size = element_size, domain_size = 1, domain_start = 0)
    print("Order =", order, ", elementSize =", element_size, "mesh size =", len(gc_points), ", m =", m)
    spacing = mesh.spacing(gc_points)
    h_max = np.max(spacing)
    support = h_max * m
    local_m = support / spacing

    epsilon = BF.shape_param_from_m(m, gc_points)
    bf = BF(epsilon)
    bfa = create_BFs(BF, m, gc_points)

    interpa_const = rbf.NoneConsistent(bfa, gc_points, tf_const(gc_points))
    interpa_hf = rbf.NoneConsistent(bfa, gc_points, tf_hf(gc_points))

    interp_const = rbf.NoneConsistent(bf, gc_points, tf_const(gc_points))
    interp_hf = rbf.NoneConsistent(bf, gc_points, tf_hf(gc_points))
    
    ss = pd.Series(data = {"BF" : BF.__name__,
                           "ElementSize" : element_size,
                           "epsilon" : epsilon,
                           "AdaptiveInfError_Constant" : np.linalg.norm(interpa_const.error(tf_const, test_mesh), ord = np.inf),
                           "AdaptiveRMSE_Constant" : interpa_const.RMSE(tf_const, test_mesh),
                           "AdaptiveInfError_Highfreq" : np.linalg.norm(interpa_hf.error(tf_hf, test_mesh), ord = np.inf),
                           "AdaptiveRMSE_Highfreq" : interpa_hf.RMSE(tf_hf, test_mesh),
                           "NonAdaptiveInfError_Constant" : np.linalg.norm(interp_const.error(tf_const, test_mesh), ord = np.inf),
                           "NonAdaptiveRMSE_Constant" : interp_const.RMSE(tf_const, test_mesh),
                           "NonAdaptiveInfError_Highfreq" : np.linalg.norm(interp_hf.error(tf_hf, test_mesh), ord = np.inf),
                           "NonAdaptiveRMSE_Highfreq" : interp_hf.RMSE(tf_hf, test_mesh),
                           "AdaptiveCondition" : interpa_const.condC,
                           "NonAdaptiveCondition" : interpa_const.condC,
                           "m" : m,
                           "m_max" : np.max(local_m)})
    ss.name = order

    df = df.append(ss)
    

df.index.name = "Order"
print(df)

for name, group in df.groupby("m"):
    group.to_csv("GC_GS_adaptive_accuracy_over_order_m_" + str(name) + ".csv")

fig, ax1 = plt.subplots()
df.plot(ax = ax1, logy = True, y = ["AdaptiveInfError_Constant", "NonAdaptiveInfError_Constant"])
ax1.set_title("Adaptive m = " + str(m))
ax1.set_xlabel("Order")
ax1.legend()

# ax2 = plt.gca().twinx()
# ax2.plot(df.index, df["epsilon"], "r-")
# ax2.plot(df.index, df["m_max"], "b-")
# ax2.legend(loc = "upper left")

plt.grid()
plt.show()


