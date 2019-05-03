import numpy as np, pandas as pd
from numpy.linalg import norm
import matplotlib.pyplot as plt
import testfunctions, rbf
from basisfunctions import *

np.random.seed(seed=1)

tf = testfunctions.Highfreq()
# BF = Gaussian
# BF = basisfunctions.CompactPolynomialC0

in_mesh = np.linspace(0, 1, 300)
in_mesh = in_mesh + 1/len(in_mesh) * 0.3 * (np.random.rand(len(in_mesh)) - 0.5)
in_vals = tf(in_mesh)
out_mesh_sizes = np.arange(200, 400, 1)

BFs = [(Gaussian, 6), (Gaussian, 8),
       (CompactPolynomialC0, 40), (CompactPolynomialC0, 80),
       (VolumeSplines, 0), (ThinPlateSplines, 0)]

res = []

for BF, m in BFs:
    # bf = BF(BF.shape_param_from_m(m, in_mesh))
    # print(str(bf), m)
    print(str(BF), m)
    for oms in out_mesh_sizes:
        out_mesh = np.linspace(0, 1, oms)
        bf = BF(BF.shape_param_from_m(m, out_mesh))              

        # out_mesh = out_mesh + 1/len(out_mesh) * 0.3 * (np.random.rand(len(out_mesh)) - 0.5)

        interp = rbf.SeparatedConservative(bf, in_mesh, in_vals)

        out_vals = interp(out_mesh)

        res.append({
            "BF" : str(bf),
            "m" : m,
            "OutMeshSize" : oms,
            "ConservativeDelta" : np.sum(out_vals) - np.sum(in_vals),
            "InfError" : norm(out_vals - tf(out_mesh), ord = np.inf),
            "RescaledError" : norm(interp.rescaled_error(tf, out_mesh), ord = np.inf),
            "WeightedError" : norm(interp.weighted_error(tf, out_mesh), ord = np.inf)

        })


df = pd.DataFrame(res)
df = df.set_index(["OutMeshSize"])

for i, (name, group) in enumerate(df.groupby(["BF", "m"])):
    label = "_".join(str(n) for n in name)
    # numplots = len(df.index.unique())
    # ax = plt.subplot(numplots // 2, 2, i+1)
    # group.plot(ax = ax, legend = False, x = "OutMeshSize", title = label)
    group.to_csv("conservative_meshsize_" + label + ".csv")
    

plt.show()
