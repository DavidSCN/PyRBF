# Accuracy over increasing values of m, with and without rescaling

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import rbf, testfunctions
from basisfunctions import *

in_mesh = np.linspace(0, 1, 1000)
test_mesh = np.linspace(0.1, 0.9, 5000) # evtl. padding wegen boundaries
tf = testfunctions.Lowfreq()
in_vals = tf(in_mesh)


tests = (
    {"label" : "GS NR",
     "ms" : np.linspace(2, 30, 100),
     "rbf" : rbf.NoneConsistent,
     "bf" : Gaussian},
    {"label" : "CP0 NR",
     "ms" : np.linspace(2, 100, 150),
     "rbf" : rbf.NoneConsistent,
     "bf" : CompactPolynomialC0},
    {"label" : "CTPS NR",
     "ms" : np.linspace(2, 100, 150),
     "rbf" : rbf.NoneConsistent,
     "bf" : CompactThinPlateSplineC2})


df = pd.DataFrame()
    
for t in tests:
    print("Working on", t["label"])
    for m in t["ms"]:
        bf = t["bf"](t["bf"].shape_param_from_m(m, in_mesh))
        rbf = t["rbf"](bf, in_mesh, in_vals, rescale = False)
        error = rbf(test_mesh) - tf(test_mesh)
        ss = pd.Series({"m" : m,
                        "RBF" : str(rbf),
                        "BF" : str(bf),
                        "RMSE" : np.sqrt((error ** 2).mean()),
                        "InfError" : np.linalg.norm(error, ord=np.inf)})
        
        df = df.append(ss, ignore_index = True)

df = df.set_index("m")
print(df)

fig, ax = plt.subplots()

for name, group in df.groupby(["BF"]):
    # import ipdb; ipdb.set_trace()
    plt.plot(group.index.values, group["InfError"].values, label = name + "_inf")
    # plt.plot(group.index.values, group["RMSE"].values, label = name + "_RMSE")


plt.yscale("log")
plt.grid()
plt.legend()
plt.show()

