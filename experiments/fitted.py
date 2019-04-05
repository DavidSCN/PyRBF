""" Compares polynomial with increasing degree on a number of test functions on a normal sized test mesh and a slighty enlarged one. """

import numpy as np, matplotlib.pyplot as plt, pandas as pd
import basisfunctions, rbf, testfunctions
from mesh import GaussChebyshev_1D
from basisfunctions import Gaussian

norm = lambda x: np.linalg.norm(x, ord = np.inf)

BF = basisfunctions.Gaussian

tfs = [testfunctions.Highfreq(), testfunctions.Lowfreq(), testfunctions.Constant(1), testfunctions.Jump()]

in_mesh = np.linspace(0, 1, 100)
test_mesh0 = np.linspace(0, 1, 20000)
test_mesh1 = np.linspace(-0.1, 1.1, 20000)

bf = BF(BF.shape_param_from_m(6, in_mesh))

cols = [str(tf) + "_" + l for l in ["InfError", "InfErrorLargerMesh"] for tf in tfs]

df = pd.DataFrame(columns = ["degree"] + cols, dtype = np.float64)
df = df.set_index("degree")
                  
for deg in range(1, 33):
    df.loc[deg] = np.NaN
    for tf in tfs:
        in_vals = tf(in_mesh)
        sepfit = rbf.SeparatedConsistentFitted(bf, in_mesh, in_vals, degree = deg)
        err0 = norm(sepfit(test_mesh0) - tf(test_mesh0))
        err1 = norm(sepfit(test_mesh1) - tf(test_mesh1))
        
        df.loc[deg][str(tf) + "_InfError"] = err0
        df.loc[deg][str(tf) + "_InfErrorLargerMesh"] = err1
        print("Degree =", deg, ", TF =", str(tf))
        print("deg1 - e0 =",  df.loc[1][str(tf) + "_InfError"] - err0)
        print("deg1 - e1 =",  df.loc[1][str(tf) + "_InfErrorLargerMesh"] - err1)




sep = rbf.SeparatedConsistent(bf, in_mesh, in_vals)
sepfit = rbf.SeparatedConsistentFitted(bf, in_mesh, in_vals, degree = 8)

print()
print("Sep Inf Error    =", norm(sep(test_mesh0) - tf(test_mesh0)))
print("Sep FitInf Error =", norm(sepfit(test_mesh0) - tf(test_mesh0)))

for c in df:
    print(c, df[c].idxmin())


# plt.plot(test_mesh1, tf(test_mesh1), label = "TF")
# plt.plot(test_mesh1, sep(test_mesh1), label = "Sep")
# plt.plot(test_mesh1, sepfit(test_mesh1), label = "SepFit")
# plt.plot(test_mesh, sep(test_mesh) - tf(test_mesh), label ="Error Sep")
# plt.plot(test_mesh, sepfit(test_mesh) - tf(test_mesh), label ="Error SepFit")
# plt.legend()
# plt.grid()
# plt.show()

df.to_csv("higher_degree_polynomial.csv")
df.plot(logy = True)
plt.show()
