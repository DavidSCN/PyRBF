""" Compares polynomial with non-polynomial reaction on a constant shift"""

import itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import testfunctions, rbf
from basisfunctions import *

in_mesh = np.linspace(0, 1, 1000)
test_mesh = np.linspace(0, 1, 10000)
shifts = np.linspace(-100, 100, 100)

tf = testfunctions.Highfreq()
# tf = testfunctions.Constant(3)

bfs = [ VolumeSplines(),
        ThinPlateSplines(),
        Gaussian(Gaussian.shape_param_from_m(6, in_mesh)),
        CompactPolynomialC0(CompactPolynomialC0.shape_param_from_m(6, in_mesh))]

df = pd.DataFrame()

for s in shifts:
    in_vals = tf(in_mesh) + s
    test_vals = tf(test_mesh) + s

    ss = pd.Series()
    for bf in bfs:
        print("Shift =", s, "BF =", bf)
        
        sep = rbf.SeparatedConsistent(bf, in_mesh, in_vals)
        none = rbf.NoneConsistent(bf, in_mesh, in_vals)

        ss = ss.append(pd.Series(data = {
            "InfError_Sep_" + str(bf)  : np.linalg.norm(sep(test_mesh) - test_vals, ord=np.inf),
            "InfError_None_" + str(bf) : np.linalg.norm(none(test_mesh) - test_vals, ord=np.inf)        
        }))

    ss.name = s
    df = df.append(ss)


df.index.name = "Shift"

df.plot()
# for col in df:
    # plt.plot(df.index, df[])

plt.legend()
plt.show()

for bf in bfs:
    df["Delta_" + str(bf)] = df["InfError_None_" + str(bf)] - df["InfError_Sep_" + str(bf)]
    plt.plot(df.index, df["Delta_" + str(bf)], label = str(bf))
print(df)

df.to_csv("polynomial_linear_shift.csv")

plt.legend()
plt.grid()
plt.show()


    
# plt.plot(df.index, df["InfError_Sep_VolumeSplines"], label = "Sep VS")
# plt.plot(df.index, df["InfError_None_VolumeSplines"], label = "None VS")
# plt.plot(df.index, df["InfError_Sep_Gaussian"], label = "Sep GS")
# plt.plot(df.index, df["InfError_None_Gaussian"], label = "None GS")
# plt.plot(df.index, df["InfError_Sep_ThinPlateSplines"], label = "Sep TPS")
# plt.plot(df.index, df["InfError_None_ThinPlateSplines"], label = "None TPS")
# plt.plot(df.index, df["InfError_Sep_CompactPolynomialC0"], label = "Sep CP0")
# plt.plot(df.index, df["InfError_None_CompactPolynomialC0"], label = "None CP0")
