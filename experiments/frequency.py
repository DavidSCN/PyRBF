""" Test function frequency dependency of RBF methods. """
import testfunctions
from basisfunctions import *
from rbf import NoneConsistent
import numpy as np, matplotlib.pyplot as plt, pandas as pd


class OneFrequency(testfunctions.Testfunction):
    def __init__(self, frequency):
        self.freq = frequency

    def __str__(self):
        return "OneFrequency" + str(self.freq)

    def __call__(self, x):
        return np.sin(self.freq * x)



fs = [1, 2, 3, 10]
in_mesh = np.linspace(0, 1, 10000)
test_mesh = np.linspace(0.3, 0.7, 23000)
freqs = np.linspace(1, 3000, 300)

basisfunctions = [
    Gaussian(Gaussian.shape_param_from_m(6, in_mesh)),    
    CompactPolynomialC0(CompactThinPlateSplineC2.shape_param_from_m(40, in_mesh)),    
    ThinPlateSplines(),
    VolumeSplines()
    ]



df = pd.DataFrame()


for bf in basisfunctions:
    for f in freqs:
        print("Basisfunction:", bf, ", Frequency:", f)
        tf = lambda x: 0.1 * np.sin(f*x) + 4
        rbf = NoneConsistent(bf, in_mesh, tf(in_mesh))
        error = rbf(test_mesh) - tf(test_mesh)
        ss = pd.Series({"Frequency" : f,
                        "RBF" : str(rbf),
                        "BF" : str(bf),
                        "RMSE" : np.sqrt((error ** 2).mean()),
                        "InfError" : np.linalg.norm(error, ord=np.inf)})
        df = df.append(ss, ignore_index = True)
    

    
df = df.set_index("Frequency", drop = True)
print(df)

# for bf in basisfunctions:
    # d = df[ (df["BF"] == str(bf))
    # plt.plot(df.index.values)

df.to_csv("frequency.csv")
for name, group in df.groupby(["BF"]):
    group.to_csv("frequency_" + name + ".csv")


    
ax = plt.gca()
for name, group in df.groupby(["BF"]):
    # group.plot(ax = ax, y = "RMSE", label = "_".join(str(g) for g in name))
    group.plot(ax = ax, y = "InfError", label = name, logy = True, legend = True, grid = True)

plt.show()
    
# plt.plot(freqs, results)
# plt.legend()
# plt.grid()
# plt.show()
