""" Plots RMSE over condition over a range of shape parameters"""

import matplotlib.pyplot as plt, numpy as np, pandas as pd
import testfunctions, basisfunctions
from rbf import *

tf = testfunctions.Highfreq()
in_mesh = np.linspace(0, 1, 192)
in_vals = tf(in_mesh)
test_mesh = np.linspace(0, 1, 2000)

BF = basisfunctions.Gaussian

ms = np.linspace(1, 16, 50)

cols = ["None", "Sep", "Int", "NoneResc", "SepResc", "SepDeg3", "SepRescDeg3"]
df = pd.DataFrame(columns = cols + ["m"])

for m in ms:
    print("Working on m =", m)
    bf = BF(BF.shape_param_from_m(m, in_mesh))
    rbfs = [NoneConsistent(bf, in_mesh, in_vals, False),
            SeparatedConsistent(bf, in_mesh, in_vals, False),
            IntegratedConsistent(bf, in_mesh, in_vals),
            NoneConsistent(bf, in_mesh, in_vals, True),
            SeparatedConsistent(bf, in_mesh, in_vals, True),
            SeparatedConsistentFitted(bf, in_mesh, in_vals, False, degree=3),
            SeparatedConsistentFitted(bf, in_mesh, in_vals, True, degree=3)

    ]

    for c, r in zip(cols, rbfs):
        rmse = r.RMSE(tf, test_mesh)
        df.loc[rmse] = np.NaN
        df.loc[rmse][c] = r.condC
        df.loc[rmse]["m"] = m



df.index.name = "RMSE"
df = df.sort_index()
df.to_csv("rmse_over_cond.csv")
