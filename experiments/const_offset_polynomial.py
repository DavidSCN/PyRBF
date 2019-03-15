""" Plots RMSE for a simple function with varying y0 offsets with and without polynomial. """

import functools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import basisfunctions, rbf

func = lambda x, y0 : np.power(x,2) - x + y0


offsets = np.linspace(0, 20, 10)

in_mesh = np.linspace(0, 1, 100)
test_mesh = np.linspace(np.min(in_mesh), np.max(in_mesh), 2000)

rescale = False
m = 4
bf = basisfunctions.ThinPlateSplines()

df = pd.DataFrame()
df.index.name = "y0"

for y0 in offsets:
    f = functools.partial(func, y0 = y0)
    in_vals = f(in_mesh)
    separated = rbf.SeparatedConsistent(bf, in_mesh, in_vals, rescale)
    integrated = rbf.IntegratedConsistent(bf, in_mesh, in_vals)
    no_poly = rbf.NoneConsistent(bf, in_mesh, in_vals, rescale)
    
    s = pd.Series( data = {"RMSE_NoPoly" : no_poly.RMSE(f, test_mesh),
                           "RMSE_Integrated" : integrated.RMSE(f, test_mesh),
                           "RMSE_Separated" : separated.RMSE(f, test_mesh)})
    
    # Condition remains unchanged, input function as generally no effect on condition

    s.name = y0
    df = df.append(s)


    
print(df)
df.plot(grid = True)
plt.show()
