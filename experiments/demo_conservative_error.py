""" Demo for conservative error metrics. """

import csv
import numpy as np, matplotlib.pyplot as plt, pandas as pd
import mesh, basisfunctions, rbf, testfunctions

testfunction = testfunctions.Highfreq() 

in_mesh = mesh.GaussChebyshev_1D(order = 12, element_size = 0.25, domain_size = 1, domain_start = 0)

# in_mesh = np.linspace(0, 1, 48)
out_mesh = np.linspace(np.min(in_mesh), np.max(in_mesh), 20) 

# in_mesh = np.geomspace(1, 100, 90)
# in_mesh = in_mesh / 100
# out_mesh = np.linspace(0, 1, 80)
    
in_vals = testfunction(in_mesh)

m = 6
bf = basisfunctions.Gaussian(basisfunctions.Gaussian.shape_param_from_m(m, in_mesh))

interp = rbf.SeparatedConservative(bf, in_mesh, in_vals)
one_interp = rbf.NoneConservative(bf, in_mesh, np.ones_like(in_vals), rescale = False)

out_vals = interp(out_mesh)
rescaled = out_vals / one_interp(out_mesh)

print(out_vals)
print("Conservativness Delta of Interp   = ", np.sum(in_vals) - np.sum(out_vals))
print("Conservativness Delta of Rescaled = ", np.sum(in_vals) - np.sum(rescaled))
print("len(in_mesh) =", len(in_mesh), "len(out_mesh) =", len(out_mesh))

plt.plot(in_mesh, in_vals, "d-", label = "f")
plt.plot(out_mesh, out_vals, "x-", label = "s")
plt.plot(out_mesh, interp.error(testfunction, out_mesh), label = "error")
plt.plot(out_mesh, interp.rescaled_error(testfunction, out_mesh), label = "rescaled error")
plt.plot(out_mesh, interp.weighted_error(testfunction, out_mesh), label = "weighted error")
plt.plot(out_mesh, one_interp(out_mesh), "x-", label = "rescaled interpolant")

plt.plot(out_mesh, rescaled, "x-", label = "rescaled")
plt.plot(out_mesh, np.ones_like(out_mesh), "--", color="black")

fieldnames = ["in_mesh", "out_mesh", "f", "s",
              "one_interp", "rescaled_interp", "error", "rescaled_error", "weighted_error"]

csv = csv.DictWriter(open("demo_conservative_error.csv", "w"),
                     fieldnames = fieldnames,
                     restval = "nan")

csv.writeheader()

for i, v in zip(in_mesh, in_vals):
    csv.writerow({"in_mesh" : i, "f" : v})

for coord, s, one, r, e, re, we in zip(out_mesh,
                                       out_vals,
                                       one_interp(out_mesh),
                                       rescaled,
                                       interp.error(testfunction, out_mesh),
                                       interp.rescaled_error(testfunction, out_mesh),
                                       interp.weighted_error(testfunction, out_mesh)):
    
    csv.writerow({"out_mesh" : coord,
                  "s" : s,
                  "one_interp" : one,
                  "rescaled_interp" : r,
                  "error" : e,
                  "rescaled_error" : re,
                  "weighted_error" : we})
    



# plt.ylim([-2, np.max(in_vals) + 1])
plt.grid()
plt.legend()
plt.show()
