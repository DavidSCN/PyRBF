""" Prints some key error measures for conservative interpolation. """

import testfunctions, rbf, basisfunctions
import numpy as np
from numpy.linalg import norm

in_mesh = np.linspace(0, 1, 300)
tf = testfunctions.Highfreq()

bf = basisfunctions.Gaussian(basisfunctions.Gaussian.shape_param_from_m(6, in_mesh))

interp = rbf.SeparatedConservative(bf, in_mesh, tf(in_mesh))

print("To 250")
out_mesh = np.linspace(0, 1, 250)
print("  Linf Error         =", norm(tf(out_mesh) - interp(out_mesh), ord = np.inf))
print("  Weighted Error     =", norm(interp.weighted_error(tf, out_mesh), ord = np.inf))
print("  Rescaled Error     =", norm(interp.rescaled_error(tf, out_mesh), ord = np.inf))
print("  Conservative Delta =", tf(in_mesh).sum() - interp(out_mesh).sum())
print()

print("To 300")
out_mesh = np.linspace(0, 1, 300)
print("  Linf Error         =", norm(tf(out_mesh) - interp(out_mesh), ord = np.inf))
print("  Weighted Error     =", norm(interp.weighted_error(tf, out_mesh), ord = np.inf))
print("  Rescaled Error     =", norm(interp.rescaled_error(tf, out_mesh), ord = np.inf))
print("  Conservative Delta =", tf(in_mesh).sum() - interp(out_mesh).sum())
print()

print("To 350")
out_mesh = np.linspace(0, 1, 350)
print("  Linf Error         =", norm(tf(out_mesh) - interp(out_mesh), ord = np.inf))
print("  Weighted Error     =", norm(interp.weighted_error(tf, out_mesh), ord = np.inf))
print("  Rescaled Error     =", norm(interp.rescaled_error(tf, out_mesh), ord = np.inf))
print("  Conservative Delta =", tf(in_mesh).sum() - interp(out_mesh).sum())
print()
