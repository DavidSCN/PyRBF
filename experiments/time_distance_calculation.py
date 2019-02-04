""" Compares different methods to find the minimal distance between all pairs. """

import numpy as np
import matplotlib.pyplot as plt
from timeit import timeit

def h10(mesh):
    """ Uses distances != 0 to filter zeros """
    h_max = 0
    if mesh.ndim == 1:
        mesh = mesh[:, np.newaxis]
        
    for i in mesh:
        distances = np.linalg.norm(mesh - i, axis=1)
        h_max = np.max([ np.min(distances[distances != 0]), h_max])

    return h_max

def h11(mesh):
    """ Uses np.nonzero to filter zeros. """
    h_max = 0
    if mesh.ndim == 1:
        mesh = mesh[:, np.newaxis]
        
    for i in mesh:
        distances = np.linalg.norm(mesh - i, axis=1)
        h_max = np.max([ np.min(distances[np.nonzero(distances)]), h_max])

    return h_max

def h12(mesh):
    """ Masked array, no copy. """
    h_max = 0
    if mesh.ndim == 1:
        mesh = mesh[:, np.newaxis]
        
    for i in mesh:
        distances = np.linalg.norm(mesh - i, axis=1)
        h_max = np.max([np.ma.masked_equal(distances, 0.0, copy = False).min(), h_max])

    return h_max

def h13(mesh):
    """ Masked array, no copy. """
    h_max = 0
    if mesh.ndim == 1:
        mesh = mesh[:, np.newaxis]
        
    for i in mesh:
        distances = np.linalg.norm(mesh - i, axis=1)
        h_max = np.max([np.ma.masked_equal(distances, 0.0, copy = True).min(), h_max])

    return h_max

from scipy.spatial import distance_matrix

def h2(mesh):
    if mesh.ndim == 1:
        mesh = mesh[:, np.newaxis]
        
    dm = distance_matrix(mesh, mesh)
    h_max = np.min(dm[np.nonzero(dm)])
    return h_max

def h3(mesh):
    if mesh.ndim == 1:
        mesh = mesh[:, np.newaxis]
        
    dm = distance_matrix(mesh, mesh)
    ma = np.ma.masked_equal(dm, 0.0, copy=False)
    h_max = ma.min()
    return h_max


reps = 10
size = 15000
mesh_sizes = np.linspace(1000, 15000, 30, dtype = int)

mesh = np.random.rand(1000, 1)
print("h10 =", h10(mesh))
print("h11 =", h11(mesh))
print("h12 =", h12(mesh))
print("h13 =", h13(mesh))
print("h2 =", h2(mesh))
print("h3 =", h3(mesh))

t10s = []
t11s = []
t12s = []
t13s = []
t2s = []
t3s = []

for s in mesh_sizes:
    print("Size =", s)
    mesh = np.geomspace(1, 10, s)

    t10s.append(timeit('h10(mesh)', setup='mesh = np.linspace(0, 1, s)', number = reps, globals = globals()))
    print("Time h10 =", t10s[-1])

    t11s.append(timeit('h11(mesh)', setup='mesh = np.linspace(0, 1, s)', number = reps, globals = globals()))
    print("Time h11 =", t11s[-1])

    t13s.append(timeit('h13(mesh)', setup='mesh = np.linspace(0, 1, s)', number = reps, globals = globals()))
    print("Time h13 =", t13s[-1])

    t12s.append(timeit('h12(mesh)', setup='mesh = np.linspace(0, 1, s)', number = reps, globals = globals()))
    print("Time h12 =", t12s[-1])

    t2s.append(timeit('h2(mesh)', setup='mesh = np.linspace(0, 1, s)', number = reps, globals = globals()))
    print("Time h2 =", t2s[-1])

    t3s.append(timeit('h3(mesh)', setup='mesh = np.linspace(0, 1, s)', number = reps, globals = globals()))
    print("Time h3 =", t3s[-1])

    print()

plt.plot(mesh_sizes, t10s, label = "distance != 0")
plt.plot(mesh_sizes, t11s, label = "np.nonzero")
plt.plot(mesh_sizes, t12s, label = "mask, nocopy")
plt.plot(mesh_sizes, t13s, label = "mask, copy")
plt.plot(mesh_sizes, t2s, label = "dm")
plt.plot(mesh_sizes, t3s, label = "dm, mask")
plt.xlabel("Mesh Size")
plt.ylabel("Time [s]")
plt.legend()
plt.show()
