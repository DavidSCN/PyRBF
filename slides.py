from rbf_qr import *
from rbf import *
import basisfunctions
import mesh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from timeit import default_timer as timer
import math
from tqdm import tqdm
from sympy.matrices import *

def intro_slide():
    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    mesh = np.array(np.meshgrid(X, Y))
    shape = 0.4
    vals = np.exp(-shape**2 * (mesh[0, :]**2 + mesh[1, :]**2))
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(mesh[0, :], mesh[1,:], vals, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_zlim(0, 1.01)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([-1])
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    #fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


def equidistant_convergence_1d():
    error = []
    error_sep = []
    cond_sep = []
    def func(x):
        return np.exp(-np.abs(x - 3)**2) + 2
    cond = []
    N_vals = []
    for N in range(5, 1000):
        halflength = 1
        N_vals.append(N)
        print("N=", N)
        X = np.linspace(-halflength, halflength, N)
        in_mesh = X[np.newaxis, :]
        halflength -= halflength*0.2
        test_X = np.linspace(-halflength, halflength, 2000)
        test_mesh = test_X[np.newaxis, :]
        rbfqr = RBF_QR_1D(1e-5, in_mesh, func(X))
        error.append(rbfqr.RMSE(lambda mesh: func(mesh[0, :]), test_mesh))
        cond.append(np.linalg.cond(rbfqr.A))
        print("Cond QR:", cond[-1])
        print("RMSE QR:", error[-1])
        separated = SeparatedConsistent(Gaussian().shaped(5, X),
                                          X, func(X))
        cond_sep.append(separated.condC)
        error_sep.append(separated.RMSE(func, test_X))
        print("RMSE Sep:", error_sep[-1])
    color_qr = "xkcd:red"
    color_poly = "xkcd:blue"
    fig, ax1 = plt.subplots()
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_ylabel("RMSE")
    ax1.plot(N_vals, error, label="Error RBF-QR",
             color=color_qr, linestyle="dashed")
    ax1.plot(N_vals, error_sep, label="Error Separated Poly",
             color=color_poly, linestyle="dashed")

    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_ylabel("Condition")
    ax1.set_xlabel("Mesh size")
    ax2.plot(N_vals, cond,
             label="Condition RBF-QR",
             color=color_qr)
    ax2.plot(N_vals, cond_sep,
             label="Condition Separated Poly", color=color_poly)

    ax1.legend(loc="upper left")
    ax2.legend(loc="center left")
    ax1.set_title("RMSE Uniform mesh")
    plt.show()


def gc_convergence_1d():
    error = []
    error_sep = []
    cond_sep = []

    def func(x):
        return np.exp(-np.abs(x - 3)**2) + 2
    cond = []
    N_vals = []
    h_vals = []
    for N in range(5, 250):
        halflength = 1
        N_vals.append(N)
        print("N=", N)
        X = np.polynomial.chebyshev.chebgauss(N)[0] * halflength
        in_mesh = X[np.newaxis, :]
        halflength -= halflength*0.2
        test_X = np.linspace(-halflength, halflength, 2000)
        test_mesh = test_X[np.newaxis, :]
        rbfqr = RBF_QR_1D(1e-5, in_mesh, func(X))
        error.append(rbfqr.RMSE(lambda mesh: func(mesh[0, :]), test_mesh))
        cond.append(np.linalg.cond(rbfqr.A))
        print("Cond QR:", cond[-1])
        print("RMSE QR:", error[-1])
        separated = SeparatedConsistent(Gaussian().shaped(5, X),
                                          X, func(X))
        cond_sep.append(separated.condC)
        error_sep.append(separated.RMSE(func, test_X))
        print("RMSE Sep:", error_sep[-1])
    color_qr = "xkcd:red"
    color_poly = "xkcd:blue"
    fig, ax1 = plt.subplots()
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_ylabel("RMSE")
    ax1.set_xlabel("Mesh size")
    ax1.plot(N_vals, error, label="Error RBF-QR",
             color=color_qr, linestyle="dashed")
    ax1.plot(N_vals, error_sep, label="Error Separated Poly",
             color=color_poly, linestyle="dashed")

    ax2 = ax1.twinx()
    color2 = "xkcd:green"
    color2sep = "xkcd:orange"
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_ylabel("Condition")
    ax2.plot(N_vals, cond,
             label="Condition RBF-QR",
             color=color_qr)
    ax2.plot(N_vals, cond_sep,
             label="Condition Separated Poly", color=color_poly)

    ax1.legend(loc="upper left")
    ax2.legend(loc="center left")
    ax1.set_title("RMSE Gauss-Chebyshev")
    plt.show()


def equidistant_convergence_2d():
    def func(mesh):
        return np.sin(mesh[0, :]) - np.cos(mesh[1, :])
    N_vals = []
    error_qr = []
    error_poly = []
    cond_poly = []
    cond_qr = []
    for N in range(5, 55):
        N_vals.append(N)
        print("N=", N)
        halflength = 0.5
        X = np.linspace(-halflength, halflength, N)
        Y = np.copy(X)
        in_mesh = np.array(np.meshgrid(X, Y))
        in_mesh_flat = in_mesh.reshape(2, -1)
        halflength -= 0.1
        X_test = np.linspace(-halflength, halflength, 50)
        Y_test = np.copy(X_test)
        test_mesh = np.array(np.meshgrid(X_test, Y_test))
        test_mesh_flat = test_mesh.reshape(2, -1)
        in_vals = func(in_mesh_flat)
        rbfqr = RBF_QR_2D(1e-3, in_mesh_flat, in_vals)
        poly = SeparatedConsistent(functools.partial(Gaussian(), shape=5), in_mesh_flat.transpose(),
                                   in_vals.transpose())
        error_qr.append(rbfqr.RMSE(func, test_mesh_flat))
        error_poly.append(poly.RMSE(lambda mesh: func(mesh.transpose()), test_mesh_flat.transpose()))
        cond_qr.append(np.linalg.cond(rbfqr.A))
        cond_poly.append(poly.condC)
        print("RMSE QR: ", error_qr[-1])
        print("RMSE poly: ", error_poly[-1])
        print("Cond QR: ", cond_qr[-1])
        print("Cond poly: ", cond_poly[-1])
    color_qr = "xkcd:red"
    color_poly = "xkcd:blue"
    fig, ax1 = plt.subplots()
    ax1.set_ylabel("RMSE")
    ax1.set_xlabel("Mesh size in one dimension")
    ax1.set_yscale("log")
    ax1.plot(N_vals, error_qr, label="RMSE RBF-QR", color=color_qr, linestyle="dashed")
    ax1.plot(N_vals, error_poly, label="RMSE Separated Poly", color=color_poly, linestyle="dashed")

    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylabel("Condition")
    ax2.plot(N_vals, cond_qr, label="Condition RBF-QR", color=color_qr)
    ax2.plot(N_vals, cond_poly, label="Condition Separated Poly", color=color_poly)
    ax1.legend(loc="upper left")
    ax2.legend(loc="center left")
    ax1.set_title("RMSE Uniform mesh 2D")
    plt.show()


def gc_convergence_2d():
    def func(mesh):
        return np.sin(mesh[0, :]) - np.cos(mesh[1, :])
    N_vals = []
    error_qr = []
    error_poly = []
    cond_poly = []
    cond_qr = []
    for N in range(5, 55):
        N_vals.append(N)
        print("N=", N)
        halflength = 0.5
        X = np.polynomial.chebyshev.chebgauss(N)[0] * halflength
        Y = np.copy(X)
        in_mesh = np.array(np.meshgrid(X, Y))
        in_mesh_flat = in_mesh.reshape(2, -1)
        halflength -= 0.1
        X_test = np.linspace(-halflength, halflength, 50)
        Y_test = np.copy(X_test)
        test_mesh = np.array(np.meshgrid(X_test, Y_test))
        test_mesh_flat = test_mesh.reshape(2, -1)
        in_vals = func(in_mesh_flat)
        rbfqr = RBF_QR_2D(1e-3, in_mesh_flat, in_vals)
        poly = SeparatedConsistent(functools.partial(Gaussian(), shape=5), in_mesh_flat.transpose(),
                                   in_vals.transpose())
        error_qr.append(rbfqr.RMSE(func, test_mesh_flat))
        error_poly.append(poly.RMSE(lambda mesh: func(mesh.transpose()), test_mesh_flat.transpose()))
        cond_qr.append(np.linalg.cond(rbfqr.A))
        cond_poly.append(poly.condC)
        print("RMSE QR: ", error_qr[-1])
        print("RMSE poly: ", error_poly[-1])
        print("Cond QR: ", cond_qr[-1])
        print("Cond poly: ", cond_poly[-1])
    color_qr = "xkcd:red"
    color_poly = "xkcd:blue"
    fig, ax1 = plt.subplots()
    ax1.set_ylabel("RMSE")
    ax1.set_yscale("log")
    ax1.set_xlabel("Mesh size in one dimension")
    ax1.plot(N_vals, error_qr, label="RMSE RBF-QR", color=color_qr, linestyle="dashed")
    ax1.plot(N_vals, error_poly, label="RMSE Separated Poly", color=color_poly, linestyle="dashed")

    ax2 = ax1.twinx()
    ax2.set_yscale("log")
    ax2.set_ylabel("Condition")
    ax2.plot(N_vals, cond_qr, label="Condition RBF-QR", color=color_qr)
    ax2.plot(N_vals, cond_poly, label="Condition Separated Poly", color=color_poly)
    ax1.legend(loc="upper left")
    ax2.legend(loc="center left")
    ax1.set_title("Gauss-Chebyshev mesh 2D")
    plt.show()


def equidistant_time_2d():
    def func(mesh):
        return np.sin(mesh[0, :]) - np.cos(mesh[1, :])
    N_vals = []
    offline_qr = []
    offline_poly = []
    online_poly = []
    online_qr = []
    for N in range(5, 35):
        N_vals.append(N)
        print("N=", N)
        halflength = 0.5
        X = np.linspace(-halflength, halflength, N)
        Y = np.copy(X)
        in_mesh = np.array(np.meshgrid(X, Y))
        in_mesh_flat = in_mesh.reshape(2, -1)
        halflength -= 0.1
        X_test = np.linspace(-halflength, halflength, 50)
        Y_test = np.copy(X_test)
        test_mesh = np.array(np.meshgrid(X_test, Y_test))
        test_mesh_flat = test_mesh.reshape(2, -1)
        in_vals = func(in_mesh_flat)
        start = timer()
        rbfqr = RBF_QR_2D(1e-3, in_mesh_flat, in_vals)
        end = timer(); offline_qr.append(end-start)
        start = timer()
        poly = SeparatedConsistent(functools.partial(Gaussian(), shape=5), in_mesh_flat.transpose(),
                                   in_vals.transpose())
        end = timer(); offline_poly.append(end - start)
        start = timer()
        rbfqr(test_mesh_flat)
        end = timer(); online_qr.append(end - start)
        start = timer()
        poly(test_mesh_flat.transpose())
        end = timer(); online_poly.append(end - start)
        print("Offline QR: ", offline_qr[-1])
        print("Offline poly: ", offline_poly[-1])
        print("Online QR: ", online_qr[-1])
        print("Online poly: ", online_poly[-1])
    color_qr = "xkcd:red"
    color_poly = "xkcd:blue"
    fig, ax1 = plt.subplots()
    ax1.set_ylabel("Time (s)")
    ax1.set_xlabel("Mesh size in one dimension")
    ax1.set_yscale("log")
    ax1.plot(N_vals, offline_qr, label="Offline RBF-QR", color=color_qr, linestyle="dashed")
    ax1.plot(N_vals, offline_poly, label="Offline Separated Poly", color=color_poly, linestyle="dashed")
    ax1.plot(N_vals, online_qr, label="Online RBF-QR", color=color_qr)
    ax1.plot(N_vals, online_poly, label="Online Separated Poly", color=color_poly)
    ax1.legend(loc=2)
    ax1.set_title("Computation time 2D, split into offline and online phase")
    plt.show()

def combined():
    equidistant_convergence_1d()
    gc_convergence_1d()

def main():
    # intro_slide()
    # combined()
    # equidistant_convergence_1d()
     gc_convergence_1d()
    # equidistant_convergence_2d()
    # equidistant_time_2d()
    # gc_convergence_2d()
if __name__ == "__main__":
    main()