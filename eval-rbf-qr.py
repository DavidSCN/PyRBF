from rbf_qr import *
import mesh
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import tqdm


def func(x):
    return np.power(np.sin(5*x), 2) + np.exp(x/2)


def plot_rbf_qr():
    # in_mesh = mesh.GaussChebyshev_2D(12, 1, 4, 1)
    in_mesh = mesh.GaussChebyshev_1D(12, 1, 4, 0) - 4
    in_vals = func(in_mesh)
    plot_mesh = np.linspace(np.min(in_mesh), np.max(in_mesh), 2000)  # Use a fine mesh for plotting

    shape_param = 1e-3
    print("Proposed shape parameter =", shape_param)
    rbf_qr = RBF_QR_1D(shape_param, in_mesh, in_vals)

    fig1 = plt.figure("RBF-QR Interpolation")
    ax1 = fig1.add_subplot(111)
    ax1.set_title("Interpolation on N= " + str(len(in_mesh)) + " nodes with m=" + str(shape_param))
    ax1.plot(in_mesh, in_vals, "d")
    ax1.plot(plot_mesh, func(plot_mesh), "-", label="Original Function")
    ax1.plot(plot_mesh, rbf_qr(plot_mesh), "--", label="Interpolation RBF_QR_1D, m=" + str(shape_param))
    ax1.set_ylabel("y")
    ax1.set_xlabel("x")
    ax1.legend(loc=2)
    ax1.grid()

    fig2 = plt.figure("Error")
    ax2 = fig2.add_subplot(111)
    ax2.set_title("Error of RBF_QR on N=" + str(len(in_mesh)) + " nodes with m=" + str(shape_param))
    ax2.plot(plot_mesh, np.abs(func(plot_mesh) - rbf_qr(plot_mesh)), "-", label="Error RBF_QR_1D")
    ax2.plot(plot_mesh, np.full(plot_mesh.shape, np.finfo(np.float64).eps), "-", label="Machine precision (eps)")
    ax2.set_yscale("log")
    ax2.set_ylabel("Error")
    ax2.set_xlabel("x")
    ax2.legend(loc=2)
    ax2.grid()

    plt.show()


def evalShapeQR():
    in_mesh = mesh.GaussChebyshev_1D(4, 1, 4, 0)
    in_vals = func(in_mesh)
    test_mesh = np.linspace(np.min(in_mesh), np.max(in_mesh), 2000)
    shape_parameter_space = np.logspace(0.8, 0, num=100)
    rmse = []
    for shape_param in shape_parameter_space:
        print("m = ", shape_param)
        rbf_qr = RBF_QR_1D(shape_param, in_mesh, in_vals)
        rmse.append(rbf_qr.RMSE(func, test_mesh))
    rmse = np.array(rmse)
    plt.plot(shape_parameter_space, rmse, "-", label="RMSE")
    plt.title("RMSE of RBF-QR for different epsilon, both axes log, mapping " + str(len(in_mesh))
              + " nodes to " + str(len(test_mesh)) + " nodes")
    plt.ylabel("RMSE")
    plt.xlabel("epsilon")
    plt.yscale("log")
    # plt.xscale("log") #mplot3d bug
    plt.legend(loc=2)
    plt.grid()
    plt.show()


def combined():
    shape_parameter_space = np.logspace(0, -10, num=50)
    chebgauss_order_space = range(1, 40, 1)
    X, Y = np.meshgrid(shape_parameter_space, chebgauss_order_space)
    rmse = np.empty(X.shape)
    for i, j in tqdm(list(np.ndindex(X.shape))):
        shape = X[i, j]
        order = Y[i, j]
        in_mesh = mesh.GaussChebyshev_1D(order, 1, 4, 0)
        in_vals = func(in_mesh)
        test_mesh = np.linspace(np.min(in_mesh), np.max(in_mesh), 100)
        rbf_qr = RBF_QR_1D(shape, in_mesh, in_vals)
        rmse[i, j] = rbf_qr.RMSE(func, test_mesh)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(np.log10(X), 4 * Y, np.log10(rmse))
    ax.set_xlabel("Shape parameter (log)")
    ax.set_ylabel("Meshsize")
    ax.set_zlabel("RMSE (log)")
    plt.show()


def test_rbf_qr_2d():
    halflength = 5
    X = np.linspace(-halflength, halflength, 9) + 10
    Y = np.linspace(-halflength, halflength, 9) + 10
    in_mesh = np.meshgrid(X, Y)

    def func(mesh):
        return np.sin(mesh[0]) - np.cos(mesh[1])

    in_vals = func(in_mesh)
    halflength = 4
    X_test = np.linspace(-halflength, halflength, 100) + 10
    Y_test = np.linspace(-halflength, halflength, 100) + 10
    test_mesh = np.meshgrid(X_test, Y_test)
    obj = RBF_QR_2D(0.001, in_mesh, in_vals)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_title("Original")
    ax.plot_surface(test_mesh[0], test_mesh[1], func(test_mesh))

    fig2 = plt.figure()
    ax2 = fig2.gca(projection="3d")
    ax2.set_title("Interpolation")
    ax2.plot_surface(test_mesh[0], test_mesh[1], obj(test_mesh))

    fig3 = plt.figure()
    ax3 = fig3.gca(projection="3d")
    ax3.set_title("Error, log")
    ax3.plot_surface(test_mesh[0], test_mesh[1], np.log(np.abs(obj(test_mesh) - func(test_mesh))))

    print("RMSE: ", obj.RMSE(func, test_mesh))
    plt.show()


def check_basisfun_2d():
    X = np.linspace(-3, 3, 10)
    Y = np.linspace(-3, 3, 10)
    in_mesh = np.meshgrid(X, Y)

    def func(mesh):
        return np.sin(mesh[0]) - np.cos(mesh[1]) + 1.2

    in_vals = func(in_mesh)
    halflength = math.sqrt(2) / 2
    X_test = np.linspace(-halflength, halflength, 100)
    Y_test = np.linspace(-halflength, halflength, 100)
    test_mesh = np.meshgrid(X_test, Y_test)
    obj = RBF_QR_2D(0.001, in_mesh, in_vals)
    fig3 = plt.figure()
    ax3 = fig3.gca(projection="3d")
    basisfuncs = [(i, obj.basisfunction_i(i)) for i in [int(3)]]

    def cart2pol(x, y):
        rho = np.sqrt(x ** 2 + y ** 2)
        phi = np.arctan2(y, x)
        return rho, phi

    temp_mesh = cart2pol(test_mesh[0].reshape(-1), test_mesh[1].reshape(-1))
    test_mesh = np.array(temp_mesh)
    # rmax = test_mesh[0, :].max()
    # test_mesh[0, :] /= rmax

    for i, basisfunc in basisfuncs:
        arr = basisfunc(test_mesh).reshape(100, 100)

        def pol2cart(rho, phi):
            x = rho * np.cos(phi)
            y = rho * np.sin(phi)
            return (x, y)

        plot_mesh = pol2cart(test_mesh[0, :], test_mesh[1, :])
        plot_mesh = np.array(plot_mesh)
        ax3.plot_wireframe(plot_mesh[0, :].reshape(100, 100),
                           plot_mesh[1, :].reshape(100, 100), arr)
    plt.show()


def test_rbf_qr_3d():
    def func(mesh):
        return np.sin(mesh[0, :]) - np.cos(mesh[1, :]) + np.sin(mesh[2, :])
    halflength = math.sqrt(3) / 3 - 0.1
    X = np.linspace(-halflength, halflength, 8)
    Y = np.linspace(-halflength, halflength, 8)
    Z = np.linspace(-halflength, halflength, 8)
    in_mesh = np.array(np.meshgrid(X, Y, Z))
    in_vals = func(in_mesh)

    halflength -= 0.2   # rigged test mesh ;)
    X_test = np.linspace(-halflength, halflength, 12)
    Y_test = np.linspace(-halflength, halflength, 12)
    Z_test = np.linspace(-halflength, halflength, 12)
    test_mesh = np.array(np.meshgrid(X_test, Y_test, Z_test))

    rbf_qr_3d = RBF_QR_3D(1e-3, in_mesh, in_vals)
    rmse = rbf_qr_3d.RMSE(func, test_mesh)
    print(rmse)


def main():
    # test_rbf_qr_3d()
     test_rbf_qr_2d()
    # evalShapeQR()
    # plot_rbf_qr()


if __name__ == "__main__":
    main()