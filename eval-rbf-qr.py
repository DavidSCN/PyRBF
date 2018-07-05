from rbf_qr import *
from rbf import *
import basisfunctions
import mesh
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from sympy.matrices import *
import coordinate_helper

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
    halflength = 1
    X = np.linspace(-halflength, halflength, 6)
    Y = np.linspace(-halflength, halflength, 6)
    in_mesh = np.meshgrid(X, Y)

    def func(mesh):
        return np.sin(mesh[0]) - np.cos(mesh[1])

    in_vals = func(in_mesh)
    halflength = 0.9
    X_test = np.linspace(-halflength, halflength, 200)
    Y_test = np.linspace(-halflength, halflength, 200)
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
        return np.sin(mesh[0, :]) - np.cos(mesh[1, :]) + mesh[2,:]
    halflength = 1e-2
    X = np.linspace(-halflength, halflength, 4)
    Y = np.linspace(-halflength, halflength, 4)
    Z = np.linspace(-halflength, halflength, 4)
    in_mesh = np.array(np.meshgrid(X, Y, Z))
    in_vals = func(in_mesh)

    halflength -= 2e-3   # rigged test mesh ;)
    X_test = np.linspace(-halflength, halflength, 10)
    Y_test = np.linspace(-halflength, halflength, 10)
    Z_test = np.linspace(-halflength, halflength, 10)
    test_mesh = np.array(np.meshgrid(X_test, Y_test, Z_test))
    rbf_qr_3d = RBF_QR_3D(1e-3, in_mesh, in_vals)
    rmse = rbf_qr_3d.RMSE(func, test_mesh)

    print(rmse)

def rigged_2d_mesh():
    def func(mesh):
        return np.sin(mesh[0, :]) - np.cos(mesh[1, :])
    halflength = 1
    N = 20
    X = np.linspace(-halflength, halflength, N)
    Y = np.linspace(-halflength, halflength, N)     # All points on a line y = x
    in_mesh = np.array((X, Y))
    in_vals = func(in_mesh)
    rbf_qr = RBF_QR_2D(1e-2, in_mesh, in_vals)

    halflength -= 1e-1
    M = 1000
    test_X = np.linspace(-halflength, halflength, M)
    test_Y = np.linspace(-halflength, halflength, M)
    test_mesh = np.array((test_X, test_Y))
    print(rbf_qr.RMSE(func, test_mesh))

class VPANoneConsistent(RBF):
    def __init__(self, shape_param, in_mesh, in_vals):
        in_vals = np.array(in_vals, copy=False).reshape(-1)
        self.shape_param, self.in_mesh = shape_param, np.array(in_mesh).reshape(2, -1) # Fixme: assumes 2D
        dim = self.in_mesh.shape[1]
        C = Matrix(np.zeros((dim, dim)))
        for i, j in np.ndindex(dim, dim):
            distance = np.linalg.norm(self.in_mesh[:, i] - self.in_mesh[:, j])
            C[i, j] = math.exp(-(shape_param**2 * distance**2))
        self.lamb = C.LUsolve(Matrix(in_vals))

    def __call__(self, out_mesh):
        original_shape = np.shape(out_mesh)
        out_mesh = np.array(out_mesh).reshape(original_shape[0],-1)
        A = Matrix(np.zeros((out_mesh.shape[1], self.in_mesh.shape[1])))
        for i, j in np.ndindex(A.shape):
            distance = np.linalg.norm(out_mesh[:, i] - self.in_mesh[:, j])
            A[i, j] = math.exp(-(self.shape_param**2 * distance**2))
        out_vals = A @ self.lamb
        return np.array(out_vals).astype(np.float64).reshape(original_shape[1:])

def equidistant_convergence_1d():
    halflength = 1
    error = []
    error_integ = []
    error_sep = []
    cond_sep = []

    def func(x):
        return np.exp(-np.abs(x - 3)**2) + 2
    cond = []
    N_vals = []
    m_vals = []
    h_vals = []
    for N in range(10, 70):
        N_vals.append(N)
        print("N=", N)
        halflength = 1
        h = halflength *2. / N
        h_vals.append(h)
        shape_param = math.sqrt(-math.log(1e-9)) / (5 * h)
        print("shape_param:", shape_param)
        #print("m = ", m_vals[-1])
        X = np.linspace(-halflength, halflength, N)
        in_mesh = X[np.newaxis, :]
        halflength -= halflength*0.2
        X = np.linspace(-halflength, halflength, N * 8)
        test_mesh = X[np.newaxis, :]
        rbfqr = RBF_QR_1D(shape_param/100, in_mesh, func(in_mesh[0,:]))
        error.append(rbfqr.RMSE(lambda mesh: func(mesh[0,:]), test_mesh))
        cond.append(np.linalg.cond(rbfqr.A))
        print("Cond QR:", cond[-1])
        print("RMSE QR:", error[-1])
        separated = SeparatedConsistent(functools.partial(Gaussian(), shape=shape_param),
                                          np.transpose(in_mesh), func(in_mesh[0, :]))
        cond_sep.append(np.linalg.cond(separated.C))
        error_sep.append(separated.RMSE(lambda mesh: func(mesh[0, :]), np.transpose(test_mesh)))
        print("RMSE Sep:", error_sep[-1])
    color1 = "xkcd:red"
    color1sep = "xkcd:blue"
    fig, ax1 = plt.subplots()
    ax1.set_yscale("log")
    ax1.set_ylabel("RMSE")
    ax1.plot(N_vals, error, label="Error RBF-QR",
             color=color1, linestyle="dashed")
    ax1.plot(N_vals, error_sep, label="Error Separated Poly",
             color=color1sep, linestyle="dashed")

    ax2 = ax1.twinx()
    color2 = "xkcd:green"
    color2sep = "xkcd:orange"
    ax2.set_yscale("log")
    ax2.set_ylabel("Condition")
    ax2.plot(N_vals, cond,
             label="Condition RBF-QR",
             color=color2)
    ax2.plot(N_vals, cond_sep,
             label="Condition Separated Poly", color=color2sep)

    ax1.legend(loc=1)
    ax2.legend(loc=2)
    ax1.grid()

    plt.show()
def equidistant_convergence_2d():
    halflength = 1
    def func(mesh):
        mesh = np.array(mesh)
        return np.sin(mesh[0, :]) - np.cos(mesh[1, :])
    error = []
    error_integ = []
    error_sep = []
    h_vals = []
    fig1 = plt.figure("Convergence")

    for N in range(5, 60):
        print("N=", N)
        halflength = math.pi
        h = halflength * 2. / N
        h_vals.append(h)
        X = np.linspace(-halflength, halflength, N)
        Y = np.linspace(-halflength, halflength, N)
        in_mesh = np.array(np.meshgrid(X, Y)).reshape(2, -1)
        halflength -= halflength*0.2
        X = np.linspace(-halflength, halflength, N * 8)
        Y = np.linspace(-halflength, halflength, N * 8)
        test_mesh = np.array(np.meshgrid(X, Y)).reshape(2, -1)
        error.append(RBF_QR_2D(1e-3, in_mesh, func(in_mesh)).RMSE(func, test_mesh))
        print("RMSE QR:", error[-1])
        #error_integ.append(
        #    IntegratedConsistent(functools.partial(Gaussian(), shape = 1e-3), np.transpose(in_mesh),func(in_mesh))
        #    .error(lambda mesh: func(np.transpose(mesh)), np.transpose(test_mesh)))
        #print("Error integrated Consistent", error_integ[-1])
        error_sep.append(
            SeparatedConsistent(functools.partial(Gaussian(), shape=1e-3), np.transpose(in_mesh), func(in_mesh))
            .RMSE(lambda mesh: func(np.transpose(mesh)), np.transpose(test_mesh))
        )
        plt.cla()
        plt.title("Error on different grid sizes")
        plt.yscale("log")
        plt.xscale("log")
        plt.plot(h_vals, error, label="Error RBF-QR")
        #plt.plot(h_vals, error_integ, label="Error RBF Integrated Polynomial")
        plt.plot(h_vals, error_sep, label="Error RBF Separated Polynomial")
        plt.legend(loc=2)
        plt.pause(0.05)
    plt.show()

def polynomials_convergence_1d():
    error_sep = []
    cond_sep = []
    N = 192
    def func(x):
        return np.exp(-np.abs(x - 3)**2) + 2
    h_vals = []
    m_vals = []
    for m in range(1, 40):
        m_vals.append(m)
        print("m=", m)
        halflength = 1
        h = halflength * 2. / N
        h_vals.append(h)
        shape_param = math.sqrt(-math.log(1e-9)) / (m * h)
        print("shape_param:", shape_param)
        # print("m = ", m_vals[-1])
        X = np.linspace(-halflength, halflength, N)
        in_mesh = X[np.newaxis, :]
        halflength -= halflength * 0.2
        X = np.linspace(-halflength, halflength, N * 8)
        test_mesh = X[np.newaxis, :]
        separated = SeparatedConsistent(functools.partial(Gaussian(), shape=shape_param),
                                        np.transpose(in_mesh), func(in_mesh[0, :]))
        cond_sep.append(np.linalg.cond(separated.C))
        error_sep.append(separated.RMSE(lambda mesh: func(mesh[0, :]), np.transpose(test_mesh)))
        print("RMSE Sep:", error_sep[-1])
    color1sep = "xkcd:blue"
    with plt.xkcd():
        start = [1e-1 for _ in m_vals]
        end = [1e-8 for _ in m_vals]
        fit = [10**(-(m/2)) if (m/2) < 8 else 10**(-8) for m in m_vals]
        plt.yscale("log")
        plt.yticks([1e-1, 1e-8],  ["bad", "good"])
        plt.xlabel("m")
        plt.ylabel("error")
        plt.plot(m_vals, start, label="Where it should start", linestyle="dashed")
        plt.plot(m_vals, fit, label="What it should look like")
        plt.plot(m_vals, error_sep, label="Error Separated Poly",
                 color=color1sep)
        plt.plot(m_vals, end, label="Where it should end", linestyle="dashed")
        plt.legend(loc=0)
        plt.show()

def main():
     polynomials_convergence_1d()
    # equidistant_convergence_1d()
    # equidistant_convergence_2d()
    # rigged_2d_mesh()
    # test_rbf_qr_3d()
    # test_rbf_qr_2d()
    # evalShapeQR()
    # plot_rbf_qr()


if __name__ == "__main__":
    main()