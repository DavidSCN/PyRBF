# import MLS
from rbf import *
# from plot_helper import *
from mesh import GaussChebyshev_1D
from ipdb import set_trace
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import mesh
import tqdm

def speedup_and_efficiency(procs, times):
    T1 = times[0] * procs[0] # Define baseline
    S = T1 / times
    E = S / procs
    return S, E


def plot_supermuc_scaling():
    data = np.genfromtxt("supermuc_scaling.csv",
                         delimiter = ",",
                         skip_header = 1,
                         names = ("ranks", "computeMapping", "fillA", "fillC", "solve", "total", "advance"))

    ranks = data["ranks"]
    # plt.plot(ranks, data["advance"], label="advance")
    # plt.plot(ranks, data["computeMapping"], label="compute Mapping")
    # plt.plot(ranks, data["fillA"], label="fill A")
    # plt.plot(ranks, data["fillC"], label="fill C")
    # plt.plot(ranks, data["total"], label="Total")

    # plt.plot(ranks, speedup_and_efficiency(ranks, data["total"])[1], label = "Total Runtime")
    plt.plot(ranks, speedup_and_efficiency(ranks, data["advance"])[1], label = "Advance")
    plt.plot(ranks, speedup_and_efficiency(ranks, data["computeMapping"])[1], label = "Compute Mapping")
    plt.plot(ranks, speedup_and_efficiency(ranks, data["solve"])[1], label = "Solve")
    plt.grid()
    plt.xlabel("Processors")
    plt.ylabel("Parallel Efficiency")
    plt.legend()
    plt.show()


def plot_mesh_sizes():
    GC = True
    # mesh_sizes = np.arange(2, 64) if GC else np.linspace(4, 1024, 200)
    mesh_sizes = np.arange(2, 32) if GC else np.linspace(4, 496, 200)
    test_mesh = np.linspace(1, 4, 4000)
    test_vals = func(test_mesh)
    m = 15
    
    separated = []
    integrated = []
    no_pol = []
    
    for size in mesh_sizes:
        in_mesh = mesh.GaussChebyshev_1D(size, 0.25, 4, 1) if GC else np.linspace(1, 4, size)
        in_vals = func(in_mesh)
        shape = rescaleBasisfunction(Gaussian, m, in_mesh)
        print("Computing with in mesh size =", int(size), ", resulting shape parameter =", shape)
        # bf = functools.partial(Gaussian, shape=shape)
        bf = create_BFs(Gaussian, m, in_mesh)
        separated.append(SeparatedConsistent(bf, in_mesh, in_vals))
        integrated.append(IntegratedConsistent(bf, in_mesh, in_vals))
        no_pol.append(NoneConsistent(bf, in_mesh, in_vals))


    if GC: mesh_sizes = mesh_sizes * 4 * 4 # for GC
    fig, ax1 = plt.subplots()
    print("Computing RMSE")
    ax1.semilogy(mesh_sizes, [i.RMSE(func, test_mesh) for i in separated], "--",
                 label = "RMSE separated Polynomial")
    ax1.semilogy(mesh_sizes, [i.RMSE(func, test_mesh) for i in integrated], "--",
                 label = "RMSE integrated polynomial")
    ax1.semilogy(mesh_sizes, [i.RMSE(func, test_mesh) for i in no_pol], "--",
                 label = "RMSE no polynomial")
    
    ax1.set_ylabel("RMSE")
    # ax1.legend(loc = 3)

    ax2 = ax1.twinx()
    print("Computing conditon")
    ax2.semilogy(mesh_sizes, [i.condC for i in separated],label = "Condition separated / no polynomial")
    ax2.semilogy(mesh_sizes, [i.condC for i in integrated], label = "Condition integrated polynomial")
    ax2.set_ylabel("Condition Number")
    # ax2.legend(loc = 2, framealpha = 1)

    multi_legend(ax1, ax2, loc = 7)
    
    ax1.set_xlabel("Input Mesh Size")
    # ax1.set_xlabel("Gauss-Chebyshev Order")
    # plt.title("m = " + str(m))

    rm = [i.RMSE(func, test_mesh) for i in separated]
        
    plt.grid()
    # plt.show()
    plt.savefig("rc-gc-size.pdf")
        
def plot_shape_parameters():
    """ Plots over a range of shape parameters. """
    # in_mesh = np.linspace(1, 4, 192)
    in_mesh = mesh.GaussChebyshev_1D(12, 0.25, 4, 1)
    in_vals = func(in_mesh)
    test_mesh = np.linspace(1, 4, 2000)
        
    ms = np.linspace(1, 30, 50)

    separated = []
    integrated = []
    no_pol = []

    for m in ms:
        print("Working on m =", m)
        # bf = create_BFs(Gaussian, m, in_mesh)
        bf = functools.partial(Gaussian, shape=rescaleBasisfunction(Gaussian, m, in_mesh))
        
        separated.append(SeparatedConsistent(bf, in_mesh, in_vals))
        integrated.append(IntegratedConsistent(bf, in_mesh, in_vals))
        no_pol.append(NoneConsistent(bf, in_mesh, in_vals))

    fig, ax1 = plt.subplots()
    ax1.semilogy(ms, [i.RMSE(func, test_mesh) for i in separated], "--",
                 label = "RMSE separated polynomial")
    ax1.semilogy(ms, [i.RMSE(func, test_mesh) for i in integrated], "--",
                 label = "RMSE integrated polynomial")
    ax1.semilogy(ms, [i.RMSE(func, test_mesh) for i in no_pol], "--",
                 label = "RMSE no polynomial")
    
    ax1.set_ylabel("RMSE")
    ax1.set_xlabel("m (included vertices in basis function)")
    
    ax2 = ax1.twinx()
    ax2.semilogy(ms, [i.condC for i in separated],label = "Condition separated / no polynomial")
    ax2.semilogy(ms, [i.condC for i in integrated], label = "Condition integrated polynomial")
    ax2.set_ylabel("Condition Number")
    
    multi_legend(ax1, ax2, loc = 7)
    plt.grid()
    rm = [i.RMSE(func, test_mesh) for i in separated]
    
    plt.show()
    # plt.savefig("rc-gc-m-rescaled.pdf")
    
       

def plot_basic_consistent():
    # in_mesh = np.concatenate((np.linspace(1, 2, 3), np.linspace(2.1, 4, 9)))
    # print(in_mesh)
    in_mesh = np.linspace(-10, 10, 4)
    # in_mesh = mesh.GaussChebyshev_2D(12, 1, 4, 1)
    in_vals = func(in_mesh)
    plot_mesh = np.linspace(np.min(in_mesh) - 0.1, np.max(in_mesh) + 0.1, 2000) # Use a fine mesh for plotting
    test_mesh = np.linspace(np.min(in_mesh), np.max(in_mesh), 2000)

    m = 3
    bf = Gaussian().shaped(m, in_mesh)
    # bf = create_BFs(Gaussian, m, in_mesh)
    print("Proposed shape parameter =", Gaussian().shape_param(m, in_mesh))
    
    plt.plot(in_mesh, in_vals, "d")
    plt.plot(plot_mesh, func(plot_mesh), "-", label = "Original Function")
    set_trace()
    sep_consistent = SeparatedConsistent(bf, in_mesh, in_vals, rescale = True)
    plt.plot(plot_mesh, sep_consistent(plot_mesh), "--", label = "Separated Interpolant")
    plt.plot(plot_mesh, sep_consistent.polynomial(plot_mesh), "-", label = "Separated Polynomial")
    plt.plot(plot_mesh, func(plot_mesh) - sep_consistent(plot_mesh), "-", label = "Error separated consistent")
    print("RMSE SeparatedConsistent ExtMesh =", sep_consistent.RMSE(func, plot_mesh))
    print("RMSE SeparatedConsistent OriMesh =", sep_consistent.RMSE(func, test_mesh))

    # fit_consistent = SeparatedConsistentFitted(bf, in_mesh, in_vals)
    # plt.plot(plot_mesh, fit_consistent(plot_mesh), "--", label = "Fitted Interpolant")
    # plt.plot(plot_mesh, fit_consistent.polynomial(plot_mesh), "-", label = "Fitted Polynomial")
    # plt.plot(plot_mesh, func(plot_mesh) - fit_consistent(plot_mesh), "-", label = "Error fitted consistent")
    # print("RMSE FittedConsistent ExtMesh =", fit_consistent.RMSE(func, plot_mesh))
    # print("RMSE FittedConsistent OriMesh=", fit_consistent.RMSE(func, test_mesh))
    
    # none_consistent = NoneConsistent(bf, in_mesh, in_vals)
    # plt.plot(plot_mesh, none_consistent(plot_mesh), "--", label = "No Polynomial Interpolant")
    # plt.plot(plot_mesh, func(plot_mesh) - none_consistent(plot_mesh), "-", label = "Error none consistent")
    # print("RMSE NoneConsistent =", none_consistent.RMSE(func, plot_mesh))
    

    plt.legend(loc=2)
    # plt.ylim( -0.1, np.amax(func(plot_mesh)) * 1.05 )
    plt.title("m = " + str(m))
    plt.grid()
    # plt.savefig("basic.pdf")
    plt.show()


def plot_basic_conservative():
    func = lambda x: np.power(np.sin(5*x), 2) + np.exp(x/2)
    # func = lambda x: np.full_like(x, 4)
    # func = lambda x: x
    

    # in_mesh = np.linspace(1, 4, 25)
    in_mesh = GaussChebyshev_1D(10, 1, 4, 1)
    in_vals = func(in_mesh)
    out_mesh = np.linspace(np.min(in_mesh), np.max(in_mesh), 20) 
    plot_mesh = np.linspace(np.min(in_mesh)-0.5, np.max(in_mesh)+0.5, 1000) # Use a fine mesh for plotting 

    m = 6
    bf = Gaussian().shaped(m, in_mesh)
    
    if len(in_mesh) < 60: plt.plot(in_mesh, in_vals, "d")
    plt.plot(plot_mesh, func(plot_mesh), "r-", label = "Original Function")
    
    none_conservative = NoneConservative(bf, in_mesh, in_vals)
    # plt.plot(out_mesh, none_conservative(out_mesh), "g-", label = "No Polynomial Interpolant")
    # plt.plot(out_mesh, none_conservative(out_mesh) - func(out_mesh), "g:", label = "Error none conservative")
    # plt.plot(out_mesh, none_conservative.weighted_error(func, out_mesh), "g-.", label = "Weighted Error none conservative")
    
    print("RMSE NoneConservative =", none_conservative.RMSE(func, out_mesh))

    # none_conservative_resc = NoneConservative(bf, in_mesh, in_vals, True) 
    # plt.plot(out_mesh, none_conservative_resc(out_mesh), "c-", label = "No Polynomial Rescaled Interpolant")
    # plt.plot(out_mesh, func(out_mesh) - none_conservative_resc(out_mesh), "c:", label = "Error none conservative rescaled")
    # plt.plot(out_mesh, none_conservative_resc.weighted_error(func, out_mesh), "c-.", label = "Weighted Error none conservative rescaled")
    
    # plt.plot(out_mesh, none_conservative_resc.rescalingInterpolant(out_mesh), "c-", label = "No Polynomial One Interpolant")
    # print("RMSE NoneConservative Rescaled =", none_conservative_resc.RMSE(func, out_mesh))
    
    in_conservative = IntegratedConservative(bf, in_mesh, in_vals, False)
    # plt.plot(out_mesh, in_conservative(out_mesh), "m-", label = "Integrated Interpolant")
    # plt.plot(out_mesh, in_conservative(out_mesh) - func(out_mesh), "m:", label = "Error integrated conservative")
    # plt.plot(out_mesh, in_conservative.weighted_error(func, out_mesh), "m-.", label = "Weighted Error integrated conservative")

    sep_conservative = SeparatedConservative(bf, in_mesh, in_vals)
    plt.plot(out_mesh, sep_conservative(out_mesh), "b-", label = "Separated Interpolant")
    plt.plot(out_mesh, sep_conservative(out_mesh) - func(out_mesh), "b:", label = "Error separated conservative")
    # plt.plot(out_mesh, sep_conservative.weighted_error(func, out_mesh), "b-.", label = "Weighted Error separated conservative")
    plt.plot(out_mesh, sep_conservative.rescaled_error(func, out_mesh), "b--", label = "Scaled Error separated conservative")
        
    
    # # plt.plot(out_mesh, in_conservative.polynomial(out_mesh), "b-.", label = "Integrated Polynomial")
    # print("RMSE InConservative =", in_conservative.RMSE(func, out_mesh))
    
    plt.legend(loc=2)
    # plt.ylim( np.amin(func(plot_mesh)) * 0.9, np.amax(func(plot_mesh)) * 1.05 )
    plt.title("GaussChebyshev Nodes, m = " + str(m) + ", " + str(len(in_mesh)) + " nodes on " + str(len(out_mesh)) + " nodes.")
    plt.grid()
    # plt.savefig("basic.pdf")
    plt.show()



    
def plot_rmse_cond():
    """ Plots over a range of shape parameters. """
    in_mesh = np.linspace(1, 4, 192)
    # in_mesh = GaussChebyshev(12, 0.25, 4, 1)
    in_vals = func(in_mesh)
    test_mesh = np.linspace(1, 4, 2000)
        
    ms = np.linspace(1, 20, 50)

    separated = []
    integrated = []
    no_pol = []
    separated_res = []
    no_pol_res = []
    sep_fitresc = []

    
    for m in ms:
        print("Working on m =", m)
        bf = functools.partial(Gaussian, shape=rescaleBasisfunction(Gaussian, m, in_mesh))
        
        separated.append(SeparatedConsistent(bf, in_mesh, in_vals, False))
        integrated.append(IntegratedConsistent(bf, in_mesh, in_vals))
        no_pol.append(NoneConsistent(bf, in_mesh, in_vals, False))
        separated_res.append(SeparatedConsistent(bf, in_mesh, in_vals, True))
        no_pol_res.append(NoneConsistent(bf, in_mesh, in_vals, True))
        sep_fitresc.append(SeparatedConsistentFitted(bf, in_mesh, in_vals, True))

    fig, ax1 = plt.subplots()
    ax1.loglog([i.RMSE(func, test_mesh) for i in no_pol], [i.condC for i in no_pol], label = "No Polynomial")
    ax1.loglog([i.RMSE(func, test_mesh) for i in integrated],[i.condC for i in integrated], label = "Integrated Polynomial")
    ax1.loglog([i.RMSE(func, test_mesh) for i in separated], [i.condC for i in separated], label = "Separated Polynomial")
    
    ax1.loglog([i.RMSE(func, test_mesh) for i in no_pol_res], [i.condC for i in no_pol_res], label = "No polynomial, rescaled")
    ax1.loglog([i.RMSE(func, test_mesh) for i in separated_res], [i.condC for i in separated_res], label = "Separated polynomial, rescaled")
    ax1.loglog([i.RMSE(func, test_mesh) for i in sep_fitresc], [i.condC for i in sep_fitresc], label = "Separated, fitted, rescaled")
    
    ax1.set_ylabel("Condition")
    ax1.set_xlabel("RMSE")

    ax1.annotate("better", weight = "bold", size = "large",
                 xycoords="axes fraction", xy = (0.05, 0.05), xytext = (0.25, 0.25),
                 arrowprops = {"width" : 5, "headwidth" : 10, "shrink": 30})

    ax1.set_xlim([5e-8, 0.1])

    ax1.legend()
    # plt.grid()
    plt.savefig("rmse_cond.pdf")
    # plt.show()
     
    
def gc_m_order():
    """ Keep number of elements constant, increase order, thus also points.
    Different plots for different m's.
    to have a comparision to equi-distant meshes """

    ms = [2, 4, 6, 8] 
    gc_orders = np.arange(2, 24)
        
    test_mesh = np.linspace(1, 4, 5000)
    test_vals = func(test_mesh)
    
    f, sp = plt.subplots(2, 2, sharex='col', sharey='row')
    sp_lin = [ sp[0][0], sp[0][1], sp[1][0], sp[1][1] ]
    
    for i, m in enumerate(ms):
        print("Working on m =", m)
        separated = []
        integrated = []
        no_pol = []
    
        for gc_order in gc_orders:
            in_mesh = mesh.GaussChebyshev_1D(gc_order, 1, 4, 1)
            in_vals = func(in_mesh)
            # shape = rescaleBasisfunction(Gaussian, m, in_mesh)
            shape = -1
            bf = create_BFs(Gaussian, m, in_mesh)
            # bf = functools.partial(Gaussian, shape=shape)
            
            separated.append(SeparatedConsistent(bf, in_mesh, in_vals))
            integrated.append(IntegratedConsistent(bf, in_mesh, in_vals))
            no_pol.append(NoneConsistent(bf, in_mesh, in_vals))

        ax1 = sp_lin[i]

        ax1.semilogy(gc_orders, [i.RMSE(func, test_mesh) for i in separated], "--",
                     label = "RMSE separated Polynomial")
        ax1.semilogy(gc_orders, [i.RMSE(func, test_mesh) for i in integrated], "--",
                     label = "RMSE integrated polynomial")
        ax1.semilogy(gc_orders, [i.RMSE(func, test_mesh) for i in no_pol], "--",
                     label = "RMSE no polynomial")

        ax1.set_ylabel("RMSE")
        # ax1.set_ylim(10e-6, 10e0)
        ax1.legend(loc = 3)

        ax2 = ax1.twinx()
        ax2.semilogy(gc_orders, [i.condC for i in separated],  label = "Condition separated / no polynomial")
        ax2.semilogy(gc_orders, [i.condC for i in integrated], label = "Condition integrated polynomial")
        ax2.set_ylabel("Condition Number")
        ax2.legend()

        ax1.set_xlabel("Gauss-Chebyshev Order")
        ax1.set_title("m = {} (shape parameter = {:.3f}), element size = 1, domain size = 4".format(m, shape))
        ax1.grid()

    plt.show()
    # plt.savefig("rc-gc-m-order.pdf")
    

def points_s():
    """ Increase number of points.
    Different plots for different s. s values come from increasing of gc order plot """

    ss = [ 33.432, 16.716, 11.144, 8.358 ]
    in_mesh_sizes = np.arange(8, 96+1)
    
    test_mesh = np.linspace(1, 4, 5000)
    test_vals = func(test_mesh)
    
    f, sp = plt.subplots(2, 2, sharex='col', sharey='row')
    sp_lin = [ sp[0][0], sp[0][1], sp[1][0], sp[1][1] ]
    
    for i, s in enumerate(ss):
        print("Working on s =", s)
        separated = []
        integrated = []
        no_pol = []
    
        for size in in_mesh_sizes:
            in_mesh = np.linspace(1, 4, size)
            # in_mesh = GaussChebyshev(gc_order, 1, 4, 1)
            in_vals = func(in_mesh)
            bf = functools.partial(Gaussian, shape=s)

            separated.append( separated_consistent(bf, in_mesh, in_vals) )
            integrated.append( integrated_consistent(bf, in_mesh, in_vals) )
            no_pol.append( no_pol_consistent(bf, in_mesh, in_vals) )

        ax1 = sp_lin[i]

        ax1.semilogy(in_mesh_sizes, [rmse(k[0](test_mesh), test_vals) for k in separated], "--",
                     label = "RMSE separated polynomial")
        ax1.semilogy(in_mesh_sizes, [rmse(k[0](test_mesh), test_vals) for k in integrated], "--",
                     label = "RMSE integrated polynomial")
        ax1.semilogy(in_mesh_sizes, [rmse(k[0](test_mesh), test_vals) for k in no_pol], "--",
                     label = "RMSE no polynomial")

        ax1.set_ylabel("RMSE")
        ax1.set_ylim(10e-6, 10e0)
        ax1.legend(loc = 3)

        ax2 = ax1.twinx()
        ax2.semilogy(in_mesh_sizes, [i[2] for i in separated],  label = "Condition separated / no polynomial")
        ax2.semilogy(in_mesh_sizes, [i[2] for i in integrated], label = "Condition integrated polynomial")
        ax2.set_ylabel("Condition Number")
        ax2.legend()

        ax1.set_xlabel("Number of points")
        ax1.set_title("s = {}".format(s))
        ax1.grid()


    plt.show()
    

    
def gc_order_m():
    """ Keep number of points constant. Increase m.
    Different plots for different orders """
    ms = np.linspace(2, 8, 100)

    gc_params =[ (2, 4/32), (4, 4/16), (8, 4/8), (16, 4/4) ] # (order, element size)
    test_mesh = np.linspace(1, 4, 2000)
    test_vals = func(test_mesh)

    f, sp = plt.subplots(2, 2, sharex='col', sharey='row')
    sp_lin = [ sp[0][0], sp[0][1], sp[1][0], sp[1][1] ]
    
    for i, gc_param in enumerate(gc_params):
        separated = []
        integrated = []
        no_pol = []
    
        order, e_size = gc_param[0], gc_param[1]
        for m in ms:
            print("Working on m =", m)
            in_mesh = GaussChebyshev(order, e_size, 4, 1)
            in_vals = func(in_mesh)
            bf = functools.partial(Gaussian, shape=rescaleBasisfunction(Gaussian, m, in_mesh))

            separated.append(separated_consistent(bf, in_mesh, in_vals))
            integrated.append(integrated_consistent(bf, in_mesh, in_vals))
            no_pol.append(no_pol_consistent(bf, in_mesh, in_vals))

        ax1 = sp_lin[i]
        
        ax1.semilogy(ms, [rmse(i[0](test_mesh), test_vals) for i in separated], "--",
                     label = "RMSE separated Polynomial")
        ax1.semilogy(ms, [rmse(i[0](test_mesh), test_vals) for i in integrated], "--",
                     label = "RMSE integrated polynomial")
        ax1.semilogy(ms, [rmse(i[0](test_mesh), test_vals) for i in no_pol], "--",
                     label = "RMSE no polynomial")

        ax1.set_ylabel("RMSE")
        ax1.set_ylim(10e-6, 10e0)
        ax1.legend(loc = 3)

        ax2 = ax1.twinx()
        ax2.semilogy(ms, [i[2] for i in separated],  label = "Condition separated / no polynomial")
        ax2.semilogy(ms, [i[2] for i in integrated], label = "Condition integrated polynomial")
        ax2.set_ylabel("Condition Number")
        ax2.legend()

        ax1.set_xlabel("m")
        ax1.set_title("Element order = {}, element size = {}, #points = {}".format(order, e_size, len(in_mesh) ))
        ax1.grid()

    plt.show()


def create_BFs(bf, m, mesh):
    BFs = []
    spaces = spacing(mesh)
    for s in spaces:
        BFs.append( functools.partial(bf, shape=rescaleBasisfunction(bf, m, s)) )

    return BFs
def plot_rbf_qr():
    #in_mesh = mesh.GaussChebyshev_2D(12, 1, 4, 1)
    in_mesh = mesh.GaussChebyshev_1D(12, 1, 4, 0)
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
    #plt.xscale("log")
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
    # ax.set_xscale("log")
    surf = ax.plot_surface(np.log10(X), 4 * Y, np.log10(rmse))
    ax.set_xlabel("Shape parameter (log)")
    ax.set_ylabel("Meshsize")
    ax.set_zlabel("RMSE (log)")
    # plt.colorbar(surf)
    plt.show()
def test_rbf_qr_2d():
    X = np.linspace(-5, 5, 10)
    Y = np.linspace(-5, 5, 10)
    in_mesh = np.meshgrid(X, Y)
    def func(mesh):
        return np.sin(mesh[0]) - np.cos(mesh[1])
    in_vals = func(in_mesh)
    X_test = np.linspace(-3, 3, 100)
    Y_test = np.linspace(-3, 3, 100)
    test_mesh = np.meshgrid(X_test, Y_test)
    obj = RBF_QR_2D(1e-3, in_mesh, in_vals)

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(test_mesh[0], test_mesh[1], func(test_mesh))

    fig2 = plt.figure()
    ax2 = fig2.gca(projection="3d")
    ax2.plot_surface(test_mesh[0], test_mesh[1], obj(test_mesh))

    plt.show()
def main():
    # test_rbf_qr_2d()
    # evalShapeQR()
     plot_rbf_qr()
    # set_save_fig_params()
    # plot_basic_consistent()
    # plot_basic_conservative()
    # plot_mesh_sizes()
    # plot_rmse_cond()
    # plot_shape_parameters()
    # gc_m_order()
    # points_s()
    # gc_m_order()
    # neda_poster()
    # plot_supermuc_scaling()
    
if __name__ == "__main__":
    main()
    
