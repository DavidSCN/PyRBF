import MLS
from RBF import *
from plot_helper import *
from ipdb import set_trace
import matplotlib.pyplot as plt
import math, ipyparallel


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
    GC = False
    mesh_sizes = np.arange(2, 64) if GC else np.linspace(4, 1024, 200)
    # mesh_sizes = np.arange(2, 32) if GC else np.linspace(4, 496, 200)
    test_mesh = np.linspace(1, 4, 4000)
    test_vals = func(test_mesh)
        
    separated = []
    integrated = []
    no_pol = []

    def exec(size, mapping):
        m = 15
        in_mesh = GaussChebyshev(size, 0.25, 4, 1) if GC else np.linspace(1, 4, size)
        in_vals = func(in_mesh)
        shape = rescaleBasisfunction(Gaussian, m, in_mesh)
        print("Computing with in mesh size =", size, ", resulting shape parameter =", shape)
        bf = functools.partial(Gaussian, shape=shape)
        m = mapping(bf, in_mesh, in_vals)
        return m, m.RMSE(func, test_mesh)
        

    separated  = lview.map(lambda size: exec(size, SeparatedConsistent), mesh_sizes)
    integrated = lview.map(lambda size: exec(size, IntegratedConsistent), mesh_sizes)
    no_pol     = lview.map(lambda size: exec(size, NoneConsistent), mesh_sizes)
    lview.wait()
    
    if GC: mesh_sizes = mesh_sizes * 4 * 4 # for GC
    fig, ax1 = plt.subplots()
    print("Computing RMSE")
    ax1.semilogy(mesh_sizes, [i[1] for i in separated], "--",  label = "RMSE separated Polynomial")
    ax1.semilogy(mesh_sizes, [i[1] for i in integrated], "--", label = "RMSE integrated polynomial")
    ax1.semilogy(mesh_sizes, [i[1] for i in no_pol], "--",     label = "RMSE no polynomial")
    
    ax1.set_ylabel("RMSE")
    # ax1.legend(loc = 3)

    ax2 = ax1.twinx()
    print("Computing conditon")
    ax2.semilogy(mesh_sizes, [i[0].condC for i in separated],  label = "Condition separated / no polynomial")
    ax2.semilogy(mesh_sizes, [i[0].condC for i in integrated], label = "Condition integrated polynomial")
    ax2.set_ylabel("Condition Number")
    # ax2.legend(loc = 2, framealpha = 1)

    multi_legend(ax1, ax2, loc = 7)
    
    ax1.set_xlabel("Input Mesh Size")
    # ax1.set_xlabel("Gauss-Chebyshev Order")
    # plt.title("m = " + str(m))

    rm = [i[1] for i in separated]
        
    plt.grid()
    # plt.show()
    plt.savefig("rc-gc-size.pdf")
        
def plot_shape_parameters():
    """ Plots over a range of shape parameters. """
    # in_mesh = np.linspace(1, 4, 192)
    in_mesh = GaussChebyshev(12, 0.25, 4, 1)
    in_vals = func(in_mesh)
    test_mesh = np.linspace(1, 4, 2000)
        
    ms = np.linspace(1, 40, 50)

    separated = []
    integrated = []
    no_pol = []

    for m in ms:
        print("Working on m =", m)
        # bf = create_BFs(Gaussian, m, in_mesh)
        bf = functools.partial(Gaussian, shape=rescaleBasisfunction(Gaussian, m, in_mesh))
        
        separated.append(SeparatedConsistent(bf, in_mesh, in_vals))
        # integrated.append(IntegratedConsistent(bf, in_mesh, in_vals))
        no_pol.append(NoneConsistent(bf, in_mesh, in_vals))

    fig, ax1 = plt.subplots()
    ax1.semilogy(ms, [i.RMSE(func, test_mesh) for i in separated], "--",
                 label = "RMSE separated polynomial")
    # ax1.semilogy(ms, [i.RMSE(func, test_mesh) for i in integrated], "--",
    #              label = "RMSE integrated polynomial")
    ax1.semilogy(ms, [i.RMSE(func, test_mesh) for i in no_pol], "--",
                 label = "RMSE no polynomial")
    
    ax1.set_ylabel("RMSE")
    ax1.set_xlabel("m (included vertices in basis function)")
    
    ax2 = ax1.twinx()
    ax2.semilogy(ms, [i.condC for i in separated],label = "Condition separated / no polynomial")
    # ax2.semilogy(ms, [i.condC for i in integrated], label = "Condition integrated polynomial")
    ax2.set_ylabel("Condition Number")
    
    multi_legend(ax1, ax2, loc = 7)
    plt.grid()
    rm = [i.RMSE(func, test_mesh) for i in separated]
    
    # plt.show()
    plt.savefig("rc-gc-m-rescaled.pdf")
    
       

def plot_basic_consistent():
    # in_mesh = np.concatenate((np.linspace(1, 2, 3), np.linspace(2.1, 4, 9)))
    # print(in_mesh)
    in_mesh = np.linspace(1, 4, 30)
    # in_mesh = GaussChebyshev(12, 1, 4, 1)
    in_vals = func(in_mesh)
    plot_mesh = np.linspace(np.min(in_mesh) - 1, np.max(in_mesh) + 1, 2000) # Use a fine mesh for plotting 

    m = 3
    bf = functools.partial(Gaussian, shape = rescaleBasisfunction(Gaussian, m, in_mesh))
    # bf = create_BFs(Gaussian, m, in_mesh)

    plt.plot(in_mesh, in_vals, "d")
    plt.plot(plot_mesh, func(plot_mesh), "-", label = "Original Function")
    
    sep_consistent = SeparatedConsistent(bf, in_mesh, in_vals)
    plt.plot(plot_mesh, sep_consistent(plot_mesh), "--", label = "Separated Interpolant")
    plt.plot(plot_mesh, sep_consistent.polynomial(plot_mesh), "-", label = "Separated Polynomial")
    plt.plot(plot_mesh, func(plot_mesh) - sep_consistent(plot_mesh), "-", label = "Error separated consistent")
    print("RMSE SeparatedConsistent =", sep_consistent.RMSE(func, plot_mesh))

    none_consistent = NoneConsistent(bf, in_mesh, in_vals)
    plt.plot(plot_mesh, none_consistent(plot_mesh), "--", label = "No Polynomial Interpolant")
    plt.plot(plot_mesh, func(plot_mesh) - none_consistent(plot_mesh), "-", label = "Error none consistent")
    print("RMSE NoneConsistent =", none_consistent.RMSE(func, plot_mesh))
    
    in_consistent = IntegratedConsistent(bf, in_mesh, in_vals)
    plt.plot(plot_mesh, in_consistent(plot_mesh), "--", label = "Integrated Interpolant")
    plt.plot(plot_mesh, in_consistent.polynomial(plot_mesh), "-", label = "Integrated Polynomial")

    plt.legend(loc=2)
    # plt.ylim( np.amin(func(plot_mesh)) * 0.9, np.amax(func(plot_mesh)) * 1.05 )
    plt.title("m = " + str(m))
    plt.grid()
    # plt.savefig("basic.pdf")
    plt.show()


def plot_basic_conservative():
    # print(in_mesh)
    in_mesh = np.linspace(1, 4, 30)
    # in_mesh = GaussChebyshev(12, 1, 4, 1)
    in_vals = func(in_mesh)
    plot_mesh = np.linspace(np.min(in_mesh) - 1, np.max(in_mesh) + 1, 60) # Use a fine mesh for plotting 

    m = 3
    bf = functools.partial(Gaussian, shape = rescaleBasisfunction(Gaussian, m, in_mesh))
    # bf = create_BFs(Gaussian, m, in_mesh)

    plt.plot(in_mesh, in_vals, "d")
    plt.plot(plot_mesh, func(plot_mesh), "-", label = "Original Function")
    
    sep_conservative = SeparatedConservative(bf, in_mesh, in_vals)
    plt.plot(plot_mesh, sep_conservative(plot_mesh), "--", label = "Separated Interpolant")
    plt.plot(plot_mesh, sep_conservative.polynomial(plot_mesh), "-", label = "Separated Polynomial")
    plt.plot(plot_mesh, func(plot_mesh) - sep_conservative(plot_mesh), "-", label = "Error separated consistent")
    print("RMSE SeparatedConsistent =", sep_consistent.RMSE(func, plot_mesh))

    none_conservative = NoneConservative(bf, in_mesh, in_vals)
    plt.plot(plot_mesh, none_conservative(plot_mesh), "--", label = "No Polynomial Interpolant")
    plt.plot(plot_mesh, func(plot_mesh) - none_conservative(plot_mesh), "-", label = "Error none consistent")
    print("RMSE NoneConsistent =", none_conservative.RMSE(func, plot_mesh))
    
    in_conservative = IntegratedConservative(bf, in_mesh, in_vals)
    plt.plot(plot_mesh, in_conservative(plot_mesh), "--", label = "Integrated Interpolant")
    # plt.plot(plot_mesh, in_conservative.polynomial(plot_mesh), "-", label = "Integrated Polynomial")
    plt.plot(plot_mesh, func(plot_mesh) - in_conservative(plot_mesh), "-", label = "Error integrated consistent")

    plt.legend(loc=2)
    # plt.ylim( np.amin(func(plot_mesh)) * 0.9, np.amax(func(plot_mesh)) * 1.05 )
    plt.title("m = " + str(m))
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

    
    for m in ms:
        print("Working on m =", m)
        bf = functools.partial(Gaussian, shape=rescaleBasisfunction(Gaussian, m, in_mesh))
        
        separated.append(SeparatedConsistent(bf, in_mesh, in_vals, False))
        integrated.append(IntegratedConsistent(bf, in_mesh, in_vals))
        no_pol.append(NoneConsistent(bf, in_mesh, in_vals, False))
        separated_res.append(SeparatedConsistent(bf, in_mesh, in_vals, True))
        no_pol_res.append(NoneConsistent(bf, in_mesh, in_vals, True))

    fig, ax1 = plt.subplots()
    ax1.loglog([i.RMSE(func, test_mesh) for i in separated], [i.condC for i in separated], label = "Separated polynomial")
    ax1.loglog([i.RMSE(func, test_mesh) for i in integrated],[i.condC for i in integrated], label = "Integrated polynomial")
    ax1.loglog([i.RMSE(func, test_mesh) for i in no_pol], [i.condC for i in no_pol], label = "RMSE no polynomial")
    
    ax1.loglog([i.RMSE(func, test_mesh) for i in separated_res], [i.condC for i in separated_res], label = "Separated polynomial, rescaled")
    ax1.loglog([i.RMSE(func, test_mesh) for i in no_pol_res], [i.condC for i in no_pol_res], label = "No polynomial, rescaled")
    
    ax1.set_ylabel("Condition")
    ax1.set_xlabel("RMSE")

    ax1.legend()
    plt.grid()
        
    plt.show()
     
    
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
            in_mesh = GaussChebyshev(gc_order, 1, 4, 1)
            in_vals = func(in_mesh)
            # shape = rescaleBasisfunction(Gaussian, m, in_mesh)
            shape = -1
            bf = create_BFs(Gaussian, m, in_mesh)
            # bf = functools.partial(Gaussian, shape=shape)
            
            separated.append(separated_consistent(bf, in_mesh, in_vals))
            integrated.append(integrated_consistent(bf, in_mesh, in_vals))
            no_pol.append(no_pol_consistent(bf, in_mesh, in_vals))

        ax1 = sp_lin[i]

        ax1.semilogy(gc_orders, [rmse(i[0](test_mesh), test_vals) for i in separated], "--",
                     label = "RMSE separated Polynomial")
        ax1.semilogy(gc_orders, [rmse(i[0](test_mesh), test_vals) for i in integrated], "--",
                     label = "RMSE integrated polynomial")
        ax1.semilogy(gc_orders, [rmse(i[0](test_mesh), test_vals) for i in no_pol], "--",
                     label = "RMSE no polynomial")

        ax1.set_ylabel("RMSE")
        # ax1.set_ylim(10e-6, 10e0)
        ax1.legend(loc = 3)

        ax2 = ax1.twinx()
        ax2.semilogy(gc_orders, [i[2] for i in separated],  label = "Condition separated / no polynomial")
        ax2.semilogy(gc_orders, [i[2] for i in integrated], label = "Condition integrated polynomial")
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


rc = ipyparallel.Client()
dview = rc[:]
dview.use_cloudpickle()
lview = rc.load_balanced_view()


def main():
    # set_save_fig_params()

    # plot_basic_consistent()
    # plot_basic_conservative()
    plot_mesh_sizes()
    # plot_rmse_cond()
    # plot_shape_parameters()
    # gc_m_order()
    # points_s()
    # gc_m_order()
    # neda_poster()
    # plot_supermuc_scaling()
    
if __name__ == "__main__":
    main()
    
