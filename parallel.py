import ipyparallel as ipp, numpy as np, matplotlib.pyplot as plt
from RBF import *
# import RBF
# from ipdb import set_trace


def plot_shape_parameters():
    in_mesh = np.linspace(1, 4, 192)
    in_vals = func(in_mesh)
    test_mesh = np.linspace(1, 4, 2000)
        
    ms = np.linspace(1, 40, 50)

    separated = []
    integrated = []
    no_pol = []

    BFs = [functools.partial(Gaussian, shape=rescaleBasisfunction(Gaussian, m, in_mesh)) for m in ms]
    
    # separated = [RBF.SeparatedConsistent(x, in_mesh, in_vals) for x in BFs]

    separated = lview.map(lambda x: SeparatedConsistent(x, in_mesh, in_vals), BFs)
    separated.wait()
    
    fig, ax1 = plt.subplots()
    ax1.semilogy(ms, [i.RMSE(func, test_mesh) for i in separated], "--", label = "RMSE")
    
    ax1.set_ylabel("RMSE")
    ax1.legend(loc = 3)

    ax2 = ax1.twinx()
    ax2.semilogy(ms, [i.condC for i in separated],label = "Condition")
    ax2.set_ylabel("Condition Number")
    ax2.legend(loc = 2)
    
    ax1.set_xlabel("m (included vertices in basis function)")
    plt.grid()
    
    plt.show()


rc = ipp.Client()
lview = rc.load_balanced_view()
dview = rc[:]
dview.use_cloudpickle()
# dview.execute("import numpy as np")
# dview.execute("import RBF")
# dview.push({"SeparatedConsistent" : SeparatedConsistent})


def main():
        
    plot_shape_parameters()

    
if __name__ == "__main__":
    main()
