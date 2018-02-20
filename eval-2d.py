from rbf import *
from plot_helper import *
from mesh import GaussChebyshev_1D
from ipdb import set_trace
import matplotlib.pyplot as plt
import math
import mesh


def func_2d(x):
    res = []
    for i in x:
        res.append(i[0] + i[1])
    return np.array(res)

def plot_basic_conservative():
    # set_trace()
    # func = lambda x: np.power(np.sin(5*x), 2) + np.exp(x/2)
    func = lambda x: x[0] + x[1]
    # func = lambda x: np.full_like(x, 4)
    # func = lambda x: x
    
    in_mesh = np.array([ [0,0], [1,0], [1,1], [0,1] ])
    out_mesh = np.array([ [0.4,0], [6,6], [7,7] ])
        
    m = 6
    bf = functools.partial(Gaussian(), shape = 1)

    in_vals = np.array([1, 1, 1, 1])
    
    none_conservative = NoneConservative(bf, in_mesh, in_vals)
    # plt.plot(out_mesh, none_conservative(out_mesh), "g-", label = "No Polynomial Interpolant")
    # plt.plot(out_mesh, none_conservative(out_mesh) - func(out_mesh), "g:", label = "Error none conservative")
    # plt.plot(out_mesh, none_conservative.weighted_error(func, out_mesh), "g-.", label = "Weighted Error none conservative")
    
    # print("RMSE NoneConservative =", none_conservative.RMSE(func, out_mesh))
    print("none out_vals =", none_conservative(out_mesh))

    # none_conservative_resc = NoneConservative(bf, in_mesh, in_vals, True) 
    # plt.plot(out_mesh, none_conservative_resc(out_mesh), "c-", label = "No Polynomial Rescaled Interpolant")
    # plt.plot(out_mesh, func(out_mesh) - none_conservative_resc(out_mesh), "c:", label = "Error none conservative rescaled")
    # plt.plot(out_mesh, none_conservative_resc.weighted_error(func, out_mesh), "c-.", label = "Weighted Error none conservative rescaled")
    
    # plt.plot(out_mesh, none_conservative_resc.rescalingInterpolant(out_mesh), "c-", label = "No Polynomial One Interpolant")
    # print("RMSE NoneConservative Rescaled =", none_conservative_resc.RMSE(func, out_mesh))
    
    in_conservative = IntegratedConservative(bf, in_mesh, in_vals, False)
    print("integrated out_vals =", in_conservative(out_mesh))

    # plt.plot(out_mesh, in_conservative(out_mesh), "m-", label = "Integrated Interpolant")
    # plt.plot(out_mesh, in_conservative(out_mesh) - func(out_mesh), "m:", label = "Error integrated conservative")
    # plt.plot(out_mesh, in_conservative.weighted_error(func, out_mesh), "m-.", label = "Weighted Error integrated conservative")

    sep_conservative = SeparatedConservative(bf, in_mesh, in_vals)
    print("separated out_vals =", sep_conservative(out_mesh))
    
    # plt.plot(out_mesh, sep_conservative(out_mesh), "b-", label = "Separated Interpolant")
    # plt.plot(out_mesh, sep_conservative(out_mesh) - func(out_mesh), "b:", label = "Error seperated conservative")
    # plt.plot(out_mesh, sep_conservative.weighted_error(func, out_mesh), "b-.", label = "Weighted Error separated conservative")
    # plt.plot(out_mesh, sep_conservative.rescaled_error(func, out_mesh), "b--", label = "Scaled Error separated conservative")
    
    
    # # plt.plot(out_mesh, in_conservative.polynomial(out_mesh), "b-.", label = "Integrated Polynomial")
    # print("RMSE InConservative =", in_conservative.RMSE(func, out_mesh))
    
    # plt.legend(loc=2)
    # plt.ylim( np.amin(func(plot_mesh)) * 0.9, np.amax(func(plot_mesh)) * 1.05 )
    # plt.title("m = " + str(m))
    # plt.grid()
    # plt.savefig("basic.pdf")
    # plt.show()


plot_basic_conservative()
