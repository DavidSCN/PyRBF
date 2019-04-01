""" Compares rescaled/non-rescaled variants for different values of m. """

from rbf import *
import basisfunctions, testfunctions
import matplotlib.pyplot as plt, pandas as pd
from numpy.linalg import norm


# tf = testfunctions.Constant(8)
tf = testfunctions.Highfreq()
test_mesh = np.linspace(0.2, 0.8, 10000)

BF = basisfunctions.Gaussian
# BF = basisfunctions.CompactPolynomialC0
# BF = basisfunctions.VolumeSplines
# BF = basisfunctions.ThinPlateSplines

def make_interpolation(m, h):
    in_mesh = np.arange(0, 1, h)
    in_vals = tf(in_mesh)
    
    print("h = ", h, ", m =", m,
          ", Size in_mesh =", len(in_mesh),
          ", Gaussian shape param for m=6 =", Gaussian.shape_param_from_m(6, in_mesh))

    epsilon = BF.shape_param_from_m(m, in_mesh)
    bf = BF(shape_parameter = epsilon)
    sep_interp = SeparatedConsistent(bf, in_mesh, in_vals, rescale = False)
    sep_rinterp = SeparatedConsistent(bf, in_mesh, in_vals, rescale = True)
    interp = NoneConsistent(bf, in_mesh, in_vals, rescale = False)
    rinterp = NoneConsistent(bf, in_mesh, in_vals, rescale = True)

    return {
        "h" : h,
        "epsilon" : epsilon,
        "m" : m,
        "BF" : bf,
        "NonRescaledSep" : norm(sep_interp(test_mesh) - tf(test_mesh), np.inf),
        "RescaledSep" :  norm(sep_rinterp(test_mesh) - tf(test_mesh), np.inf),
        "NonRescaled" : norm(interp(test_mesh) - tf(test_mesh), np.inf),
        "Rescaled" :  norm(rinterp(test_mesh) - tf(test_mesh), np.inf)
    }

    df = pd.DataFrame(results)
    df = df.set_index("epsilon")

    return df


def over_h():
    hs = np.geomspace(0.1, 0.001, 60)
    m = 60
    
    results = []

    for h in hs:
        results.append(make_interpolation(m, h))

    df = pd.DataFrame(results)
    df = df.set_index("h")
    df.to_csv(f"rescaled_{tf}_{BF.__name__}{m}_over_h.csv")

    return df

    
def over_m():
    ms = np.linspace(2, 20, 60) # for Gaussian
    # ms = np.linspace(2, 300, 60) # for CP0

    h = 0.001

    results = []
    
    for m in ms:
        results.append(make_interpolation(m, h))

    df = pd.DataFrame(results)
    df = df.set_index("epsilon")
    df.to_csv(f"rescaled_{tf}_{BF.__name__}_h{h}_over_m.csv")

    return df


# def eval_global_bf():
    # BF = basisfunctions.VolumeSplines
    # print(make_interpolation(0, 0.001))
# BF = basisfunctions.ThinPlateSplines

# df = over_h()
df = over_m()
print(df)

df.plot(y = ["NonRescaledSep", "RescaledSep", "NonRescaled", "Rescaled"], logy = True)
plt.grid()
plt.show()



# in_mesh = np.linspace(0, 1, 1000)
# test_mesh = np.linspace(0, 1, 13123)

# in_vals = tf(in_mesh)
# bf = BF(BF.shape_param_from_m(8, in_mesh))
# interp = SeparatedConsistent(bf, in_mesh, in_vals, rescale = False)
# rinterp = SeparatedConsistent(bf, in_mesh, in_vals, rescale = True)
# one_interp = NoneConsistent(bf, in_mesh, np.ones_like(in_mesh), rescale = False)


# plt.plot(test_mesh, abs(interp(test_mesh) - tf(test_mesh)), label = "Error non")
# plt.plot(test_mesh, abs(rinterp(test_mesh) - tf(test_mesh)), label = "Error resc")

# plt.plot(test_mesh, bf(test_mesh), label = "BF")
# plt.plot(test_mesh, tf(test_mesh), label = "f")
# plt.plot(test_mesh, interp(test_mesh), label = "interp")
# plt.plot(test_mesh, rinterp(test_mesh), label = "R interp")
# plt.plot(test_mesh, one_interp(test_mesh), label = "One")
# plt.plot(in_mesh, np.zeros_like(in_mesh), "x")
# plt.grid()
# plt.legend()
# plt.show()
