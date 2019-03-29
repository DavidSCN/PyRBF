import numpy as np, matplotlib.pyplot as plt, pandas as pd
import rbf, rbf_qr, basisfunctions, testfunctions, mesh
from ipdb import set_trace

norm = lambda x: np.linalg.norm(x, ord = np.inf)

def func(x):
    return np.exp(-np.abs(x - 3)**2) + 2



# tf = testfunctions.Highfreq()
tf = lambda x: testfunctions.Highfreq()(x*0.5 + 0.5)
# tf = func
BF = basisfunctions.Gaussian

test_mesh = np.linspace(-0.6, 0.6, 4000)
# test_mesh = np.linspace(0.2, 0.8, 4000)
test_vals = tf(test_mesh)


orders = np.arange(4, 128, 1)
# element_size = 0.0625

res = []

for order in orders:
    in_mesh = mesh.GaussChebyshev_1D(order = order, element_size = 0.25, domain_size = 2, domain_start = -1)
    # in_mesh = np.polynomial.chebyshev.chebgauss(order)[0]

    in_vals = tf(in_mesh)

    shape_param = BF.shape_param_from_m(6, in_mesh)
    bf = BF(shape_param)

    interp = rbf.SeparatedConsistent(bf, in_mesh, in_vals)
    rbfqr = rbf_qr.RBF_QR_1D(1e-5, in_mesh, in_vals)

    out = interp(test_mesh)
    outq = rbfqr(test_mesh)

    res.append( { "Order" : order,
                  "InfError" : norm(out - test_vals),
                  "InfErrorQR" : norm(outq - test_vals),
                  "RMSE" : interp.RMSE(tf, test_mesh),
                  "RMSEQR" : rbfqr.RMSE(tf, test_mesh),
                  "Cond" : interp.condC,
                  "CondQR" : np.linalg.cond(rbfqr.A)})
    
    # set_trace()


df = pd.DataFrame(res)
df = df.set_index("Order")
print(df)

df.to_csv("qr_over_order.csv")

fig, ax1 = plt.subplots()
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_ylabel("Error")
df.plot(ax = ax1, logx = True, logy = True, y = ["InfError", "InfErrorQR", "RMSE", "RMSEQR"])
# ax1.legend(loc = "upper left")


ax2 = ax1.twinx()
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_ylabel("Condition")
ax2.set_xlabel("Mesh size")
df.plot(ax = ax2, logx = True, logy = True, y = ["Cond", "CondQR"])
ax2.legend(loc = "upper right")


# plt.plot(test_mesh, interp(test_mesh), label = "Normal")
# plt.plot(test_mesh, rbfqr(test_mesh), label = "QR")



# plt.plot(tf(test_mesh) - out, label = "Normal")
# plt.plot(tf(test_mesh) - outq, label = "QR")

plt.legend()
plt.show()
