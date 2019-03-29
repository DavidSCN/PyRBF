import numpy as np, matplotlib.pyplot as plt, pandas as pd
import rbf, rbf_qr, basisfunctions, testfunctions

norm = lambda x: np.linalg.norm(x, ord = np.inf)

def func(x):
    return np.exp(-np.abs(x - 3)**2) + 2

tf = testfunctions.Highfreq()
# tf = func
BF = basisfunctions.Gaussian

# test_mesh = np.linspace(-0.8, 0.8, 2000)
test_mesh = np.linspace(0.2, 0.8, 4000)

test_vals = tf(test_mesh)

# mesh_sizes = np.linspace(5, 1000, 300, dtype = int)
mesh_sizes = np.geomspace(5, 4000, 300, dtype = int)

res = []

for mesh_size in mesh_sizes:
    in_mesh = np.linspace(0, 1, mesh_size)
    in_vals = tf(in_mesh)

    shape_param = BF.shape_param_from_m(6, in_mesh)
    bf = BF(shape_param)

    interp = rbf.SeparatedConsistent(bf, in_mesh, in_vals)
    rbfqr = rbf_qr.RBF_QR_1D(1e-5, in_mesh, in_vals)

    out = interp(test_mesh)
    outq = rbfqr(test_mesh)

    res.append( { "MeshSize" : mesh_size,
                  "InfError" : norm(out - test_vals),
                  "InfErrorQR" : norm(outq - test_vals),
                  "RMSE" : interp.RMSE(tf, test_mesh),
                  "RMSEQR" : rbfqr.RMSE(tf, test_mesh),
                  "Cond" : interp.condC,
                  "CondQR" : np.linalg.cond(rbfqr.A)})
    


df = pd.DataFrame(res)
df = df.set_index("MeshSize")

df.to_csv("qr_over_meshsize.csv")

fig, ax1 = plt.subplots()
ax1.set_yscale("log")
ax1.set_xscale("log")
ax1.set_ylabel("Error")
ax1.legend(loc = "upper left")
df.plot(ax = ax1, logx = True, logy = True, y = ["InfError", "InfErrorQR", "RMSE", "RMSEQR"])


ax2 = ax1.twinx()
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_ylabel("Condition")
ax2.set_xlabel("Mesh size")
ax2.legend(loc = "upper right")
df.plot(ax = ax2, logx = True, logy = True, y = ["Cond", "CondQR"])


# plt.plot(test_mesh, interp(test_mesh), label = "Normal")
# plt.plot(test_mesh, rbfqr(test_mesh), label = "QR")



# plt.plot(tf(test_mesh) - out, label = "Normal")
# plt.plot(tf(test_mesh) - outq, label = "QR")

plt.legend()
plt.show()
