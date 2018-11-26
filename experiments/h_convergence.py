""" Plots RMSE over mesh density h (number of data sites). """

import concurrent.futures, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import basisfunctions, rbf, testfunctions


def kernel(args):
    mesh_size, RBF, bf, testfunction, m = args
    in_mesh = np.linspace(0, 1, num = int(mesh_size))
    test_mesh = np.linspace(0.1, 0.9, 50000)

    in_vals = testfunction(in_mesh)
    b = bf.shaped(m, in_mesh)
    
    interp = RBF(b, in_mesh, in_vals, rescale=False)
    print(interp, testfunction, bf, "mesh_size = ", mesh_size, "m =", m)
        
    return { "h" : 1 / mesh_size,
            "RBF" : str(interp),
             "BF" : str(bf),
             "RMSE" : interp.RMSE(testfunction, test_mesh),
             "InfError" : np.linalg.norm(interp.error(testfunction, test_mesh), ord=np.inf),
             "ConditionC" : interp.condC,
             "Testfunction" : str(testfunction),
             "m" : m if bf.has_shape_param else 0}


def main():
    # mesh_sizes = np.linspace(10, 5000, num = 50)
    # mesh_sizes = np.linspace(10, 200, num = 2)
    mesh_sizes = np.logspace(10, 14, base = 2, num = 40) # = [1024, 16384]

    
    bfs = [basisfunctions.Gaussian(), basisfunctions.ThinPlateSplines(),
           basisfunctions.VolumeSplines(), basisfunctions.MultiQuadrics()]
    RBFs = [rbf.NoneConsistent, rbf.SeparatedConsistent]
    tfs = [testfunctions.highfreq(), testfunctions.lowfreq(), testfunctions.jump()]
    ms = [4, 6, 8, 10, 14]
    
    params = []

    for mesh_size, RBF, bf, tf, m in itertools.product(mesh_sizes, RBFs, bfs, tfs, ms):
        if (not bf.has_shape_param) and (m != ms[0]):
            # skip iteration when the function has no shape parameter and it's not the first iteration in m
            continue

        params.append((mesh_size, RBF, bf, tf, m))


    with concurrent.futures.ProcessPoolExecutor(max_workers = 10) as executor:
        result = executor.map(kernel, params)
    # for p in params:
        # kernel(**p)


    df = pd.DataFrame(list(result))
    df = df.set_index("h")

    df.to_pickle("h_convergence.pkl")
    df.to_csv("h_convergence.csv")

    for name, group in df.groupby(["RBF", "BF", "Testfunction", "m"]):
        group.to_csv("h_convergence_" + "_".join(str(g) for g in name) + ".csv")

    print(df)


if __name__ == "__main__":
    main()
