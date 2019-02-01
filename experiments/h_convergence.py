""" Plots RMSE over mesh density h (number of data sites). """

import concurrent.futures, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import basisfunctions, rbf, testfunctions


def kernel(args):
    mesh_size, RBF, bf, testfunction, m = args
    in_mesh = np.linspace(0, 1, num = int(mesh_size))
    test_mesh = np.linspace(0.1, 0.9, 50000)

    in_vals = testfunction(in_mesh)
    b = bf(shape_parameter = bf.shape_param_from_m(m, in_mesh))
    
    interp = RBF(b, in_mesh, in_vals, rescale=False)

    # Compute the error here, for performance reasons
    error = interp(test_mesh) - testfunction(test_mesh)
    
    print(interp, testfunction, b, "mesh_size =", mesh_size, "m =", m)
        
    return { "h" : 1 / mesh_size,
             "RBF" : str(interp),
             "BF" : str(b),
             "RMSE" : np.sqrt((error ** 2).mean()),
             "InfError" : np.linalg.norm(error, ord=np.inf),
             "ConditionC" : interp.condC,
             "Testfunction" : str(testfunction),
             "m" : m}


def main():
    parallel = True
    workers = 2
    writeCSV = False
    
    # mesh_sizes = np.linspace(10, 5000, num = 50)
    # mesh_sizes = np.linspace(10, 200, num = 2, dtype = int)
    # mesh_sizes = np.logspace(10, 14, base = 2, num = 40) # = [1024, 16384]
    # mesh_sizes = np.geomspace(100, 15000, num = 10)

    mesh_sizes = np.array([100, 200, 300])

    bfs = [basisfunctions.Gaussian, basisfunctions.ThinPlateSplines,
           basisfunctions.VolumeSplines, basisfunctions.MultiQuadrics,
           basisfunctions.CompactPolynomialC0, basisfunctions.CompactThinPlateSplineC2]
    RBFs = [rbf.NoneConsistent, rbf.SeparatedConsistent]
    tfs = [testfunctions.highfreq(), testfunctions.lowfreq(), testfunctions.jump()]
    ms = [4, 6, 8, 10, 14]

    params = []
    for mesh_size, RBF, bf, tf, m in itertools.product(mesh_sizes, RBFs, bfs, tfs, ms):
        if (not bf.has_shape_param) and (m != ms[0]):
            # skip iteration when the function has no shape parameter and it's not the first iteration in m
            continue

        params.append((mesh_size, RBF, bf, tf, m))


    # params = itertools.chain(
        # itertools.product(mesh_sizes, RBFs,
        #                   [basisfunctions.ThinPlateSplines, basisfunctions.VolumeSplines],
        #                   tfs, [0]),
        # itertools.product(mesh_sizes, RBFs,
        #                   [basisfunctions.Gaussian],
        #                   tfs, [4, 6, 8, 10, 14]),
        # itertools.product(mesh_sizes, RBFs,
                          # [basisfunctions.MultiQuadrics],
                          # tfs, [0.1, 0.5, 1, 1.5])
    # )

    if parallel:
        with concurrent.futures.ProcessPoolExecutor(max_workers = workers) as executor:
            result = executor.map(kernel, params)
    else:
        result = []
        for p in params:
            result.append(kernel(p))
        
    df = pd.DataFrame(list(result))
    df = df.set_index("h")

    df.to_pickle("h_convergence.pkl")

    if writeCSV:
        df.to_csv("h_convergence.csv")
        for name, group in df.groupby(["RBF", "BF", "Testfunction", "m"]):
            group.to_csv("h_convergence_" + "_".join(str(g) for g in name) + ".csv")

    print(df)


if __name__ == "__main__":
    main()
