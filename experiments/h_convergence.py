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
    workers = 10
    writeCSV = True
    
    # mesh_sizes = np.linspace(10, 5000, num = 50)
    # mesh_sizes = np.linspace(10, 200, num = 2, dtype = int)
    # mesh_sizes = np.logspace(10, 14, base = 2, num = 40) # = [1024, 16384]

    # Output of np..geomspace(100, 15000, num = 40), not available at neon
    mesh_sizes = np.array([  100.        ,   113.70962094,   129.29877894,   147.02515141,
                             167.18174235,   190.1017255 ,   216.16395146,   245.79920981,
                             279.49734974,   317.81537692,   361.38666038,   410.93140164,
                             467.26853912,   531.32928459,   604.17251544,   687.00227711,
                             781.18768514,   888.28555558,  1010.0661381 ,  1148.54237685,
                             1306.00318302,  1485.05126885,  1688.64616853,  1920.15315722,
                             2183.39887649,  2482.73458601,  2823.10808664,  3210.14550398,
                             3650.24428412,  4150.67893877,  4719.72128761,  5366.77718545,
                             6102.54199414,  6939.17736909,  7890.51228258,  8972.27160655,
                             10202.3360333 , 11601.03763024, 13191.49591417, 15000.        ])


    # mesh_sizes = np.array([100, 200])

    bfs = [basisfunctions.Gaussian, basisfunctions.ThinPlateSplines,
           basisfunctions.VolumeSplines, basisfunctions.MultiQuadrics,
           basisfunctions.CompactPolynomialC0, basisfunctions.CompactThinPlateSplineC2]
    RBFs = [rbf.NoneConsistent, rbf.SeparatedConsistent]
    tfs = [testfunctions.Highfreq(), testfunctions.Lowfreq(), testfunctions.Jump(), testfunctions.Constant(1)]
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

    print("Size of parameter space =", len(params))
    print()
    
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
