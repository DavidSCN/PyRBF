#!/usr/bin/env python3
""" Evaluate RBFs / Basisfunctions / Testfunctions over mesh density h. """

import datetime, concurrent.futures, itertools, multiprocessing
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import basisfunctions, rbf, testfunctions

# from ipdb import set_trace

def filter_existing(input_set, df):
    output_set = []
    for i in input_set:
        exists = (
            (df["MeshSize"] == i["mesh_size"]) &
            (df["RBF"] == i["RBF"].__name__) &
            (df["BF"] == i["basisfunction"].__name__) &
            (df["Testfunction"] == str(i["testfunction"])) &
            (df["m"] == i["m"] if not np.isnan(i["m"]) else True) # comparing np.NaN is not possible
        ).any()
        if not exists:
            output_set.append(i)

    print("Filtered from", len(input_set), "to", len(output_set))
    return output_set


def unpack_args_wrapper(args):
    """ Wrapper to perform dictionary argument unpacking. """
    return kernel(**args)


def kernel(mesh_size, RBF, basisfunction, testfunction, m):
    global runCounter, runTotal
    with runCounter.get_lock():
        runCounter.value += 1

    in_mesh = np.linspace(0, 1, num = mesh_size)
    test_mesh = np.linspace(0.16, 0.84, 40000) # padding of 0.16

    in_vals = testfunction(in_mesh)
    bf = basisfunction(shape_parameter = basisfunction.shape_param_from_m(m, in_mesh))

    print("{datetime}: ({runCounter} / {runTotal}): {interp}, {testfunction}, {b}, mesh size = {mesh_size}, m = {m}".format(
        datetime = str(datetime.datetime.now()),
        runCounter = runCounter.value, runTotal = runTotal, interp = RBF.__name__,
        testfunction = testfunction, b = bf, mesh_size = mesh_size, m = m))

    try:
        interp = RBF(bf, in_mesh, in_vals, rescale=False)
        error = interp(test_mesh) - testfunction(test_mesh)
        condC = interp.condC
    except np.linalg.LinAlgError as e:
        print(e)
        interp = RBF.__name__ # if exception is raised in interp ctor
        error = np.full_like(test_mesh, np.NaN)
        condC = np.NaN

    return { "h" : 1 / mesh_size,
             "MeshSize" : mesh_size,
             "RBF" : str(interp),
             "BF" : str(bf),
             "RMSE" : np.sqrt((error ** 2).mean()),
             "InfError" : np.linalg.norm(error, ord=np.inf),
             "ConditionC" : condC,
             "Testfunction" : str(testfunction),
             "m" : m}


def write(df, writeCSV):
    df.to_pickle("h_convergence.pkl")

    if writeCSV:
        df2 = df.fillna({"m" : 0}) # replace NaN in column m with zero, because groupby drops NaNs
        df2.to_csv("h_convergence.csv")
        for name, group in df2.groupby(["RBF", "BF", "Testfunction", "m"]):
            group.to_csv("h_convergence_" + "_".join(str(g) for g in name) + ".csv")


def main():
    parallel = True
    workers = 8
    writeCSV = True
    chunk_size = 30
    
    # mesh_sizes = np.linspace(10, 5000, num = 50)
    # mesh_sizes = np.linspace(10, 200, num = 2, dtype = int)
    # mesh_sizes = np.logspace(10, 14, base = 2, num = 40) # = [1024, 16384]

    # Output of np.geomspace(100, 15000, num = 40, dtype = int) , not available at neon
    mesh_sizes = np.array([  100,   113,   129,   147,   167,   190,   216,   245,   279,
                             317,   361,   410,   467,   531,   604,   687,   781,   888,
                             1010,  1148,  1306,  1485,  1688,  1920,  2183,  2482,  2823,
                             3210,  3650,  4150,  4719,  5366,  6102,  6939,  7890,  8972,
                             10202, 11601, 13191, 15000])


    # mesh_sizes = np.array([100, 200])

    bfs = [basisfunctions.Gaussian, basisfunctions.ThinPlateSplines,
           basisfunctions.VolumeSplines, basisfunctions.MultiQuadrics,
           basisfunctions.CompactPolynomialC0, basisfunctions.CompactThinPlateSplineC2]
    RBFs = [rbf.NoneConsistent, rbf.SeparatedConsistent]
    tfs = [testfunctions.Highfreq(), testfunctions.Lowfreq(), testfunctions.Jump(), testfunctions.Constant(1)]
    ms = [4, 6, 8, 12, 16]

    print("Minimum padding needed to avoid boundary effects =", 1/np.min(mesh_sizes) * max(ms))
    
    params = []
    for mesh_size, RBF, bf, tf, m in itertools.product(mesh_sizes, RBFs, bfs, tfs, ms):
        if (not bf.has_shape_param) and (m != ms[0]):
            # skip iteration when the function has no shape parameter and it's not the first iteration in m
            continue

        if not bf.has_shape_param:
            m = np.NaN

        params.append({"mesh_size" : mesh_size, "RBF" : RBF, "basisfunction" : bf, "testfunction" : tf, "m" : m})


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

    try:
        df = pd.read_pickle("h_convergence.pkl")
        params = filter_existing(params, df)
    except FileNotFoundError:
        df = pd.DataFrame()        
    
    global runTotal
    runTotal = len(params)

    global runCounter
    runCounter = multiprocessing.Value("i", 0) # i is type id for integer
    
    chunks = [params[i:i + chunk_size] for i in range(0, len(params), chunk_size)]

    for chunk in chunks:
        results = []
        
        if parallel:
            with concurrent.futures.ProcessPoolExecutor(max_workers = workers) as executor:
                results = executor.map(unpack_args_wrapper, chunk)
        else:
            for p in chunk:
                results.append(kernel(**p))

        print("Chunk computed, writing...")

        df = df.append(pd.DataFrame(list(results)).set_index("h"))
        write(df, writeCSV)

    write(df, writeCSV) # write out, also if we filtered everything
    print(df)


if __name__ == "__main__":
    main()
