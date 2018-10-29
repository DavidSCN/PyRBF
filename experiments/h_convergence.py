""" Plots RMSE over mesh density h (number of data sites). """

import concurrent.futures, itertools
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import basisfunctions, rbf, testfunctions
from ipdb import set_trace

rescale = False

# mesh_sizes = np.linspace(10, 5000, num = 50)
mesh_sizes = np.linspace(10, 50, num = 2)
basisfunctions = [basisfunctions.Gaussian(), basisfunctions.ThinPlateSplines(),
                  basisfunctions.VolumeSplines(), basisfunctions.MultiQuadrics()]
RBFs = [rbf.NoneConsistent, rbf.SeparatedConsistent]
testfunctions = [testfunctions.highfreq(), testfunctions.lowfreq(), testfunctions.jump()]
ms = [4, 6, 8, 10, 14]


def kernel(mesh_size, RBF, bf, f, m):
    in_mesh = np.linspace(0, 1, num = int(mesh_size))
    test_mesh = np.linspace(0.1, 0.9, 50000)

    in_vals = f(in_mesh)
    b = bf.shaped(m, in_mesh)
    
    interp = RBF(b, in_mesh, in_vals, rescale)
    print(interp, f, bf, "mesh_size = ", mesh_size, "m =", m)
        
    return { "h" : 1 / mesh_size,
            "RBF" : str(interp),
             "BF" : str(bf),
             "RMSE" : interp.RMSE(f, test_mesh),
             "InfError" : interp.error(f, test_mesh),
             "ConditionC" : interp.condC,
             "Testfunction" : str(f),
             "m" : m if bf.has_shape_param else 0}


def main():
    params = []

    for mesh_size, RBF, bf, f, m in itertools.product(mesh_sizes, RBFs, basisfunctions, testfunctions, ms):
        if (not bf.has_shape_param) and (m != ms[0]):
            # skip iteration when the function has no shape parameter and it's not the first iteration
            continue

        params.append({"mesh_size" : mesh_size,
                       "RBF" : RBF,
                       "bf" : bf,
                       "f" : f,
                       "m" : m})


    with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as executor:
        print("Exec")
        result = executor.map(lambda a: kernel(**a), params)
    # for p in params:
        # kernel(**p)

    
    # set_trace()

    df = pd.DataFrame(result)
    df = df.set_index("h")

    df.to_pickle("h_convergence_full.pkl")
    df.to_csv("h_convergence_full.csv")
    set_trace()

    for name, group in df.groupby(["RBF", "BF", "Testfunction", "m"]):
        group.to_csv("h_convergence_" + "_".join(str(g) for g in name))
        # print("h_convergence_" + "_".join(str(g) for g in name) + ".csv")

    print(df)


if __name__ == "__main__":
    main()
