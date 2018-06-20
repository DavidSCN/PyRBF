import numpy as np, matplotlib.pyplot as plt
from numpy.linalg import norm, inv
from scipy.optimize import fsolve, fminbound
import functools, matplotlib
import RBF

from ipdb import set_trace

eps = 1e-6


np.set_printoptions(precision = 5, linewidth = 150)

def get_RBF_matrix(basisfunction, in_mesh, in_vals):    
    gamma = np.linalg.solve(C, in_vals)

    def interpolant(out_mesh):
        A = eval_BF(out_mesh, in_mesh, basisfunction)
        out_vals = A @ gamma
        return out_vals
        
    return interpolant, None, np.linalg.cond(C)


# def MLS_preconditioner(A, n = 10):
#     # Perform eigen - decomposition
#     delta, X = np.linalg.eigh(A)
#     Delta = np.diag(delta)
#     assert(np.allclose(X @ Delta @ np.linalg.inv(X), A))

#     I = np.identity(len(A))
#     P_old = np.identity(len(A))
#     for k in range(1, n+1):
#         P = P_old @ (np.identity(len(A)) - (Delta @ P_old))
#         P_old = np.copy(P)
#         # print("P =\n", P)

#     P = X @ P @ np.linalg.inv(X)
#     return P


def get_optimal_n(A):
    """ Computes the optimal number of iterations from a nonlinear equation, see paper (21)"""
    EVs = np.linalg.eigh(A)[0]
    EV_max, EV_min = np.max(EVs), np.min(EVs)
    assert(0 < EV_min and EV_max < 1)
    assert(norm(np.identity(len(A))-A, 2) < 1)
    print("EV_max =", EV_max, "EV_min =", EV_min)

    func = lambda n: (1-np.power(1-EV_max, 2*n)) / (1-np.power(1-EV_min, 2*n)) - np.sqrt(EV_max / EV_min)
    result = fminbound(func, x1 = 1, x2 = 10000, full_output = True)


def printMtxProps(A, prefix = ""):
    if prefix: prefix = prefix + " "
    I = np.identity(len(A))
    EVs = np.real(np.linalg.eigh(A)[0])
    SVs = np.real(np.linalg.svd(A)[1])
    SV_max, SV_min = np.max(SVs), np.min(SVs)
    EV_max, EV_min = np.max(EVs), np.min(EVs)
    print(prefix + "max(EV) =", EV_max, " min(EV) =", EV_min)
    print(prefix + "max(SV) =", SV_max, " min(SV) =", SV_min)
    print(prefix + "cond(A) =", np.linalg.cond(A))
    print(prefix + "||I-A||_2  =", norm(I-A, 2)) # Should be < 1 according to 5.3

    return {"maxEV": EV_max, "minEV": EV_min,
            "maxSV": SV_max, "minSV": SV_min}
    

def MLS_PC(A, n = 60):
    I = np.identity(len(A))
    props = printMtxProps(A)    
    # set_trace()
    scale_factor = 1 / (1.1 * props["maxEV"])
    Aunscaled = A.copy()
    A = A * scale_factor
    # A = A*inv(diag(A)) # rhs ebenfalls diag^-1
        
    # assert(norm(I-A, 2) <= 1)

    props = printMtxProps(A, "Scaled")

    assert(0-eps < props["minEV"] and props["maxEV"] < 1+eps)
   
    P = I
    conds = []
    for k in range(1, n+1):
        P = P @ (2*I - A @ P)
        conds.append(np.linalg.cond(A@P))
        print("PC Iteration", k, " condition = ", conds[-1])

    # plt.semilogy(range(1, n+1), conds)
    # plt.grid()
    # plt.show()
    print("Scale Factor =", scale_factor)
    return P, A


if __name__ == "__main__":
    N = 500
    a = np.random.randint(0,100, size=(N,N))
    A = np.tril(a) + np.tril(a, -1).T

    # in_mesh = np.linspace(1, 4, N)
    in_mesh = RBF.GaussChebyshev(24, 1, 4, 1)
    basisfunction = functools.partial(RBF.Gaussian, shape=RBF.rescaleBasisfunction(RBF.Gaussian, 10, in_mesh))
    A = RBF.eval_BF(in_mesh, in_mesh, basisfunction)

    I = np.identity(len(A))
    print("max(EV(A))   =", np.max(np.linalg.eig(A)[0]))
    print("EV(A) > 0    =", np.all(np.linalg.eig(A)[0] > 0) )
    print("||I-A||_2    =", norm(I-A, 2)) # Should be < 1 according to 5.3
    
    P, A = MLS_PC(A)

    print()
    I = np.identity(len(A))
    print("||A^-1 - P|| =", norm(inv(A) - P))
    print("||A - P||    =", norm(A - P))
    print("cond(A)      =", np.linalg.cond(A)) # cond(A) == cond(P) for n->oo
    print("cond(P)      =", np.linalg.cond(P))
    print("cond(P^-1 A) =", np.linalg.cond(inv(P) @ A)) # Wikipedia
    print("cond(A P)    =", np.linalg.cond(A @ P)) # Algorithm 1, should be 1 for n->oo, according to 5.3
    print("cond(P A)    =", np.linalg.cond(P @ A)) # Algorithm 1, should be 1 for n->oo, according to 5.3

    # set_trace()
    fig, ax = plt.subplots(2,2)
    ax = ax.flat
    ax[0].matshow(A)
    ax[0].set_title("A")
    ax[1].matshow(P)
    ax[1].set_title("P")
    ax[2].matshow(A @ P)
    ax[2].set_title("A @ P")
    ax[3].matshow(P @ A)
    ax[3].set_title("P @ A")
        
    plt.show()
