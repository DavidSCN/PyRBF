from basisfunctions import *

from ipdb import set_trace
import functools
import numpy as np
import scipy.sparse.linalg
import scipy.spatial

dimension = 1
rescaled = False
func = lambda x: np.power(np.sin(5*x), 2) + np.exp(x/2) # auch mit reinnehmen, als akustisches Beispiel
# func = functools.partial(lambda x: Gaussian(x-1, 1) + 2)
# heaviside = lambda x: 0 if x < 2 else 1
# func = np.vectorize(heaviside)
# func = lambda x: x


def coordify(array):
    """ Changes [a, b, c] to [ [a], [b], [c] ]"""
    return array[:, np.newaxis] if array.ndim == 1 else array
    # return np.atleast_2d(array).T
           


def spacing(a):
    """ Returns spaces around vertices """
    spaces = np.zeros_like(a).astype("float")
    for i, e in enumerate(a[1:-1], start=1):
        spaces[i] = (a[i+1] - a[i-1]) / 2.0

    spaces[0] = a[1] - a[0]
    spaces[-1] = a[-1] - a[-2]

    return spaces


class RBF:
    def RMSE(self, func, test_mesh):
        """ Returns the root mean squared error. """
        targets = func(test_mesh)
        predictions = self(test_mesh)
        return np.sqrt(((predictions - targets) ** 2).mean())

    def error(self, func, test_mesh):
        targets = func(test_mesh)
        predictions = self(test_mesh)
        return np.linalg.norm(predictions - targets, ord=np.inf)

    def weighted_error(self, func, out_mesh):
        """ Weighted error to get a better error for conservative interpolation. """
        print("Weighted error factor =", len(out_mesh) / len(self.in_mesh))
        return self(out_mesh) * len(out_mesh) / len(self.in_mesh) - func(out_mesh)

    def rescaled_error(self, func, out_mesh):
        g = NoneConservative(self.basisfunction, self.in_mesh, np.ones_like(self.in_vals), False)
        return self(out_mesh) / g(out_mesh) - func(out_mesh)


    def eval_BF(self, meshA, meshB):
        """ Evaluates single BF or list of BFs on the meshes. """
        if type(self.basisfunction) is list:
            A = np.empty((len(meshA), len(meshB)))
            # for i, row in enumerate(meshA):
                # for j, col in enumerate(meshB):
                    # A[i, j] = self.basisfunction[j](row - col)
            for j, col in enumerate(meshB):
                A[:,j] = self.basisfunction[j](meshA - col)
        else:
            # mgrid = np.meshgrid(meshB, meshA)
            # A = self.basisfunction( np.abs(mgrid[0] - mgrid[1]) )

            # A = np.zeros([len(meshA), len(meshB)])
            # for i, x in enumerate(meshA):
            #     print(i)
            #     for j, y in enumerate(meshB):
            #         A[i, j] = self.basisfunction(np.linalg.norm(x-y))
            if meshA.ndim == 1:
                meshA = meshA[:, np.newaxis]
            if meshB.ndim == 1:
                meshB = meshB[:, np.newaxis]
            A = scipy.spatial.distance_matrix(meshA, meshB)
            A = self.basisfunction(A)
            
        return A

    def polynomial(self, out_mesh):
        raise NotImplementedError

    @property
    def condC(self):
        cond = getattr(self, "_condC", np.linalg.cond(self.C))
        self._condC = cond
        return cond


    
class SeparatedConsistent(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = rescaled):
        in_mesh = coordify(in_mesh)
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        Q = np.zeros( [in_mesh.shape[0], in_mesh.shape[1] + 1] )
        Q[:, 0] = 1
        Q[:,1:] = in_mesh
        
        lsqrRes  = scipy.sparse.linalg.lsqr(Q, in_vals)
        self.beta = lsqrRes[0]

        self.C = self.eval_BF(in_mesh, in_mesh)
        rhs = in_vals - Q @ self.beta
        self.gamma = np.linalg.solve(self.C, rhs)

        self.rescaled = rescale
        if rescale:
            self.gamma_rescaled = np.linalg.solve(self.C, np.ones_like(self.gamma))
                
    def polynomial(self, out_mesh):
        return self.beta[0] + self.beta[1] * out_mesh

    def __call__(self, out_mesh):
        out_mesh = coordify(out_mesh)
        A = self.eval_BF(out_mesh, self.in_mesh)
        V = np.zeros( [out_mesh.shape[0], out_mesh.shape[1] +1 ])
        V[:, 0] = 1
        V[:, 1:] = out_mesh

        out_vals = A @ self.gamma
        if self.rescaled:
            out_vals = out_vals / (A @ self.gamma_rescaled)

        return out_vals + V @ self.beta

class SeparatedConsistentFitted(RBF):        
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = rescaled, degree = 2):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        self.poly = np.poly1d(np.polyfit(in_mesh, in_vals, degree))

        self.C = self.eval_BF(in_mesh, in_mesh)
        rhs = in_vals - self.poly(in_mesh)
        self.gamma = np.linalg.solve(self.C, rhs)
        
        self.rescaled = rescale
        if rescale:
            self.gamma_rescaled = np.linalg.solve(self.C, np.ones_like(in_mesh))
            
        
    def polynomial(self, out_mesh):
        return self.poly(out_mesh)

    def __call__(self, out_mesh):
        A = self.eval_BF(out_mesh, self.in_mesh)
        out_vals = A @ self.gamma
        
        if self.rescaled:
            out_vals = out_vals / (A @ self.gamma_rescaled)

        return out_vals + self.poly(out_mesh)



class IntegratedConsistent(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        polyparams = dimension + 1
        C = np.zeros( [len(in_mesh)+polyparams, len(in_mesh)+polyparams] )

        C[0, polyparams:] = 1
        C[1, polyparams:] = in_mesh
        C[polyparams:, 0] = 1
        C[polyparams:, 1] = in_mesh

        C[polyparams:, polyparams:] = self.eval_BF(in_mesh, in_mesh)
    
        invec = np.hstack((np.zeros(polyparams), in_vals))
        self.p = np.linalg.solve(C, invec)
        self.C = C
        
    def polynomial(self, out_mesh):
        return self.p[0] + self.p[1] * out_mesh

    def __call__(self, out_mesh):
        A = self.eval_BF(out_mesh, self.in_mesh)

        V = np.zeros( [len(out_mesh), dimension + 1 ])
        V[:, 0] = 1
        V[:, 1] = out_mesh

        VA = np.concatenate((V,A), axis=1)
        return VA @ self.p

    

class NoneConsistent(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = rescaled):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        self.C = self.eval_BF(in_mesh, in_mesh)
        self.gamma = np.linalg.solve(self.C, in_vals)
        
        self.rescaled = rescale
        if rescale:
            self.gamma_rescaled = np.linalg.solve(self.C, np.ones_like(in_mesh))


    def __call__(self, out_mesh):
        A = self.eval_BF(out_mesh, self.in_mesh)
        out_vals = A @ self.gamma
        if self.rescaled:
            out_vals = out_vals / (A @ self.gamma_rescaled)
        return out_vals


class NoneConservative(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = rescaled):
        self.basisfunction, self.in_mesh, self.in_vals, self.rescale = basisfunction, in_mesh, in_vals, rescale
        
    def __call__(self, out_mesh):
        self.C = self.eval_BF(out_mesh, out_mesh)
        A = self.eval_BF(self.in_mesh, out_mesh)

        au = A.T @ self.in_vals
        self.out_vals = np.linalg.solve(self.C, au)

        if self.rescale:
            self.au_rescaled = A.T @ np.ones_like(self.in_vals)
            self.rescaled_interp = np.linalg.solve(C, self.au_rescaled)
            self.rescaled_interp = self.rescaled_interp + (1 - np.mean(self.rescaled_interp))
            self.out_vals = self.out_vals / self.rescaled_interp
                    
        print("Conservativeness None Delta =", np.sum(self.out_vals) - np.sum(self.in_vals),
              ", rescaling =", self.rescale)
        return self.out_vals


    def rescalingInterpolant(self, out_mesh):
        try:
            return self.rescaled_interp
        except NameError:
            print("Rescaling not available.")
            return np.ones_like(out_mesh)

    
class IntegratedConservative(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = rescaled):
        self.basisfunction, self.in_vals, self.rescale = basisfunction, in_vals, rescale
        self.in_mesh = coordify(in_mesh)
        
    def __call__(self, out_mesh):
        # set_trace()
        out_mesh = coordify(out_mesh)
        dimension = len(out_mesh[0])
        polyparams = dimension + 1
        # polyparams = 0
        C = np.zeros( [len(out_mesh)+polyparams, len(out_mesh)+polyparams] )
        C[0, polyparams:] = 1
        C[1:polyparams, polyparams:] = out_mesh.T
        C[polyparams:, 0] = 1
        C[polyparams:, 1:polyparams] = out_mesh
        C[polyparams:, polyparams:] = self.eval_BF(out_mesh, out_mesh)

        A = np.zeros( [len(self.in_mesh), len(out_mesh) + polyparams])
        A[:, 0] = 1
        A[:, 1:polyparams] = self.in_mesh
        A[:, polyparams:] = self.eval_BF(self.in_mesh, out_mesh)

        au = A.T @ self.in_vals
        self.out_vals = np.linalg.solve(C, au)[polyparams:]

        if self.rescale:
            au_rescaled =  self.eval_BF(self.in_mesh, out_mesh).T @ np.ones_like(self.in_vals)
            self.out_vals = self.out_vals / np.linalg.solve(self.eval_BF(out_mesh, out_mesh), au_rescaled)
        
        
        print("Conservativeness Integrated Delta =", np.sum(self.out_vals[polyparams:]) - np.sum(self.in_vals),
              ", rescaling =", self.rescale)

        return self.out_vals


   
class SeparatedConservative(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals):
        self.basisfunction, self.in_vals = basisfunction, in_vals
        self.in_mesh = coordify(in_mesh)
        
    def __call__(self, out_mesh):
        # set_trace()
        from scipy.linalg import inv

        out_mesh = coordify(out_mesh)
        dimension = len(out_mesh[0])
        
        self.C = self.eval_BF(out_mesh, out_mesh)
        A = self.eval_BF(self.in_mesh, out_mesh)
        
        V = np.zeros( [len(self.in_mesh), dimension + 1] )
        V[:, 0] = 1
        V[:, 1:] = self.in_mesh

        Q = np.zeros( [len(out_mesh), dimension + 1 ])
        Q[:, 0] = 1
        Q[:, 1:] = out_mesh

        # Q, V = V, Q # Swap

        f_xi = self.in_vals

        # import ipdb; ipdb.set_trace()

        QQ, QR = scipy.linalg.qr(Q, mode = "economic")

        epsilon = V.T @ f_xi
        eta     = A.T @ f_xi
        # mu      = inv(C) @ eta
        mu      = np.linalg.solve(self.C, eta)
        tau     = Q.T @ mu - epsilon
        sigma   = (QQ @ inv(QR).T) @ tau
        output  = mu - sigma
        # output = sigma
        
                
        # ST = inv(P).T @ Pt.T - (QQ @ inv(QR).T) @ (Q.T @ inv(P).T @ Pt.T + V.T)

        return output
        


