from basisfunctions import *
import mesh

import functools
import numpy as np
import scipy.sparse.linalg
import scipy.spatial
from scipy.linalg import lu
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import matplotlib.pylab as plt
import scipy.sparse as sparse

dimension = 1
# func = functools.partial(lambda x: Gaussian(x-1, 1) + 2)
# heaviside = lambda x: 0 if x < 2 else 1
# func = np.vectorize(heaviside)
# func = lambda x: x


class RBF:
    def __str__(self):
        return type(self).__name__

    def RMSE(self, func, test_mesh):
        """ Returns the root mean squared error. """
        return np.sqrt((self.error(func, test_mesh) ** 2).mean())

    def error(self, func, test_mesh):
        return self(test_mesh) - func(test_mesh)

    def weighted_error(self, func, out_mesh):
        """ Weighted error to get a better error for conservative interpolation. """
        return self(out_mesh) * len(out_mesh) / len(self.in_mesh) - func(out_mesh)

    def rescaled_error(self, func, out_mesh):
        g = NoneConservative(self.basisfunction, self.in_mesh, np.ones_like(self.in_vals), False)
        return self(out_mesh) / g(out_mesh) - func(out_mesh)


    def eval_BF(self, meshA, meshB):
        """ Evaluates single BF or list of BFs on the meshes. """
        if meshA.ndim == 1:
            meshA = meshA[:, np.newaxis]
        if meshB.ndim == 1:
            meshB = meshB[:, np.newaxis]
        dm = scipy.spatial.distance_matrix(meshA, meshB)
        
        if type(self.basisfunction) is list:
            A = np.empty((len(meshA), len(meshB)))
            # for i, row in enumerate(meshA):
                # for j, col in enumerate(meshB):
                    # A[i, j] = self.basisfunction[j](row - col)
            for j, _ in enumerate(meshB):
                A[:,j] = self.basisfunction[j](dm[:,j])
        else:
            A = self.basisfunction(dm)

        return A

    def polynomial(self, out_mesh):
        raise NotImplementedError

    @property
    def condC(self):
        cond = getattr(self, "_condC", np.linalg.cond(self.C))
        self._condC = cond
        return cond


    
class SeparatedConsistent(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
        in_mesh = mesh.coordify(in_mesh)
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
        out_mesh = mesh.coordify(out_mesh)
        A = self.eval_BF(out_mesh, self.in_mesh)
        V = np.zeros( [out_mesh.shape[0], out_mesh.shape[1] + 1 ])
        V[:, 0] = 1
        V[:, 1:] = out_mesh

        out_vals = A @ self.gamma
        if self.rescaled:
            out_vals = out_vals / (A @ self.gamma_rescaled)

        return out_vals + V @ self.beta

class SeparatedConsistentFitted(RBF):        
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False, degree = 2):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        # self.poly = np.poly1d(np.polyfit(in_mesh, in_vals, degree))
        self.polyfit = np.polynomial.polynomial.Polynomial.fit(in_mesh, in_vals, deg = degree)
        
        self.C = self.eval_BF(in_mesh, in_mesh)
        rhs = in_vals - self.polyfit(in_mesh)
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

        return out_vals + self.polyfit(out_mesh)



class IntegratedConsistent(RBF): #Fixme this does not work in 2d (yet)!
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
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
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


class NoneConsistent(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
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

class AMLS(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        self.C = self.eval_BF(in_mesh, in_mesh)
        self.gamma = np.linalg.solve(self.C, in_vals)
        self.in_vals = in_vals
        
        self.rescaled = rescale
        if rescale:
            self.gamma_rescaled = np.linalg.solve(self.C, np.ones_like(in_mesh))


    def __call__(self, out_mesh):
        A = self.eval_BF(out_mesh, self.in_mesh)
        Q = self.in_vals @ self.C
        for i in range(0,1):
          res = self.in_vals - Q
          u = res @ self.C
          Q += u

        out_vals = A @ self.gamma
        if self.rescaled:
            out_vals = out_vals / (A @ self.gamma_rescaled)
        return out_vals, Q

class AMLSInverse(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        self.C = self.eval_BF(in_mesh, in_mesh)
        self.gamma = np.linalg.solve(self.C, in_vals)
        self.Cinv = np.linalg.inv(self.C)
        self.in_vals = in_vals
        
        self.rescaled = rescale
        if rescale:
            self.gamma_rescaled = np.linalg.solve(self.C, np.ones_like(in_mesh))


    def __call__(self, out_mesh):
        A = self.eval_BF(out_mesh, self.in_mesh)
        out_vals = A @ self.gamma
        if self.rescaled:
            out_vals = out_vals / (A @ self.gamma_rescaled)
        return self.C, self.Cinv

class LOOCV(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        self.C = self.eval_BF(in_mesh, in_mesh)
        self.Cinv = np.linalg.inv(self.C)
        #print(self.Cinv)
        self.gamma = np.linalg.solve(self.C, in_vals)
        
        self.rescaled = rescale

    def __call__(self):
        error = []
        for i in range(0,len(self.in_mesh)):
          error.append(self.gamma[i]/self.Cinv[i][i])
        return error

class LUDecomp(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        self.C = self.eval_BF(in_mesh, in_mesh)
        self.Cinv = np.linalg.inv(self.C)
        self.gamma = np.linalg.solve(self.C, in_vals)
        
        self.rescaled = rescale

    def __call__(self):
        p, l, u = lu(self.C)
        return p, l, u

class Rational(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        self.C = self.eval_BF(in_mesh, in_mesh)
        self.Cinv = np.linalg.inv(self.C)
        self.D = 0*self.C
        self.Stemp = 0*self.C
        plt.spy(self.C, markersize=1)
        
        self.rescaled = rescale

    def __call__(self, in_vals, out_mesh):

        A = self.eval_BF(out_mesh, self.in_mesh)
        print("in vals: ", in_vals)

        for i in range(0,len(in_vals)):
            self.D[i][i] = in_vals[i]

        #print("D: ", self.D)

        sumF = 0
        for i in range(0,len(in_vals)):
            sumF += pow(in_vals[i],2)
        K = 1.0/sumF
        for i in range(0,len(in_vals)):
            self.Stemp[i][i] = 1.0/(K*pow(in_vals[i],2) + 1)
        #print("S: ", self.Stemp)
        S = self.Stemp @ (K * self.D @ self.Cinv @ self.D + self.Cinv)

        EigValues, EigVectors = np.linalg.eig(S)
        
        #print("Eigen Vectors: ", EigVectors)

        minValue = EigValues[0]
        minLoc = 0
        for i in range(1,len(in_vals)):
            if (EigValues[i] < minValue):
                minValue = EigValues[i]
                minLoc = i
        #q = np.transpose(EigVectors[minLoc])
        q = EigVectors[:,minLoc]
        #for i in range(0,len(in_vals)):
        #    p[i] = self.D[i][i] * q[i]
        p = self.D @ q
        #print("Q: ", q)
        #print("P: ", p)

        pAlpha = self.Cinv @ p 
        qAlpha = self.Cinv @ q
        #print("qAlpha: ", qAlpha)
        #print("pAlpha: ", pAlpha)

        fr = (A @ pAlpha)/(A @ qAlpha)

        return fr

class fullSVD(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        self.C = self.eval_BF(in_mesh, in_mesh)
        self.Cinv = np.linalg.inv(self.C)
        self.gamma = np.linalg.solve(self.C, in_vals)
        
        self.rescaled = rescale

    def __call__(self):
        U, s, Vh = svd(self.C)
        return self.C, U, s, Vh

class truncSVD(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
        self.in_mesh, self.basisfunction = in_mesh, basisfunction
        
        self.C = self.eval_BF(in_mesh, in_mesh)
        self.Cinv = np.linalg.inv(self.C)
        self.gamma = np.linalg.solve(self.C, in_vals)
        
        self.rescaled = rescale

    def __call__(self):
        U, s, Vh = svd(self.C)
        return self.C, U, s, Vh


class NoneConservative(RBF):
    """ No polynomial, conservative interpolation. """
    
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
        self.basisfunction, self.in_mesh, self.in_vals, self.rescale = basisfunction, in_mesh, in_vals, rescale
        
    def __call__(self, out_mesh):
        self.C = self.eval_BF(out_mesh, out_mesh)
        A = self.eval_BF(self.in_mesh, out_mesh)

        au = A.T @ self.in_vals
        out_vals = np.linalg.solve(self.C, au)

        if self.rescale:
            self.au_rescaled = A.T @ np.ones_like(self.in_vals)
            self.rescaled_interp = np.linalg.solve(self.C, self.au_rescaled)
            self.rescaled_interp = self.rescaled_interp + (1 - np.mean(self.rescaled_interp))
            self.out_vals = self.out_vals / self.rescaled_interp
                    
        return out_vals


    def rescaled_interpolant(self, out_mesh):
        try:
            return self.rescaled_interp
        except NameError:
            print("Rescaling not available.")
            return np.ones_like(out_mesh)

    
class IntegratedConservative(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals, rescale = False):
        self.basisfunction, self.in_vals, self.rescale = basisfunction, in_vals, rescale
        self.in_mesh = mesh.coordify(in_mesh)
        
    def __call__(self, out_mesh):
        out_mesh = mesh.coordify(out_mesh)
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
        out_vals = np.linalg.solve(C, au)[polyparams:]

        if self.rescale:
            au_rescaled =  self.eval_BF(self.in_mesh, out_mesh).T @ np.ones_like(self.in_vals)
            out_vals = out_vals / np.linalg.solve(self.eval_BF(out_mesh, out_mesh), au_rescaled)
        
        
        return out_vals


   
class SeparatedConservative(RBF):
    def __init__(self, basisfunction, in_mesh, in_vals):
        self.basisfunction, self.in_vals = basisfunction, in_vals
        self.in_mesh = mesh.coordify(in_mesh)
        
    def __call__(self, out_mesh):
        from scipy.linalg import inv
        out_mesh = mesh.coordify(out_mesh)
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

        QQ, QR = scipy.linalg.qr(Q, mode = "economic")

        epsilon = V.T @ f_xi
        eta     = A.T @ f_xi
        mu      = np.linalg.solve(self.C, eta)
        tau     = Q.T @ mu - epsilon
        sigma   = (QQ @ inv(QR).T) @ tau
        output  = mu - sigma
        # output = sigma
        
                
        # ST = inv(P).T @ Pt.T - (QQ @ inv(QR).T) @ (Q.T @ inv(P).T @ Pt.T + V.T)

        return output
