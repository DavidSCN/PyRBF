from basisfunctions import *

# from ipdb import set_trace
import functools
import numpy as np
import scipy.sparse.linalg
import scipy.spatial

from scipy.special import hyp0f1, poch
from scipy.linalg import solve_triangular
import math

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
        self.condC = np.linalg.cond(C)
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
        # print('C.shape = ', self.C.shape)
        self.gamma = np.linalg.solve(self.C, in_vals)
        
        self.rescaled = rescale
        if rescale:
            self.gamma_rescaled = np.linalg.solve(self.C, np.ones_like(in_mesh))


    def __call__(self, out_mesh):
        A = self.eval_BF(out_mesh, self.in_mesh)
        # print('A.shape = ', A.shape)
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
            self.rescaled_interp = np.linalg.solve(self.C, self.au_rescaled)
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


class RBF_QR(RBF):
    def __init__(self, shape_param, in_mesh, in_vals, min, max):
        self.shape_param, self.in_mesh, self.in_vals \
            = shape_param, np.copy(in_mesh), np.copy(in_vals)
        if min is not None and max is not None:
            self.scale_ratio = 1 / np.abs([min, max]).max
        else:
            scale_ratio = 1 / np.max(np.abs(self.in_mesh[0, :]))     # scale according to r
            if scale_ratio < 1.0:
                self.in_mesh *= scale_ratio
                self.scale_ratio = scale_ratio
            else:
                self.scale_ratio = None
        in_mesh = self.in_mesh      # update
        self.N = M = N = in_mesh.shape[1]
        # Step 1: Compute jmax + K
        self.K = K = self._get_K(np.float64)
        # Step 2: Assemble C
        C = self._get_C()
        # Step 3: QR decomposition of C and R_tilde
        Q, R = np.linalg.qr(C)
        R_dot = solve_triangular(R[:, :N], R[:, N:K])
        D_fraction = self._get_D_fraction()
        R_tilde = R_dot * D_fraction
        # Step 4: Evaluate expansion functions on in_mesh and compute A
        T = np.empty((K, M))
        for i in range(K):
            T[i, :] = self._get_T()[i](in_mesh)
        A = T[:N, :].T + T[N:K, :].T @ R_tilde.T
        # Step 5: Solve for lambda
        self.lamb = np.linalg.solve(A, in_vals)
        # Step 6: Prepare evaluation
        self.I_R_tilde = np.hstack((np.identity(N), R_tilde))

    def __call__(self, out_mesh):
        # Step 6: Evaluate
        out_mesh = np.copy(out_mesh)
        if self.scale_ratio is not None:
            out_mesh[0, :] = self.scale_ratio * out_mesh[0, :]
        out_length = out_mesh.shape[1]
        T_out = np.empty((self.K, out_length))
        for i in range(self.K):
            T_out[i, :] = self._get_T()[i](out_mesh)
        Psi_out = self.I_R_tilde @ T_out
        predicition = Psi_out.T @ self.lamb
        return predicition

    def _get_K(self, dtype):
        """
        Compute K for a given datatype's precision. This method can assume that self.shape_param
        is set to the correct value
        :return: K
        """
        raise NotImplementedError()

    def _get_C(self):
        """
        Compute coefficient matrix C.
        :return: Coefficient matrix of shape (N, K)
        """
        raise NotImplementedError()

    def _get_T(self):
        """
        Get modified chebyshev polynomials T(x)
        :return: Array (length K) of functions operating on meshes
        """
        raise NotImplementedError()

    def _get_D_fraction(self):
        """
        Compute fraction d_{N+j}/d_i of scaling coefficients for 0 <= i < N and 0 <= j < K - N
        :return: Fraction matrix of shape (N, K-N)
        """
        raise NotImplementedError()

    def basisfunction(self):
        raise NotImplementedError()


class RBF_QR_1D(RBF_QR):
    def __init__(self, shape_param, in_mesh, in_vals, min=None, max=None):
        if len(in_mesh.shape) <= 1:
            in_mesh = in_mesh[np.newaxis, :]
        super(RBF_QR_1D, self).__init__(shape_param, in_mesh, in_vals, min, max)
    def __call__(self, out_mesh):
        if len(out_mesh.shape) <= 1:
            out_mesh = out_mesh[np.newaxis, :]
        return super(RBF_QR_1D, self).__call__(out_mesh)
    def _get_K(self, dtype):
        mp = np.finfo(dtype).eps
        ep = self.shape_param

        jN = self.N - 1
        jmax = 1
        ratio = ep ** 2
        while jmax < jN and ratio > 1:
            jmax += 1
            ratio *= ep ** 2 / jmax
        if ratio < 1:  # d_jN was smallest
            jmax = jN
        ratio = ep ** 2 / (jmax + 1)
        while ratio > mp:
            jmax += 1
            ratio *= ep ** 2 / (jmax + 1)
        jmax = max(jmax, jN)  # ensure that K >= N
        return jmax + 1

    def _get_C(self):
        C = np.empty((self.N, self.K))
        for k, j in np.ndindex(self.N, self.K):
            t = 0.5 if j == 0 else 1
            C[k, j] = t * math.exp(-self.shape_param ** 2 * self.in_mesh[0, k]) * self.in_mesh[0, k] ** j * \
                  hyp0f1(j + 1, self.shape_param ** 4 * self.in_mesh[0, k] ** 2)
        return C

    def _get_T(self):
        def expansion_func(i, x):
            return np.exp(-x ** 2 * self.shape_param**2) \
                * np.cos(i * np.arccos(x))
        return [functools.partial(expansion_func, i) for i in range(self.K)]

    def _get_D_fraction(self):
        D = np.empty((self.N, self.K - self.N))
        for i, j in np.ndindex(self.N, self.K - self.N):
            prod = 2
            for k in range(1, i + 1):
                prod *= self.shape_param ** 2 / k
            D[i,j] = prod
        return D

class RBF_QR_2D(RBF_QR):
    def __init__(self, shape_param, in_mesh, in_vals, min=None, max=None, is_polar=False):
        in_mesh = np.array(in_mesh)     # Fine for now
        in_vals = np.array(in_vals).reshape(-1)
        if not is_polar:
            in_mesh = np.copy(in_mesh.reshape((2, -1))) # Copy before you slice

            def cart2pol(x, y):
                rho = np.sqrt(x ** 2 + y ** 2)
                phi = np.arctan2(y, x)
                return rho, phi
            in_mesh[0,:], in_mesh[1, :] = cart2pol(in_mesh[0, :], in_mesh[1, :])
        super(RBF_QR_2D, self).__init__(shape_param, in_mesh, in_vals, min, max)

    def __call__(self, out_mesh, is_polar=False):
        out_mesh = np.array(out_mesh) # This is fine for now
        original_shape = out_mesh.shape
        out_mesh = out_mesh.reshape((2, -1))
        if not is_polar:
            out_mesh = np.copy(out_mesh)  # Copy before you slice

            def cart2pol(x, y):
                rho = np.sqrt(x ** 2 + y ** 2)
                phi = np.arctan2(y, x)
                return rho, phi

            out_mesh[0, :], out_mesh[1, :] = cart2pol(out_mesh[0, :], out_mesh[1, :])

        result = super(RBF_QR_2D, self).__call__(out_mesh)
        return result.reshape(original_shape[1:])

    def _get_K(self, dtype):
        mp = np.finfo(dtype).eps
        ep = self.shape_param

        jN = math.ceil(-3 / 2 + math.sqrt(9 / 4 + 2 * self.N - 2))
        jmax = 1
        ratio = ep ** 2 / 2
        while jmax < jN and ratio > 1:
            jmax += 1
            ratio *= ep ** 2 / (jmax + (jmax % 2))
        if ratio < 1:
            jmax = jN
        ratio *= ep ** 2 / (jmax + 1 + (jmax + 1) % 2)
        while ratio * math.exp(0.223 * (jmax + 1) + 0.212 - 0.657 * ((jmax + 1) % 2)) > mp:
            jmax += 1
            ratio *= ep ** 2 / (jmax + 1 + (jmax + 1) % 2)
        K = int((jmax + 2) * (jmax + 1) / 2)
        return K

    @staticmethod
    def __index_convert(i):
        j = math.floor(0.5 * (math.sqrt(1 + 8 * i) - 1))
        m = i - int(j * (j + 1) / 2)
        return j, m


    def _get_C(self):
        def hyp1f2(a, b, c, x):
            eps = np.finfo(np.float64).eps
            alpha = 1
            sum = 1  # first summand is always 1.
            n = 1  # skip first summand
            while alpha > eps:
                alpha *= (a + n - 1) / ((b + n - 1) * (c + n - 1) * n) * x
                sum += alpha
                n += 1
            return sum

        def sc_at(trigfunc, j, m, k):
            p = j % 2
            b = 1 if 2 * m + p == 0 else 2
            t = 0.5 if j - 2 * m == 0 else 1
            alpha = (j - 2 * m + p + 1) / 2
            beta = (j - 2 * m + 1, int((j + 2 * m + p + 2) / 2))    # Note that (j + 2 * m + p + 2) / 2 is always int
            return b * t * math.exp(-self.shape_param ** 2 * self.in_mesh[0, k] ** 2) \
                   * self.in_mesh[0, k] ** j \
                   * trigfunc((2 * m + p) * self.in_mesh[1, k]) \
                   * hyp1f2(alpha, beta[0], beta[1],
                            self.shape_param ** 4 * self.in_mesh[0, k] ** 2)

        c_at = functools.partial(sc_at, math.cos)
        s_at = functools.partial(sc_at, math.sin)

        C = np.empty((self.N, self.K))
        for k, i in np.ndindex(C.shape):
            j, m = self.__index_convert(i)
            C[k, i] = c_at(j, m, k) if m <= (j - j % 2) / 2 \
                else s_at(j, m - (j + j % 2)/2, k)
        return C

    def _get_T(self):

        def cheby_at(i, x):
            j, m = self.__index_convert(i)

            def modified_cheby(trigfunc, j, m, r, Theta):
                return np.exp(-self.shape_param**2 * r ** 2) * r ** (2 * m) \
                       * np.cos((j - 2*m) * np.arccos(r)) \
                       * trigfunc((2 * m + j % 2) * Theta)
            return modified_cheby(np.cos, j, m, x[0, :], x[1, :]) if m <= (j - j % 2)/2 \
                else modified_cheby(np.sin, j, m - (j + j % 2)/2, x[0, :], x[1, :])
        return [functools.partial(cheby_at, i) for i in range(self.K)]

    def _get_D_fraction(self):
        def prodprod(*args):
            maxidx = 0
            for pair in args:
                assert(len(pair) == 2)
                assert(int(pair[1]) == pair[1])
                maxidx = max(maxidx, pair[1])
            prod = 1
            for k in range(1, int(maxidx) + 1):
                for pair in args:
                    prod *= pair[0](k) if pair[1] >= k else 1
            return prod

        def d_quot(num_idx, denom_idx):
            num_idx += self.N
            y = np.array([num_idx, denom_idx])
            j = np.floor(0.5*(np.sqrt(1 + 8*y) - 1))
            m = y - (j * (j+1) / 2).astype(int)
            if m[0] > (j[0] - j[0] % 2)/2:
                m[0] -= (j[0] + j[0] % 2)/2
            if m[1] > (j[1] - j[1] % 2)/2:
                m[1] -= (j[1] + j[1] % 2)/2
            assert(0 <= m[0] <= j[0] and 0 <= m[1] <= j[1])
            eps_power = (lambda x: self.shape_param ** 2, j[0] - j[1])
            two_power = (lambda x: 0.5, j[0] - 2*m[0] - j[1] + 2*m[1])
            fact_one_num = (lambda x: x, (j[1] + 2 * m[1] + j[1] % 2)/2)
            fact_one_denom = (lambda x: 1/x, (j[0] + 2 * m[0] + j[0] % 2)/2)
            fact_two_num = (lambda x: x, (j[1] - 2 * m[1] - j[1] % 2)/2)
            fact_two_denom = (lambda x: 1/x, (j[0] - 2 * m[0] - j[0] % 2)/2)
            result = prodprod(eps_power, two_power, fact_one_num,
                              fact_one_denom, fact_two_num, fact_two_denom)
            return result
        D = np.empty((self.N, self.K - self.N))
        for i, j in np.ndindex(D.shape):
            D[i, j] = d_quot(j, i)
        return D
