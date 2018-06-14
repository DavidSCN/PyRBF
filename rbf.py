from basisfunctions import *

# from ipdb import set_trace
import functools
import numpy as np
import scipy.sparse.linalg
import scipy.spatial

from scipy.special import hyp0f1, hyp1f2, poch
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
        

class RBF_QR_1D(RBF):
    def __init__(self, shape_param, in_mesh, in_vals):
        self.shape_param, self.in_vals, self.in_mesh = shape_param, np.copy(in_vals), np.copy(in_mesh)
        scale_ratio = 1/np.max(np.abs(self.in_mesh))
        if scale_ratio < 1.0:
            self.in_mesh *= scale_ratio
        self.scale_ratio = min(1.0, scale_ratio)
        in_mesh = self.in_mesh      # This does no longer refer to the parameter of the constructor!

        def get_jmax(type, N, shape_param):
            mp = np.finfo(type).eps
            ep = shape_param

            jN = N - 1
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
            return jmax

        M = N = len(in_mesh)
        # Step 1: Compute jmax
        jmax = get_jmax(np.float64, N, shape_param)
        self.K = K = jmax + 1
        # Step 2: Assemble C (D gets assembled impicitly)
        C = np.empty((N, K))
        # TODO: Both matrices can be assembled using dynamic programming!!
        for k, j in np.ndindex(N, K):
            t = 0.5 if j == 0 else 1
            C[k, j] = t * math.exp(-shape_param ** 2 * in_mesh[k]) * in_mesh[k] ** j * \
                      hyp0f1(j + 1, shape_param ** 4 * in_mesh[k] ** 2)
        # Step 3: QR Decomposition of C and R_tilde
        Q, R = np.linalg.qr(C)
        R_dot = solve_triangular(R[:, :N], R[:, N: K])
        R_tilde = np.empty((N, K - N))
        # TODO: dynamic programming!!
        for i, j in np.ndindex(N, K - N):
            R_tilde[i, j] = R_dot[i, j] * (shape_param ** (2 * (N + j - i))) / math.factorial(N + j - i)
        # Step 4: Evaluate chebyshev polynomials at x_k and compute A
        T_1 = np.empty((N, M))
        for i, j in np.ndindex(N, M):
            T_1[i, j] = math.exp(-in_mesh[j] ** 2 * shape_param ** 2) * self.chebyshev(i, in_mesh[j])
        T_2 = np.empty((K - N, M))
        for i, j in np.ndindex(K - N, M):
            T_2[i, j] = math.exp(-in_mesh[j] ** 2 * shape_param ** 2) * self.chebyshev(N + i, in_mesh[j])
        A = T_1.T + T_2.T @ R_tilde.T
        # Step 5: Solve for lambda
        self.lamb = np.linalg.solve(A, in_vals)
        # Step 6:  Prepare evaluation
        self.I_R_tilde = np.hstack((np.identity(N), R_tilde))

    def __call__(self, out_mesh):
        # Step 6: Evaluate
        out_mesh = self.scale_ratio * np.copy(out_mesh)
        out_length = len(out_mesh)
        T_out = np.empty((self.K, out_length))
        for i, j in np.ndindex(self.K, out_length):
            T_out[i, j] = math.exp(-out_mesh[j] ** 2 * self.shape_param ** 2) * self.chebyshev(i, out_mesh[j])
        Psi_out = self.I_R_tilde @ T_out
        prediction = Psi_out.T @ self.lamb
        return prediction

    def chebyshev(self, n, x):
        return math.cos(n * math.acos(x)) if x <= 1 else math.cosh(n * math.acosh(x))

class RBF_QR_2D(RBF):
    def __init__(self, shape_param, in_mesh, in_vals, is_polar=False):
        self.shape_param = shape_param
        self.in_mesh = in_mesh = np.copy(self.in_mesh)
        if not is_polar:
            def cart2pol(x, y):
                rho = np.sqrt(x ** 2 + y ** 2)
                phi = np.arctan2(y, x)
                return rho, phi
            in_mesh[:,0], in_mesh[:, 1] = cart2pol(in_mesh[:, 0], in_mesh[:, 1])
        else:
            in_mesh = np.copy(in_mesh)

        scale_ratio = 1 / np.max(np.abs(self.in_mesh[:, 0]))
        if scale_ratio < 1.0:
            self.in_mesh *= scale_ratio
        self.scale_ratio = min(1.0, scale_ratio)

        def get_jmax(type, N, shape_param):
            mp = np.finfo(type).eps
            ep = shape_param

            jN = math.ceil(-3/2 + math.sqrt(9/4 + 2 * N - 2))
            jmax = 1
            ratio = ep**2/2
            while jmax < jN and ratio > 1:
                jmax += 1
                ratio *= ep**2 / (jmax + (jmax % 2))
            if ratio < 1:
                jmax = jN
            ratio *= ep**2 / (jmax + 1 + (jmax + 1) % 2)
            # while ratio * math.exp(0.223 * (jmax + 1) + 0.212 * (1 - 3.097 * ((jmax + 1) % 2))) > mp:
            while ratio * math.exp(0.223 * (jmax + 1) + 0.212 - 0.657 * ((jmax + 1) % 2)) > mp:
                jmax += 1
                ratio *= ep**2 / (jmax + 1 + (jmax + 1) % 2)
            return jmax

        M = N = len(in_mesh)
        # Step 1: Compute jmax
        jmax = get_jmax(np.float64, N, shape_param)
        self.K = K = (jmax + 2) * (jmax + 1) / 2
        assert(K >= N)
        # Step 2: Assemble C (D gets assembled implicitly)
        C = np.full((N, K), 42) # Check if that indexing

        def sc_at(trigfunc, j, m, k):
            p = j % 2
            b = 1 if 2 * m + p == 0 else 2
            t = 0.5 if j - 2 * m == 0 else 1
            alpha = (j - 2 * m + p + 1) / 2
            beta = (j - 2 * m + 1, (j + 2 * m + p + 2) / 2)
            return b * t * math.exp(-shape_param ** 2 + in_mesh[k, 0] ** 2) \
                   * in_mesh[k, 0] ** j \
                   * trigfunc((2 * m + p) * in_mesh[k, 1]) \
                   * hyp1f2(alpha, beta[0], beta[1], shape_param ** 4 * in_mesh[k, 0] ** 2)

        c_at = functools.partial(sc_at, math.cos)
        s_at = functools.partial(sc_at, math.sin)

        for k, j in np.ndindex(N, jmax + 1):
            for m in range(0, j + 1):
                C[k, (j+1) * (j+2) / 2] = c_at(j, m, k) if m <= (j - j % 2) / 2 \
                    else s_at(j, m - (j + j % 2) / 2, k)

        assert(np.argwhere(C - 42).size == 0)   # Make sure we forgot no index.
        # Step 3: QR Decomposition of C and R_tilde
        Q, R = np.linalg.qr(C)
        R_dot = solve_triangular(R[:, :N], R[:, N: K])
        R_tilde = np.empty((N, K - N))
        def d_quot(num_idx, denom_idx):
            y = np.array([num_idx, denom_idx])
            j = np.floor(0.5*(np.sqrt(1 + 8*y) - 1))
            m = i - j
            assert(0 <= m <= j)
            result = shape_param ** (2 * (j[0] - j[1])) / 2**(j[0] - 2*m[0] - j[1] + 2 * m[1])
            def fact_quot(a, b):
                assert(math.floor(a) == a and math.floor(b) == b)
                return poch(b+1, a - b - 1) if a >= b else 1/fact_quot(b, a)
            result *= fact_quot((j[1] + 2 * m[1] + j[1] % 2)/2, (j[0] + 2 * m[0] + j[0] % 2)/2)
            result *= fact_quot((j[1] - 2 * m[1] - j[1] % 2)/2, (j[0] - 2 * m[0] - j[0] % 2)/2)
            return result
        # for i, j in np.ndindex(N, K - N):
        # R_tilde[i, j] = R_dot[i, j] * (shape_param ** (2 * (N + j - i))) / math.factorial(N + j - i)
        for (i, k) in np.ndindex(N, K):
            R_tilde[i, k] = R_dot[i,k] * d_quot(N + k, i)
        # Step 4: Evaluate chebyshev polynomial at x_k and compute A

        def cheby_at(i, k):
            j = j = math.floor(0.5*(math.sqrt(1 + 8*i) - 1))
            m = i - j

            def modified_cheby(trigfunc, j, m, x, k):

                def chebyshev(n, x):
                    return math.cos(n * math.acos(x)) if x <= 1 else math.cosh(n * math.acosh(x))

                return math.exp(-shape_param**2*x[k, 0]**2)*x[k, 0]**(2*m) \
                    * chebyshev(j - 2*m, x[k, 0]) \
                    * trigfunc((2 * m + j % 2) * x[k, 1])
            return modified_cheby(math.cos, j, m, in_mesh, k) if m <= (j - j % 2)/2 \
                else modified_cheby(math.sin, j, m, in_mesh, k)
        T_1 = np.empty((N, M))
        for i, j in np.ndindex(N, M):
            T_1[i, j] = cheby_at(i, j)
        T_2 = np.empty((K - N, M))
        for i, j in np.ndindex(K - N, M):
            T_2[i, j] = cheby_at(N + i, j)
        A = T_1.T + T_2.T @ R_tilde.T
        # Step 5: Solve for lambda
        self.lamb = np.linalg.solve(A, in_vals)
        # Step 6:  Prepare evaluation
        self.I_R_tilde = np.hstack((np.identity(N), R_tilde))

    def __call__(self, out_mesh, is_polar):
        if not is_polar:
            def cart2pol(x, y):
                rho = np.sqrt(x ** 2 + y ** 2)
                phi = np.arctan2(y, x)
                return rho, phi
            out_mesh[:, :] = cart2pol(out_mesh[:, 0], out_mesh[:, 1])
        else:
            out_mesh = np.copy(out_mesh)

        def cheby_at(i, k):
            j = j = math.floor(0.5*(math.sqrt(1 + 8*i) - 1))
            m = i - j

            def modified_cheby(trigfunc, j, m, x, k):

                def chebyshev(n, x):
                    return math.cos(n * math.acos(x)) if x <= 1 else math.cosh(n * math.acosh(x))

                return math.exp(-self.shape_param**2*x[k, 0]**2)*x[k, 0]**(2*m) \
                    * chebyshev(j - 2*m, x[k, 0]) \
                    * trigfunc((2 * m + j % 2) * x[k, 1])
            return modified_cheby(math.cos, j, m, out_mesh, k) if m <= (j - j % 2)/2 \
                else modified_cheby(math.sin, j, m, out_mesh, k)

        out_mesh = self.scale_ratio * np.copy(out_mesh)
        out_length = len(out_mesh)
        T_out = np.empty((self.K, out_length))
        for i, j in np.ndindex(self.K, out_length):
            T_out[i, j] = cheby_at(i, j)
        Psi_out = self.I_R_tilde @ T_out
        prediction = Psi_out.T @ self.lamb
        if not is_polar:
            def pol2cart(r, theta):
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                return x, y
            prediction[:, 0], prediction[:, 1] = pol2cart(prediction[:, 0], prediction[:, 1])
        return prediction




