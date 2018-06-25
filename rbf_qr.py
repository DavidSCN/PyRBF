from rbf import RBF
import functools
import numpy as np
from scipy.special import hyp0f1
from scipy.linalg import solve_triangular
from coordinate_helper import *
import math


from tqdm import tqdm

class RBF_QR(RBF):
    def __init__(self, shape_param, in_mesh, in_vals, translate, scale):
        self.shape_param, self.in_mesh, self.in_vals, self.translate, self.scale \
            = shape_param, np.copy(in_mesh), np.copy(in_vals), translate, scale
        in_mesh = self.in_mesh      # update
        self.N = M = N = in_mesh.shape[1]
        # Step 1: Compute jmax + K
        self.K = K = self._get_K(np.float64)
        print("K=", K)
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

    def _get_D(self):
        """
        Compute scaling coefficient matrix (diagonal matrix), not needed for RBF-QR algorithm
        :return: Scaling coefficient matrix
        """
        raise NotImplementedError()

    def _get_D_fraction(self):
        """
        Compute fraction d_{N+j}/d_i of scaling coefficients for 0 <= i < N and 0 <= j < K - N
        :return: Fraction matrix of shape (N, K-N)
        """
        raise NotImplementedError()

    def basisfunction_i(self, i):
        def eval(i, x):
            T_at = [self._get_T()[i](x) for i in range(self.K)]
            return self.I_R_tilde[i, :] @ np.array(T_at)

        return functools.partial(eval, i)
    def old_basis_i(self, i):
        def eval(i, x):
            T_at = [self._get_T()[i](x) for i in range(self.K)]
            D = self._get_D()
            C = self._get_C()
            return (C @ D @ np.array(T_at))[i]
        return functools.partial(eval, i)
    def basisfunction(self):
        raise NotImplementedError()


class RBF_QR_1D(RBF_QR):
    def __init__(self, shape_param, in_mesh, in_vals, center=None, extents=None):
        if len(in_mesh.shape) <= 1:
            in_mesh = in_mesh[np.newaxis, :]
        if center is None or extents is None:
            center, extents = get_center_extents(in_mesh)
        in_mesh, translate, scale = translate_scale_hyperrectangle(np.copy(in_mesh), center, extents)
        assert(in_mesh[0, :].max() <= 1)
        super(RBF_QR_1D, self).__init__(shape_param, in_mesh, in_vals, translate, scale)
    def __call__(self, out_mesh):
        if len(out_mesh.shape) <= 1:
            out_mesh = out_mesh[np.newaxis, :]
        out_mesh = translate_scale_with(out_mesh, self.translate, self.scale)
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
            for k in range(1, self.N + j - i + 1):
                prod *= self.shape_param ** 2 / k
            D[i, j] = prod
        return D

    def _get_D(self):
        D = np.zeros((self.K, self.K))
        for i in range(self.K):
            prod = 2
            for k in range(1, i + 1):
                prod *= self.shape_param**2 / k
            D[i, i] = prod
        return D


class RBF_QR_2D(RBF_QR):
    def __init__(self, shape_param, in_mesh, in_vals, center = None, extents = None):
        in_mesh = np.array(in_mesh).reshape((2, -1))
        in_vals = np.array(in_vals).reshape(-1)
        if center is None or extents is None:
            center, extents = get_center_extents(in_mesh)
        in_mesh, translate, scale = translate_scale_hyperrectangle(in_mesh, center, extents)

        def cart2pol(mesh):
            result = np.empty(mesh.shape)
            result[0, :] = np.sqrt(mesh[0, :] ** 2 + mesh[1, :] ** 2)
            result[1, :] = np.arctan2(mesh[1, :], mesh[0, :])
            return result
        in_mesh = cart2pol(in_mesh)
        assert(in_mesh[0, :].max() <= 1)
        super(RBF_QR_2D, self).__init__(shape_param, in_mesh, in_vals, translate, scale)

    def __call__(self, out_mesh):
        out_mesh = translate_scale_with(np.array(out_mesh), self.translate, self.scale)
        original_shape = np.array(out_mesh).shape
        out_mesh = out_mesh.reshape((2, -1))

        def cart2pol(mesh):
            result = np.empty(mesh.shape)
            result[0, :] = np.sqrt(mesh[0, :] ** 2 + mesh[1, :] ** 2)
            result[1, :] = np.arctan2(mesh[1, :], mesh[0, :])
            return result
        out_mesh = cart2pol(out_mesh)
        assert(out_mesh[0, :].max() <= 1)
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
    def _get_D(self):
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
        D = np.zeros((self.K, self.K))
        for i in range(self.K):
            j, m = self.__index_convert(i)
            eps_power = (lambda x: self.shape_param**2, j)
            two_power = (lambda x: 0.5, j - 2*m + 1)
            fact_one = (lambda x: 1/x, (j + 2 * m + j % 2)/2)
            fact_two = (lambda x: 1/x, (j - 2*m - j % 2) /2)
            ratio = prodprod(eps_power, two_power, fact_one, fact_two)
            D[i,i] = ratio
        return D
