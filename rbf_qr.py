import numpy as np
from rbf import RBF
import functools
from scipy.special import hyp0f1, lpmn
from scipy.linalg import solve_triangular
from coordinate_helper import *
import math

class RBF_QR(RBF):
    def __init__(self, shape_param, in_mesh, in_vals, translate, scale):
        self.shape_param, self.in_mesh, self.in_vals, self.translate, self.scale \
            = shape_param, np.copy(in_mesh), np.copy(in_vals), translate, scale
        in_mesh = self.in_mesh  # update
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
        assert (in_mesh[0, :].max() <= 1)
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
            return np.exp(-x ** 2 * self.shape_param ** 2) \
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
                prod *= self.shape_param ** 2 / k
            D[i, i] = prod
        return D


class RBF_QR_2D(RBF_QR):
    def __init__(self, shape_param, in_mesh, in_vals, center=None, extents=None):
        in_mesh = np.array(in_mesh).reshape((2, -1))
        in_vals = np.array(in_vals).reshape(-1)
        if center is None or extents is None:
            center, extents = get_center_extents(in_mesh)
        in_mesh, translate, scale = translate_scale_hyperrectangle(in_mesh, center, extents)
        in_mesh = cart2polar(in_mesh)
        assert (in_mesh[0, :].max() <= 1)
        super(RBF_QR_2D, self).__init__(shape_param, in_mesh, in_vals, translate, scale)

    def __call__(self, out_mesh):
        out_mesh = translate_scale_with(np.array(out_mesh), self.translate, self.scale)
        original_shape = np.array(out_mesh).shape
        out_mesh = out_mesh.reshape((2, -1))
        out_mesh = cart2polar(out_mesh)
        assert (out_mesh[0, :].max() <= 1)
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
            beta = (j - 2 * m + 1, int((j + 2 * m + p + 2) / 2))  # Note that (j + 2 * m + p + 2) / 2 is always int
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
                else s_at(j, m - (j + j % 2) / 2, k)
        return C

    def _get_T(self):

        def cheby_at(i, x):
            j, m = self.__index_convert(i)

            def modified_cheby(trigfunc, j, m, r, Theta):
                return np.exp(-self.shape_param ** 2 * r ** 2) * r ** (2 * m) \
                       * np.cos((j - 2 * m) * np.arccos(r)) \
                       * trigfunc((2 * m + j % 2) * Theta)

            return modified_cheby(np.cos, j, m, x[0, :], x[1, :]) if m <= (j - j % 2) / 2 \
                else modified_cheby(np.sin, j, m - (j + j % 2) / 2, x[0, :], x[1, :])

        return [functools.partial(cheby_at, i) for i in range(self.K)]

    def _get_D_fraction(self):

        def d_quot(num_idx, denom_idx):
            num_idx += self.N
            y = np.array([num_idx, denom_idx])
            j = np.floor(0.5 * (np.sqrt(1 + 8 * y) - 1))
            m = y - (j * (j + 1) / 2).astype(int)
            if m[0] > (j[0] - j[0] % 2) / 2:
                m[0] -= (j[0] + j[0] % 2) / 2
            if m[1] > (j[1] - j[1] % 2) / 2:
                m[1] -= (j[1] + j[1] % 2) / 2
            assert (0 <= m[0] <= j[0] and 0 <= m[1] <= j[1])
            eps_power = (lambda x: self.shape_param ** 2, j[0] - j[1])
            two_power = (lambda x: 0.5, j[0] - 2 * m[0] - j[1] + 2 * m[1])
            fact_one_num = (lambda x: x, (j[1] + 2 * m[1] + j[1] % 2) / 2)
            fact_one_denom = (lambda x: 1 / x, (j[0] + 2 * m[0] + j[0] % 2) / 2)
            fact_two_num = (lambda x: x, (j[1] - 2 * m[1] - j[1] % 2) / 2)
            fact_two_denom = (lambda x: 1 / x, (j[0] - 2 * m[0] - j[0] % 2) / 2)
            result = RBF_QR_2D.__prodprod(eps_power, two_power, fact_one_num,
                            fact_one_denom, fact_two_num, fact_two_denom)
            return result

        D = np.empty((self.N, self.K - self.N))
        for i, j in np.ndindex(D.shape):
            D[i, j] = d_quot(j, i)
        return D

    def _get_D(self):

        D = np.zeros((self.K, self.K))
        for i in range(self.K):
            j, m = self.__index_convert(i)
            eps_power = (lambda x: self.shape_param ** 2, j)
            two_power = (lambda x: 0.5, j - 2 * m + 1)
            fact_one = (lambda x: 1 / x, (j + 2 * m + j % 2) / 2)
            fact_two = (lambda x: 1 / x, (j - 2 * m - j % 2) / 2)
            ratio = RBF_QR_2D.__prodprod(eps_power, two_power, fact_one, fact_two)
            D[i, i] = ratio
        return D

    @staticmethod
    def __prodprod(*args):
        maxidx = 0
        for pair in args:
            assert (len(pair) == 2)
            assert (int(pair[1]) == pair[1])
            maxidx = max(maxidx, pair[1])
        prod = 1
        for k in range(1, int(maxidx) + 1):
            for pair in args:
                prod *= pair[0](k) if pair[1] >= k else 1
        return prod

class RBF_QR_3D(RBF_QR):
    def __init__(self, shape_param, in_mesh, in_vals, center=None, extents=None):
        in_mesh = np.array(in_mesh).reshape(3, -1)
        in_vals = np.array(in_vals).reshape(-1)
        if center is None or extents is None:
            center, extents = get_center_extents(in_mesh)
        in_mesh, translate, scale = translate_scale_hyperrectangle(in_mesh, center, extents)
        in_mesh = cart2polar(in_mesh)
        assert (in_mesh[0, :].max() <= 1)
        super().__init__(shape_param, in_mesh, in_vals, translate, scale)

    def __call__(self, out_mesh):
        out_mesh = translate_scale_with(np.array(out_mesh), self.translate, self.scale)
        original_shape = out_mesh.shape
        out_mesh = np.array(out_mesh).reshape(3, -1)
        out_mesh = cart2polar(out_mesh)
        assert (out_mesh[0, :].max() <= 1)
        prediction = super().__call__(out_mesh)
        return prediction.reshape(original_shape[1:])

    def _get_K(self, dtype):
        mp = np.finfo(dtype).eps
        ep = self.shape_param

        def degree(N):
            K = 42  # debugging only
            for k in range(N - 1):
                dim = np.prod(np.arange(k + 1, k + 4) / np.arange(1, 4))
                if dim >= N:
                    K = k
                    break
            assert (K != 42)
            return K

        N = self.in_mesh.shape[1]
        jmax = 1
        jN = degree(N)
        fac = ep ** 2 / 6
        ratio = fac * (jmax + 1)
        while jmax < jN and ratio > 1:
            jmax += 1
            fac *= ep ** 2
            if jmax % 2 == 0:
                ratio = fac
            else:
                fac /= (jmax + 1) / (jmax + 2)
                ratio *= jmax + 1
        if ratio < 1:
            jmax = jN
            fac = 1
            if jN % 2 == 1:
                fac /= jN + 1
        fac *= ep ** 2
        if (jmax + 1) % 2 == 1:
            fac /= (jmax + 2) / (jmax + 3)
            ratio = fac * (jmax + 2)
        while ratio * math.exp(0.223 * (jmax + 1) - 0.012 - 0.649 * ((jmax + 1) % 2)) > mp:
            jmax += 1
            fac *= ep ** 2
            if (jmax + 1) % 2 == 1:
                fac /= (jmax + 2) * (jmax + 3)
                ratio = fac * (jmax + 2)
            else:
                ratio = fac
        K = int(1 / 6 * (jmax + 1) * (jmax + 2) * (jmax + 3))
        j = m = v = 0
        self.__indices = np.empty((K, 3))
        for i in range(K):
            if v > 2 * m + j % 2:
                v = -(2 * m + j % 2)
                m += 1
            if m > (j - j % 2) / 2:
                m = 0
                j += 1
                v = -(2 * m + j % 2)  # reset v with new m!
            self.__indices[i, :] = [int(j), int(m), int(v)]
            v += 1
        return K

    def _get_C(self):
        C = np.empty((self.N, self.K))
        for k, i in np.ndindex(C.shape):
            j, m, v = self.__indices[i, :]
            t = 0.5 if j - 2 * m == 0 else 1
            y = 0.5 if v == 0 else 1
            expfact = math.exp(-self.shape_param ** 2 * self.in_mesh[0, k] ** 2)
            Y = RBF_QR_3D.__calc_Y(v, 2 * m + j % 2,
                                   self.in_mesh[1, k], self.in_mesh[2, k])
            hypergeometric = RBF_QR_3D.hyp_pfq([(j - 2 * m + 1) / 2, (j - 2 * m + 2) / 2],
                                               [j - 2 * m + 1, (j - 2 * m - j % 2 + 2) / 2,
                                                (j + 2 * m + j % 2 + 3) / 2],
                                               self.shape_param ** 4 * self.in_mesh[0, k] ** 2)
            C[k, i] = t * y * expfact * Y * hypergeometric
        return C

    def _get_T(self):
        def T_at(i, x):
            j, m, v = self.__indices[i, :]
            prefact = np.exp(-self.shape_param ** 2 * x[0, :] ** 2) * x[0, :] ** (2 * m)
            Y = RBF_QR_3D.__calc_Y(v, 2 * m + (j % 2), x[1, :], x[2, :])
            cheby = np.cos((j - 2 * m) * np.arccos(x[0, :]))
            return prefact * Y * cheby

        return [functools.partial(T_at, i) for i in range(self.K)]

    def _get_D_fraction(self):

        def d_quot(num_idx, denom_idx):
            num_idx += self.N  # j+N
            j_1, m_1, _ = self.__indices[num_idx]
            j_2, m_2, _ = self.__indices[denom_idx]
            p_1 = j_1 % 2
            p_2 = j_2 % 2
            epsilon = self.shape_param
            two_power_1 = (lambda x: 2, 3 + p_1 + 4 * m_1)
            two_power_2 = (lambda x: 0.5, 3 + p_2 + 4 * m_2)
            eps_power_1 = (lambda x: epsilon, 2 * j_1)
            eps_power_2 = (lambda x: 1/epsilon, 2 * j_2)
            num_fact_1 = (lambda x: x, (j_1 + p_1 + 2 * m_1) / 2)
            num_fact_2 = (lambda x: 1 / x, (j_2 + p_2 + 2 * m_2) / 2)
            denom_first_fact_1 = (lambda x: 1 / x, (j_1 - p_1 - 2 * m_1) / 2)
            denom_first_fact_2 = (lambda x: x, (j_2 - p_2 - 2 * m_2) / 2)
            denom_second_fact_1 = (lambda x: 1 / x, (j_1 + 1 + p_1 + 2 * m_1))
            denom_second_fact_2 = (lambda x: x, (j_2 + 1 + p_2 + 2 * m_2))
            product = RBF_QR_3D.__prodprod(two_power_1, two_power_2, eps_power_1, eps_power_2,
                                           num_fact_1, num_fact_2, denom_first_fact_1, denom_first_fact_2,
                                           denom_second_fact_1, denom_second_fact_2)
            return product

        D = np.empty((self.N, self.K - self.N))
        for i, j in np.ndindex(D.shape):
            D[i, j] = d_quot(j, i)
        return D

    @staticmethod
    def __prodprod(*args):
        maxidx = 0
        for pair in args:
            assert (len(pair) == 2)
            assert (int(pair[1]) == pair[1])
            maxidx = max(maxidx, pair[1])
        prod = 1
        for k in range(1, int(maxidx) + 1):
            for pair in args:
                prod *= pair[0](k) if pair[1] >= k else 1
        return prod

    @staticmethod
    # normalized legendre functions N^m_n
    def __normalized_legendre(m, n, x):
        upper_fact = (lambda x: x, n - m)
        lower_fact = (lambda x: 1 / x, n + m)
        factor = RBF_QR_3D.__prodprod(upper_fact, lower_fact)
        factor = (-1) ** m * math.sqrt(factor * (n + 0.5))
        if not np.isscalar(x):
            return np.array([lpmn(m, n, x_i)[0][-1, -1] for x_i in x])
        return lpmn(m, n, x)[0][-1, -1] * factor

    @staticmethod
    def __calc_Y(v, mu, theta, phi):
        assert (int(v) == v and int(mu) == mu)
        legendre = RBF_QR_3D.__normalized_legendre(abs(int(v)), int(mu), np.cos(theta))
        return legendre * (np.cos if v >= 0 else np.sin)(v * phi)

    @staticmethod
    # tested
    def hyp_pfq(upper, lower, x):
        eps = np.finfo(np.float64).eps
        alpha = 1
        sum = 1
        n = 1
        while alpha > eps:
            for up in upper:
                alpha *= up + n - 1
            for low in lower:
                alpha /= low + n - 1
            alpha *= x / n
            sum += alpha
            n += 1
        return sum
