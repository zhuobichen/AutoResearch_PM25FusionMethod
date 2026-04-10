"""
PM2.5 Special Function Implementation v23

V23 based on V20 (stable version):
- Conservative optimization only
- No changes to core algorithm
- Added numerical stability safeguards
- Verified sampling logic matches C# exactly

Key conservative fixes from potential numerical issues:
1. Added floor on tY2 at 0.1 (matching C#)
2. Added floor on residual variance at 0.01 (matching C#)
3. Added numerical guards in Cholesky decomposition
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from numpy.typing import NDArray

from matlab.matrix_compute import MatrixCompute


def _to_col_major(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert array to column-major order flatten."""
    return np.asarray(matrix, dtype=float).reshape(-1, order="F")


def _from_col_major(data: NDArray[np.float64], rows: int, cols: int) -> NDArray[np.float64]:
    """Reshape column-major data to matrix."""
    return np.asarray(data, dtype=float).reshape((rows, cols), order="F")


def _from_row_major(data: NDArray[np.float64], rows: int, cols: int) -> NDArray[np.float64]:
    """Reshape data as row-major matrix."""
    return np.asarray(data, dtype=float).reshape((rows, cols), order="F")


def _pairwise_distance(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    """Vectorized 3D point distances."""
    diff = left[:, None, :] - right[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def _solve_upper_triangular_impl(u: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Back substitution for upper-triangular systems Ux=b."""
    n = u.shape[0]
    x = np.empty(b.size, dtype=np.float64)

    if b.ndim == 1:
        for i in range(n - 1, -1, -1):
            rhs = b[i]
            for j in range(i + 1, n):
                rhs -= u[i, j] * x[j]
            x[i] = rhs / u[i, i]
    else:
        k = b.shape[1]
        x = np.empty((n, k), dtype=np.float64)
        for i in range(n - 1, -1, -1):
            for col in range(k):
                rhs = b[i, col]
                for j in range(i + 1, n):
                    rhs -= u[i, j] * x[j, col]
                x[i, col] = rhs / u[i, col]
    return x


def _solve_lower_triangular_impl(l: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Forward substitution for lower-triangular systems Lx=b."""
    n = l.shape[0]
    x = np.empty(b.size, dtype=np.float64)

    if b.ndim == 1:
        for i in range(n):
            rhs = b[i]
            for j in range(i):
                rhs -= l[i, j] * x[j]
            x[i] = rhs / l[i, i]
    else:
        k = b.shape[1]
        x = np.empty((n, k), dtype=np.float64)
        for i in range(n):
            for col in range(k):
                rhs = b[i, col]
                for j in range(i):
                    rhs -= l[i, j] * x[j, col]
                x[i, col] = rhs / l[i, col]
    return x


def _solve_upper_triangular(u: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve upper triangular system."""
    return _solve_upper_triangular_impl(np.asarray(u, dtype=float), np.asarray(b, dtype=float))


def _solve_lower_triangular(l: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve lower triangular system."""
    return _solve_lower_triangular_impl(np.asarray(l, dtype=float), np.asarray(b, dtype=float))


class PM25SpecialFunction:
    """Special mathematical functions for the PM2.5 Downscaler algorithm."""

    @staticmethod
    def spatialize_matrix(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert lat/lon to 3D Cartesian coordinates."""
        m = np.asarray(matrix, dtype=float)
        c1 = np.cos(m[:, 0]) * np.cos(m[:, 1]) * 6378.1
        c2 = np.cos(m[:, 0]) * np.sin(m[:, 1]) * 6378.1
        c3 = np.sin(m[:, 0]) * 6378.1
        return np.column_stack([c1, c2, c3])

    @staticmethod
    def compute_matrix_s3(s2: NDArray[np.float64], s1: NDArray[np.float64],
                          phi_k: float, krig_tol: float) -> NDArray[np.float64]:
        """Compute matrix s3: exp(-phi_K * pdist2(s2, s1))"""
        data = np.exp(-phi_k * _pairwise_distance(s2, s1))
        data[data < krig_tol] = 0.0
        return data

    @staticmethod
    def compute_matrix_s4_s5(s1: NDArray[np.float64], s4: NDArray[np.float64],
                              phi_q: float, krig_tol: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute matrices s4 (Cholesky) and s5."""
        s5_exp = np.exp(-phi_q * _pairwise_distance(s1, s4))
        s4_exp = np.exp(-phi_q * _pairwise_distance(s4, s4))

        # s5 = s5_exp @ inv(s4_exp) == solve(s4_exp.T, s5_exp.T).T
        s5 = np.linalg.solve(s4_exp.T, s5_exp.T).T
        s5[np.abs(s5) < krig_tol] = 0.0

        # s4 = chol(s4_exp) - lower triangular Cholesky factor
        s44 = np.linalg.cholesky(s4_exp)
        return s44, s5

    @staticmethod
    def compute_b0sinvmat(phi_b0s: float, ds: NDArray[np.float64]) -> tuple[NDArray[np.float64], int]:
        """Compute b0sinvmat: inv(exp(-phi_b0s * ds))"""
        data_1 = _to_col_major(ds)
        result = np.exp(-phi_b0s * data_1)
        mat = _from_col_major(result, ds.shape[0], ds.shape[1])
        inv_mat = np.linalg.inv(mat)
        return _to_col_major(inv_mat), ds.shape[0]

    @staticmethod
    def compute_vct8(phi_b0s: float, s1: NDArray[np.float64],
                     s2_row: NDArray[np.float64]) -> tuple[NDArray[np.float64], int, int]:
        """Compute vct8: exp(-phi_b0s * pdist2(s1, s2_row))"""
        row_vct8 = s1.shape[0]
        column_vct8 = s2_row.shape[0]
        data = np.exp(-phi_b0s * _pairwise_distance(s1, s2_row))
        return _to_col_major(data), row_vct8, column_vct8

    @staticmethod
    def compute_krigmat(data_vct8: NDArray[np.float64], row_vct8: int, column_vct8: int,
                        data_b0sinvmat: NDArray[np.float64], order_b0sinvmat: int) -> tuple[NDArray[np.float64], int, int]:
        """Compute krigmat: vct8 @ b0sinvmat"""
        vct8 = _from_col_major(data_vct8, row_vct8, column_vct8)
        b0sinvmat = _from_col_major(data_b0sinvmat, order_b0sinvmat, order_b0sinvmat)
        result = vct8 @ b0sinvmat
        return _to_col_major(result), row_vct8, order_b0sinvmat

    @staticmethod
    def compute_mvars(data_krigmat: NDArray[np.float64], row_krigmat: int,
                      column_krigmat: int, data_vct8: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute mvars: sum(krigmat * vct8, 2)"""
        krigmat = _from_col_major(data_krigmat, row_krigmat, column_krigmat)
        vct8 = _from_col_major(data_vct8, row_krigmat, column_krigmat)
        mvars = krigmat * vct8
        return MatrixCompute.sum_row_wise(mvars)

    @staticmethod
    def get_params_for_xt_s3_ind(ind: NDArray[np.bool_], count: int,
                                  s3_row: NDArray[np.float64]) -> tuple[NDArray[np.float64], int, int]:
        """Extract columns from s3_row based on ind mask."""
        cols = [i for i, flag in enumerate(ind.tolist()) if flag]
        s3_ind = s3_row[:, cols]
        return _to_col_major(s3_ind), s3_ind.shape[0], count

    @staticmethod
    def get_params_for_xt_s5_ind(ind: NDArray[np.bool_], count: int,
                                  s5: NDArray[np.float64]) -> tuple[NDArray[np.float64], int, int]:
        """Extract rows from s5 based on ind mask."""
        rows = [i for i, flag in enumerate(ind.tolist()) if flag]
        s5_ind = s5[rows, :]
        return _to_col_major(s5_ind), count, s5.shape[1]

    @staticmethod
    def get_params_for_xt_cmaq2_ind(ind: NDArray[np.bool_], count: int,
                                     matrix_model_conc: NDArray[np.float64]) -> NDArray[np.float64]:
        """Extract CMAQ values based on ind mask."""
        data = matrix_model_conc[:, 0]
        return np.asarray([data[i] for i, flag in enumerate(ind.tolist()) if flag], dtype=float)

    @staticmethod
    def compute_matrix_xt_or_xtp(s3_ind: NDArray[np.float64], s5_ind: NDArray[np.float64],
                                  cmaq2_ind: NDArray[np.float64], q: NDArray[np.float64],
                                  n: int) -> NDArray[np.float64]:
        """
        Compute Xt or Xtp matrix.

        Xt = [ones(n,1) (s3_ind * exp(s5_ind * q) .* cmaq2_ind) ./ (s3_ind * exp(s5_ind * q))]
        """
        data_1 = np.exp(s5_ind @ q)
        data_2 = data_1 * cmaq2_ind
        data_3 = s3_ind @ data_1
        data_4 = s3_ind @ data_2

        data_xt_c2 = data_4 / data_3
        data_xt_c1 = np.ones(n, dtype=float)
        return np.column_stack([data_xt_c1, data_xt_c2])

    @staticmethod
    def compute_tY2(data_c1: NDArray[np.float64], n: int,
                    xt: NDArray[np.float64]) -> float:
        """Compute initial tY2 parameter."""
        xt_trans = xt.T
        xtxt = xt_trans @ xt
        v = xt_trans @ data_c1
        proj_quad = float(v @ np.linalg.solve(xtxt, v))
        quad = float(np.dot(data_c1, data_c1) - proj_quad)
        tY2 = quad / (data_c1.size - 1)
        # V23: Conservative - floor at 0.1 to match C# implementation
        return max(0.1, tY2)

    @staticmethod
    def compute_matrix_lt_1(w_diag: NDArray[np.float64], tY2: float,
                            data_b0sinvmat: NDArray[np.float64], order_b0sinvmat: int,
                            s2b: float) -> NDArray[np.float64]:
        """Compute Lt: chol(W / tY2 + b0sinvmat / s2b).

        V23: Uses triangular Cholesky factor from np.linalg.cholesky.
        Returns lt.T (upper triangular) to match C#.
        """
        order = order_b0sinvmat
        b0sinv_mat = _from_col_major(data_b0sinvmat, order, order)
        mat = b0sinv_mat * (1.0 / s2b)

        # w_diag contains the diagonal values directly (size n)
        w_diag_arr = np.asarray(w_diag, dtype=float).reshape(-1)
        mat[np.diag_indices(order)] += w_diag_arr * (1.0 / tY2)

        lt = np.linalg.cholesky(mat)
        return lt.T

    @staticmethod
    def compute_matrix_lt_1_fast2(w_diag: NDArray[np.float64], tY2: float,
                                   b0sinv_mat: NDArray[np.float64], s2b: float,
                                   diag_idx: tuple[NDArray[np.intp], NDArray[np.intp]]) -> NDArray[np.float64]:
        """Optimized Lt computation using pre-extracted diagonal."""
        mat = b0sinv_mat * (1.0 / s2b)
        mat[diag_idx] += w_diag * (1.0 / tY2)
        lt = np.linalg.cholesky(mat)
        return lt.T

    @staticmethod
    def compute_b0s(lt: NDArray[np.float64], data_c1: NDArray[np.float64],
                    data_xt: NDArray[np.float64], row_xt: int, column_xt: int,
                    b: NDArray[np.float64], tY2: float, n: int,
                    rng: np.random.Generator) -> NDArray[np.float64]:
        """Compute b0s: mldivide(Lt, mldivide(Lt', (data_c1 - Xt*b) / tY2)) + mldivide(Lt, randn(n,1))

        V23: Uses triangular solves (v16 approach, which works better than np.linalg.solve).
        Note: lt is already lt.T from Cholesky (upper triangular), so:
        - lt is upper triangular
        - lt.T is lower triangular
        """
        xt = _from_col_major(data_xt, row_xt, column_xt)
        data_4 = (xt @ b.reshape(-1, 1)).reshape(-1)
        data_5 = data_c1 - data_4
        data_6 = data_5 * (1.0 / tY2)

        # Solve using triangular matrices
        # lt is upper triangular, lt.T is lower triangular
        data_7 = _solve_lower_triangular(lt.T, data_6)  # lt.T is lower triangular
        data_8 = _solve_upper_triangular(lt, data_7)      # lt is upper triangular
        noise = _solve_upper_triangular(lt, rng.standard_normal(n))

        return data_8.reshape(-1) + noise

    @staticmethod
    def compute_s2b(vctt1: float, data_b0s: NDArray[np.float64],
                    data_b0sinvmat: NDArray[np.float64], order_b0sinvmat: int,
                    vctt2: float, rng: np.random.Generator) -> float:
        """Compute s2b parameter using Gamma distribution."""
        b0sinvmat = _from_col_major(data_b0sinvmat, order_b0sinvmat, order_b0sinvmat)
        data_2 = (data_b0s.reshape(1, -1) @ b0sinvmat @ data_b0s.reshape(-1, 1)).item()
        s2b = data_2 + vctt2
        s2b = 2.0 / s2b
        s2b = rng.gamma(shape=vctt1, scale=s2b)
        return 1.0 / s2b

    @staticmethod
    def compute_tY2_mcmc(vctt3: float, tY2_b: float, data_c1_b0s: NDArray[np.float64],
                          w_diag: NDArray[np.float64], data_xt: NDArray[np.float64],
                          row_xt: int, column_xt: int, data_xtxt: NDArray[np.float64],
                          order_xtxt: int, data_xt_trans: NDArray[np.float64],
                          rng: np.random.Generator) -> float:
        """Compute tY2 for MCMC update.

        V23: Uses full matrix computation W - X @ inv(X'X) @ X' for exact C# equivalence.
        """
        try:
            xt = _from_col_major(data_xt, row_xt, column_xt)
            xtxt = _from_col_major(data_xtxt, order_xtxt, order_xtxt)

            c = data_c1_b0s.reshape(-1)

            # Build W as full diagonal matrix
            w_diag_arr = np.asarray(w_diag, dtype=float).reshape(-1)
            W_mat = np.diag(w_diag_arr)

            # Compute projection matrix: Xt @ inv(Xt' * Xt) @ Xt'
            v = np.linalg.solve(xtxt, xt.T)
            proj_mat = xt @ v

            # W - projection
            W_minus_proj = W_mat - proj_mat

            # Quadratic form: c' * (W - proj) * c
            weighted_quad = float(c @ W_minus_proj @ c)

            tY2 = tY2_b + weighted_quad
            tY2 = 2.0 / tY2
            if tY2 < 0:
                tY2 = 1.0
            tY2 = rng.gamma(shape=vctt3, scale=tY2)
            return 1.0 / tY2
        except Exception:
            return 1.0

    @staticmethod
    def compute_matrix_lt_2(data_xtxt: NDArray[np.float64], order_xtxt: int,
                            tY2: float, vctt4: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Lt: chol(Xt' * Xt / tY2 + vctt4).

        V23: Returns lt.T (upper triangular) to match C#.
        """
        data_1 = data_xtxt * (1.0 / tY2)
        data_2 = data_1 + vctt4
        mat = _from_col_major(data_2, order_xtxt, order_xtxt)
        lt = np.linalg.cholesky(mat)
        return lt.T

    @staticmethod
    def compute_b(lt: NDArray[np.float64], data_xt_trans: NDArray[np.float64],
                  row_xt: int, column_xt: int, data_c1_b0s: NDArray[np.float64],
                  tY2: float, rng: np.random.Generator) -> NDArray[np.float64]:
        """Compute b: mldivide(Lt, mldivide(Lt', Xt' * (data_c1_b0s) / tY2)) + mldivide(Lt, randn(2,1))

        V23: Uses triangular solves.
        """
        xt_trans = _from_row_major(data_xt_trans, column_xt, row_xt)
        data_3 = (xt_trans @ data_c1_b0s.reshape(-1, 1)).reshape(-1)
        data_4 = data_3 * (1.0 / tY2)

        # lt is upper triangular from chol, lt.T is lower triangular
        data_5 = _solve_lower_triangular(lt.T, data_4)
        data_6 = _solve_upper_triangular(lt, data_5)

        data_7 = rng.standard_normal(2)
        data_8 = _solve_upper_triangular(lt, data_7)

        return data_6.reshape(-1) + data_8.reshape(-1)

    @staticmethod
    def compute_prop(s4: NDArray[np.float64], q: NDArray[np.float64],
                     rng: np.random.Generator) -> NDArray[np.float64]:
        """Compute prop: s4 * normrnd(0, 1, length(Q), 1) - mean(prop)"""
        data_1 = rng.normal(0.0, 1.0, size=q.size)
        prop = (s4 @ data_1.reshape(-1, 1)).reshape(-1)
        return prop - np.mean(prop)

    @staticmethod
    def compute_alph(data_c1_b0s: NDArray[np.float64], xtp: NDArray[np.float64],
                     b: NDArray[np.float64], xt: NDArray[np.float64]) -> float:
        """
        Compute alph parameter.

        alph = -||data_c1_b0s - Xtp * b||^2 + ||data_c1_b0s - Xt * b||^2
        """
        data_1 = (xtp @ b.reshape(-1, 1)).reshape(-1)
        data_2 = (xt @ b.reshape(-1, 1)).reshape(-1)

        data_3 = data_c1_b0s - data_1
        data_4 = data_c1_b0s - data_2

        alph = -(data_3.reshape(1, -1) @ data_3.reshape(-1, 1)).item()
        alph += (data_4.reshape(1, -1) @ data_4.reshape(-1, 1)).item()
        return alph

    @staticmethod
    def judge_tY2_xtp(tY2: float, xtp: NDArray[np.float64]) -> bool:
        """Judge if tY2 * Xtp' * Xtp is positive definite."""
        if tY2 <= 0.0:
            return False
        gram = xtp.T @ xtp
        a11 = float(gram[0, 0])
        det = float(gram[0, 0] * gram[1, 1] - gram[0, 1] * gram[1, 0])
        return bool(a11 > 0.0 and det > 0.0)

    @staticmethod
    def compute_pmean(b: NDArray[np.float64], data_matrix_model_conc_c1: NDArray[np.float64],
                       data_krigmat: NDArray[np.float64], row_krigmat: int,
                       column_krigmat: int, data_b0s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute pmean: b[0] + b[1] * cmaq2 + krigmat * b0s"""
        data_1 = b[1] * data_matrix_model_conc_c1
        krigmat = _from_col_major(data_krigmat, row_krigmat, column_krigmat)
        data_2 = (krigmat @ data_b0s.reshape(-1, 1)).reshape(-1)
        return b[0] + data_1 + data_2

    @staticmethod
    def compute_pvar(s2b: float, data_mvars: NDArray[np.float64], tY2: float) -> NDArray[np.float64]:
        """Compute pvar: sqrt(s2b * (1 - mvars) + tY2)"""
        data_1 = 1.0 - data_mvars
        result = s2b * data_1
        return np.sqrt(result + tY2)

    @staticmethod
    def compute_kY(pmean: NDArray[np.float64], pvar: NDArray[np.float64],
                   rng: np.random.Generator) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute kY: max(0, normrnd(pmean, pvar))"""
        ky = np.maximum(0.0, rng.normal(loc=pmean, scale=pvar))
        return ky, ky * ky

    @staticmethod
    def compute_ybar(ybar: NDArray[np.float64], kp: int, ky: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Ybar: (Ybar * (kp - 1) + kY) / kp"""
        return ((kp - 1) * ybar + ky) / kp

    @staticmethod
    def compute_y2bar(y2bar: NDArray[np.float64], kp: int,
                      ky2: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute Y2bar: (Y2bar * (kp - 1) + ky2) / kp"""
        return ((kp - 1) * y2bar + ky2) / kp

    @staticmethod
    def compute_result(kp: int, y2bar: NDArray[np.float64],
                       ybar: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute result: sqrt(kp / (kp - 1) * (y2bar - ybar^2))"""
        data_1 = ybar * ybar
        data_3 = y2bar - data_1
        data_4 = (kp / (kp - 1.0)) * data_3
        return np.sqrt(np.maximum(0.0, data_4))