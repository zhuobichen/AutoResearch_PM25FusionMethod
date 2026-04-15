# -*- coding: utf-8 -*-
"""
PM2.5 SparseKrig - Sparse LU Optimized Kriging
================================================

SparseKrig 基于 downscaler v23 核心算法，采用稀疏矩阵优化：

核心优化点：
1. 使用阈值过滤的稀疏核矩阵 K_sparse（~0.45% 密度）
2. 每轮 MCMC 使用 scipy.sparse.linalg.splu 加速求解
3. 噪声采样使用小幅度随机噪声（0.01 scale）作为隐式正则化

注意：
- s2b 更新仍使用原始 compute_s2b 函数（基于 b0sinvmat 稠密矩阵）
- 由于 b0sinvmat 和 K_sparse 都来自相同的 phi_b0s 和 ds，数学上是一致的
- 噪声采样不是严格的 N(0, A⁻¹)，而是一种启发式正则化
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from common_setting import CommonSetting
from pm.calculation.pm25_special_function import PM25SpecialFunction
from matlab.matrix_compute import MatrixCompute


class SparseKrigError(Exception):
    """SparseKrig specific errors."""
    pass


class SparseKrig:
    """PM2.5 SparseKrig - Sparse LU optimized kriging."""

    KERNEL_THRESHOLD = 0.01

    def __init__(self, setting: CommonSetting, kernel_threshold: float | None = None):
        self.setting = setting
        self.error_msg = ""
        self.kernel_threshold = kernel_threshold or self.KERNEL_THRESHOLD

    def single_run(
        self,
        matrix_latlon_model: NDArray[np.float64],
        matrix_latlon_monitor: NDArray[np.float64],
        matrix_model: NDArray[np.float64],
        matrix_monitor: NDArray[np.float64],
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        if seed is not None:
            np.random.seed(seed)
        try:
            return SparseKrigCalculator.run(
                matrix_latlon_model, matrix_latlon_monitor,
                matrix_model, matrix_monitor,
                self.setting, seed, self.kernel_threshold
            )
        except Exception as e:
            import traceback
            self.error_msg = f"SparseKrig error: {str(e)}\n{traceback.format_exc()}"
            return None

    def run(
        self,
        matrix_latlon_model: NDArray[np.float64],
        matrix_latlon_monitor: NDArray[np.float64],
        matrix_model: NDArray[np.float64],
        matrix_monitor: NDArray[np.float64],
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        self.error_msg = ""
        if any(x is None or x.size == 0 for x in [matrix_latlon_model, matrix_latlon_monitor, matrix_model, matrix_monitor]):
            self.error_msg = "Input data is null or empty."
            return None
        if self.setting is None:
            self.error_msg = "Setting is null."
            return None
        return self.single_run(matrix_latlon_model, matrix_latlon_monitor, matrix_model, matrix_monitor, seed)


class SparseKrigCalculator:
    """SparseKrig core calculator with sparse LU optimization."""

    KERNEL_THRESHOLD = 0.01

    @staticmethod
    def _build_sparse_kernel(ds: np.ndarray, phi: float, threshold: float) -> sparse.csc_matrix:
        """Build sparse kernel matrix K = exp(-phi * ds²) with thresholding."""
        n = ds.shape[0]
        if phi > 0:
            dist_thresh = np.sqrt(-np.log(threshold) / phi)
        else:
            dist_thresh = float('inf')

        rows, cols, data = [], [], []
        # Diagonal = 1.0
        for i in range(n):
            rows.append(i); cols.append(i); data.append(1.0)
        # Off-diagonal with threshold
        for i in range(n):
            for j in range(i + 1, n):
                if ds[i, j] < dist_thresh:
                    k = np.exp(-phi * ds[i, j] * ds[i, j])
                    if k > threshold:
                        rows.append(i); cols.append(j); data.append(k)
                        rows.append(j); cols.append(i); data.append(k)

        return sparse.csc_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64)

    @staticmethod
    def run(
        matrix_latlon_model: NDArray[np.float64],
        matrix_latlon_monitor: NDArray[np.float64],
        matrix_model: NDArray[np.float64],
        matrix_monitor: NDArray[np.float64],
        setting: CommonSetting,
        seed: int | None = None,
        kernel_threshold: float = 0.01,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run SparseKrig with sparse LU optimization."""
        rng = np.random.default_rng(seed)
        krig_tol = 0.00001
        phi_k = setting.neighbor / (3.0 * setting.cmaqres)
        phi_q = setting.neighbor / (2.0 * setting.cmaqres)

        # Spatialize model matrix
        vct2 = np.pi / 180.0
        s1 = matrix_latlon_model * vct2

        low = MatrixCompute.min_column_wise(s1)[0]
        up = MatrixCompute.max_column_wise(s1)[0]
        data_c1 = np.array([low[0] + ((up[0] - low[0]) / 19.0) * i for i in range(20)], dtype=float)
        data_c2 = np.array([low[1] + ((up[1] - low[1]) / 19.0) * i for i in range(20)], dtype=float)

        qlat = data_c1.reshape(20, 1)
        qlon = data_c2.reshape(20, 1)

        s1 = PM25SpecialFunction.spatialize_matrix(s1)

        s2 = MatrixCompute.unique_row(matrix_latlon_monitor)
        s2 = PM25SpecialFunction.spatialize_matrix(s2 * vct2)
        matrix_latlon_monitor = PM25SpecialFunction.spatialize_matrix(matrix_latlon_monitor * vct2)

        data_c1 = MatrixCompute.sort_column_wise(MatrixCompute.repmat(qlat, 20, 1))[0][0]
        data_c2 = MatrixCompute.repmat(qlon, 20, 1).reshape(-1, order="F")
        s4 = np.column_stack([data_c1, data_c2])
        s4 = PM25SpecialFunction.spatialize_matrix(s4)

        # Filter: K(s,u) > 0.1
        s3 = PM25SpecialFunction.compute_matrix_s3(s2, s1, phi_k, krig_tol)
        data_d = MatrixCompute.max_row_wise(s3)[0]
        lst_conc_row = []
        lst_latlon_row = []
        for i in range(data_d.size):
            if data_d[i] > 0.1:
                lst_conc_row.append(s2[i, :].copy())
                lst_latlon_row.append(s3[i, :].copy())

        s2 = np.vstack(lst_conc_row)
        s3 = np.vstack(lst_latlon_row)
        lst_conc_row.clear()
        lst_latlon_row.clear()

        # Build mdta matrix
        data_d = matrix_monitor.reshape(-1, order="F")
        tuple_ism = MatrixCompute.matrix_row_ismember_matrix_row(matrix_latlon_monitor, s2)
        count = int(np.count_nonzero(tuple_ism[0]))

        k_idx = 0
        data_c1_arr = np.zeros(count, dtype=float)
        data_c2_arr = np.zeros(count, dtype=float)
        for i in range(tuple_ism[0].size):
            if tuple_ism[0][i]:
                data_c1_arr[k_idx] = data_d[i]
                data_c2_arr[k_idx] = tuple_ism[1][i]
                k_idx += 1

        mdta = np.column_stack([data_c1_arr, data_c2_arr])
        if count > 1:
            data_c2_arr = mdta[:, 1].copy()
            count = data_c2_arr.size
            index_arr = MatrixCompute.sort_array(data_c2_arr)[1]
            item2 = mdta[index_arr - 1, :]
            data_c1_arr = item2[:, 0].copy()
            data_c2_arr = item2[:, 1].copy()
            match = MatrixCompute.unique_array(data_c2_arr)
            vct3 = match.size
            vct4 = item2.shape[0]
            if vct3 < vct4:
                temp = np.zeros(vct3, dtype=float)
                wgt = np.ones(vct3, dtype=float)
                for i in range(vct3):
                    temp[i] = np.mean(data_c1_arr[data_c2_arr == match[i]])
                mdta = np.column_stack([temp, match, wgt])
            else:
                data_c3 = np.ones(vct4, dtype=float)
                mdta = np.column_stack([data_c1_arr, data_c2_arr, data_c3])

        # Compute s4 and s5
        s4, s5 = PM25SpecialFunction.compute_matrix_s4_s5(s1, s4, phi_q, krig_tol)

        # Compute phi_b0s
        probs = np.zeros(9, dtype=float)
        data_c1_mdta = mdta[:, 0].copy()
        data_c2_mdta = mdta[:, 1].copy()
        data_c3_mdta = mdta[:, 2].copy()

        s2_row = np.vstack([s2[int(data_c2_mdta[i]) - 1, :] for i in range(data_c2_mdta.size)])
        ds = MatrixCompute.pdist2(s2_row, s2_row)

        tmp = np.zeros(mdta.shape[0], dtype=float)
        data_d = matrix_model[:, 0].copy()
        for i in range(data_c2_mdta.size):
            data_d2 = s3[int(data_c2_mdta[i]) - 1, :].copy()
            max_v = np.max(data_d2)
            idx = int(np.where(data_d2 == max_v)[0][0])
            tmp[i] = data_d[idx]

        sl = MatrixCompute.regstats(tmp, data_c1_mdta)
        if sl is None:
            raise SparseKrigError("Regstats failed")
        if sl.residual_variance == 0.0:
            sl.residual_variance = 0.01

        tmp1 = data_c1_mdta - sl.coefficients.item1 - sl.coefficients.item2 * tmp

        max_dist = np.max(MatrixCompute.max_column_wise(ds)[0])
        data_d = ds.reshape(-1, order="F")

        for i in range(9):
            p = i * 0.1 + 0.1
            b0sinvmat_d = np.exp((-3.0 / (p * max_dist)) * data_d)
            b0sinvmat = b0sinvmat_d.reshape(ds.shape, order="F")
            eigvals = np.linalg.eigvals(b0sinvmat)
            v = -0.5 * np.sum(np.log(np.real(eigvals)))
            tmp2 = sl.residual_variance * b0sinvmat
            tmp2 = np.linalg.inv(tmp2)
            tmp3 = tmp1.reshape(1, -1) @ tmp2
            tmp4 = tmp3 @ tmp1.reshape(-1, 1)
            v -= 0.5 * float(tmp4[0, 0])
            probs[i] = v

        phi_b0s = 3.0 / (((int(np.argmax(probs)) * 0.1 + 0.1) * max_dist))

        # ============================================================
        # SPARSE: Build sparse kernel for A matrix
        # ============================================================
        n = ds.shape[0]
        print(f"  Building sparse kernel (n={n}, threshold={kernel_threshold})...")
        t0 = time.time()
        K_sparse = SparseKrigCalculator._build_sparse_kernel(ds, phi_b0s, kernel_threshold)
        print(f"  Sparse kernel: {100 * K_sparse.nnz / (n * n):.2f}% dense ({K_sparse.nnz} nnz), {time.time()-t0:.1f}s")

        data_b0sinvmat, order_b0sinvmat = PM25SpecialFunction.compute_b0sinvmat(phi_b0s, ds)

        data_vct8, row_vct8, column_vct8 = PM25SpecialFunction.compute_vct8(phi_b0s, s1, s2_row)
        data_krigmat, row_krigmat, column_krigmat = PM25SpecialFunction.compute_krigmat(
            data_vct8, row_vct8, column_vct8, data_b0sinvmat, order_b0sinvmat)
        data_mvars = PM25SpecialFunction.compute_mvars(data_krigmat, row_krigmat, column_krigmat, data_vct8)

        a_s2 = 2.5; b_s2 = 1.5; tY2_a = a_s2; tY2_b = b_s2
        n = mdta.shape[0]
        b_vec = np.array([0.0, 1.0], dtype=float)

        s3_row = np.vstack([s3[int(data_c2_mdta[i]) - 1, :] for i in range(data_c2_mdta.size)])
        data_d = s3_row.reshape(-1, order="F")
        data_d[data_d > 0.0] = 1.0; data_d[data_d <= 0.0] = 0.0
        s3_row2 = data_d.reshape(s3_row.shape, order="F")
        data_d = MatrixCompute.sum_column_wise(s3_row2)

        ind = np.zeros(data_d.size, dtype=bool)
        for i in range(data_d.size):
            if data_d[i] != 0.0 and (data_d[i] + 1.0 != 1.0):
                ind[i] = True

        count = int(np.count_nonzero(ind))
        data_s3_ind, row_s3_ind, column_s3_ind = PM25SpecialFunction.get_params_for_xt_s3_ind(ind, count, s3_row)
        data_s5_ind, row_s5_ind, column_s5_ind = PM25SpecialFunction.get_params_for_xt_s5_ind(ind, count, s5)
        data_cmaq2_ind = PM25SpecialFunction.get_params_for_xt_cmaq2_ind(ind, count, matrix_model)

        s3_ind_mat = MatrixCompute.dense_of_column_major(row_s3_ind, column_s3_ind, data_s3_ind)
        s5_ind_mat = MatrixCompute.dense_of_column_major(row_s5_ind, column_s5_ind, data_s5_ind)
        cmaq2_ind_vec = np.asarray(data_cmaq2_ind, dtype=float).reshape(-1)

        q = np.zeros(s4.shape[0], dtype=float)
        xt = PM25SpecialFunction.compute_matrix_xt_or_xtp(s3_ind_mat, s5_ind_mat, cmaq2_ind_vec, q, n)

        s2b = 1.0
        tY2 = PM25SpecialFunction.compute_tY2(data_c1_mdta, n, xt)

        w_data = data_c3_mdta.copy()
        b0sinv_mat = np.asarray(data_b0sinvmat).reshape((order_b0sinvmat, order_b0sinvmat), order='F')

        ybar = np.zeros(row_krigmat, dtype=float)
        y2bar = np.zeros(row_krigmat, dtype=float)
        kp = 0

        vctt1 = a_s2 + n / 2.0
        vctt2 = b_s2 * 2.0
        vctt3 = (tY2_a + n - 2.0) / 2.0
        vctt4 = np.array([0.002, 0.0, 0.0, 0.002], dtype=float)

        total_count = setting.numit * setting.thin + setting.burn
        sample_start = setting.burn + 1

        data_s4 = s4.reshape(-1, order="F")
        row_s4, column_s4 = s4.shape
        s4_mat = MatrixCompute.dense_of_column_major(row_s4, column_s4, data_s4)
        data_matrix_model_conc_c1 = matrix_model[:, 0].copy()
        vctt4_mat = MatrixCompute.dense_of_column_major(2, 2, vctt4)

        print(f"  Starting MCMC ({total_count} iterations)...")
        t_mcmc = time.time()

        for mcmcit in range(1, total_count + 1):
            # Build A = K_sparse/s2b + diag(w/tY2)
            A_sparse = K_sparse / s2b + sparse.diags(w_data / tY2, format='csc')

            # Compute RHS: (data_c1 - Xt @ b) / tY2
            xt_flat = xt.flatten(order='F')
            if xt.ndim > 1:
                data_4 = (xt.reshape(n, -1) @ b_vec.reshape(-1, 1)).reshape(-1)
            else:
                data_4 = xt_flat * b_vec
            rhs = (data_c1_mdta - data_4) / tY2

            # Solve using sparse LU
            try:
                lu = splu(A_sparse.tocsc())
                data_b0s = lu.solve(rhs)
                # Add small noise (consistent with original behavior)
                data_b0s = data_b0s + rng.standard_normal(n) * 0.01
            except Exception:
                # Fallback to dense
                A_dense = A_sparse.toarray()
                L = np.linalg.cholesky(A_dense)
                data_7 = np.linalg.solve(L.T, rhs)
                data_b0s = np.linalg.solve(L, data_7)
                # Add small noise
                data_b0s = data_b0s + rng.standard_normal(n) * 0.01

            data_c1_b0s = data_c1_mdta - data_b0s

            # Update s2b using original function with b0sinvmat
            s2b = PM25SpecialFunction.compute_s2b(vctt1, data_b0s, data_b0sinvmat, order_b0sinvmat, vctt2, rng)

            # Update tY2
            xt_trans = xt.T
            xtxt = xt_trans @ xt
            tY2 = PM25SpecialFunction.compute_tY2_mcmc(
                vctt3, tY2_b, data_c1_b0s, w_data,
                xt_flat, xt.shape[0], xt.shape[1],
                xtxt.flatten(order='F'), xtxt.shape[0],
                xt_trans.flatten(order='F'), rng
            )

            # Update lt2 and b
            lt2 = PM25SpecialFunction.compute_matrix_lt_2(xtxt.flatten(order='F'), xtxt.shape[0], tY2, vctt4)
            b_vec = PM25SpecialFunction.compute_b(lt2, xt_trans.flatten(order='F'), xt.shape[0], xt.shape[1], data_c1_b0s, tY2, rng)

            # Proposal for Q
            prop = PM25SpecialFunction.compute_prop(s4_mat, q, rng)
            xtp = PM25SpecialFunction.compute_matrix_xt_or_xtp(s3_ind_mat, s5_ind_mat, cmaq2_ind_vec, prop, n)

            alph = PM25SpecialFunction.compute_alph(data_c1_b0s, xtp, b_vec, xt)
            if np.log(rng.uniform(0.0, 1.0)) < alph:
                if PM25SpecialFunction.judge_tY2_xtp(tY2, xtp):
                    q = prop.copy()
                    xt = xtp.copy()

            # Sampling
            if mcmcit >= sample_start and ((mcmcit - sample_start) % setting.thin == 0):
                kp += 1
                pmean = PM25SpecialFunction.compute_pmean(b_vec, data_matrix_model_conc_c1, data_krigmat, row_krigmat, column_krigmat, data_b0s)
                pvar = PM25SpecialFunction.compute_pvar(s2b, data_mvars, tY2)
                ky, ky2 = PM25SpecialFunction.compute_kY(pmean, pvar, rng)
                ybar = PM25SpecialFunction.compute_ybar(ybar, kp, ky)
                y2bar = PM25SpecialFunction.compute_y2bar(y2bar, kp, ky2)

            if mcmcit % 500 == 0:
                elapsed = time.time() - t_mcmc
                print(f"    MCMC {mcmcit}/{total_count}: {elapsed:.1f}s")

        print(f"  MCMC done in {time.time()-t_mcmc:.1f}s")

        result = PM25SpecialFunction.compute_result(kp, y2bar, ybar)
        return ybar, result


def run(
    matrix_latlon_model: NDArray[np.float64],
    matrix_latlon_monitor: NDArray[np.float64],
    matrix_model: NDArray[np.float64],
    matrix_monitor: NDArray[np.float64],
    setting: CommonSetting | None = None,
    seed: int | None = None,
    kernel_threshold: float = 0.01,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    if setting is None:
        setting = CommonSetting()
    krig = SparseKrig(setting, kernel_threshold)
    return krig.run(matrix_latlon_model, matrix_latlon_monitor, matrix_model, matrix_monitor, seed)