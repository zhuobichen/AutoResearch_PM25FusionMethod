from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist


class MatrixCompute:
    """Matrix computation utilities matching C# MATLAB functions."""

    @staticmethod
    def inv(matrix: np.ndarray) -> tuple[bool, np.ndarray]:
        """Gauss-Jordan elimination for matrix inversion."""
        m, n = matrix.shape
        if m != n:
            raise ValueError("Matrix must be square.")
        result = matrix.copy().astype(float)
        pn_row = np.zeros(n, dtype=int)
        pn_col = np.zeros(n, dtype=int)

        for k in range(n):
            d = 0.0
            for i in range(k, n):
                for j in range(k, n):
                    p = abs(result[i, j])
                    if p > d:
                        d = p
                        pn_row[k] = i
                        pn_col[k] = j
            if d == 0.0 or d + 1.0 == 1.0:
                return False, np.zeros((n, n))

            if pn_row[k] != k:
                result[[k, pn_row[k]], :] = result[[pn_row[k], k], :]

            if pn_col[k] != k:
                result[:, [k, pn_col[k]]] = result[:, [pn_col[k], k]]

            result[k, k] = 1.0 / result[k, k]
            for j in range(n):
                if j != k:
                    result[k, j] *= result[k, k]

            for i in range(n):
                if i != k:
                    result[i, k] = -result[i, k] * result[k, k]

        for k in range(n - 1, -1, -1):
            if pn_col[k] != k:
                result[[k, pn_col[k]], :] = result[[pn_col[k], k], :]
            if pn_row[k] != k:
                result[:, [k, pn_row[k]]] = result[:, [pn_row[k], k]]

        return True, result

    @staticmethod
    def matrix_row_ismember_matrix_row(
        matrix1: np.ndarray, matrix2: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Check which rows of matrix1 are in matrix2."""
        m1 = np.asarray(matrix1, dtype=float)
        m2 = np.asarray(matrix2, dtype=float)
        if m1.ndim != 2 or m2.ndim != 2:
            raise ValueError("Inputs must be 2D matrices.")
        if m1.shape[1] != m2.shape[1]:
            raise ValueError("Inputs matrix1 and matrix2 must have the same number of columns.")

        row_map: dict[tuple, int] = {}
        for i in range(m2.shape[0]):
            key = tuple(m2[i, :].tolist())
            if key not in row_map:
                row_map[key] = i + 1  # 1-indexed

        bool_result = np.zeros(m1.shape[0], dtype=bool)
        int_result = np.zeros(m1.shape[0], dtype=int)
        for i in range(m1.shape[0]):
            key = tuple(m1[i, :].tolist())
            idx = row_map.get(key)
            if idx is not None:
                bool_result[i] = True
                int_result[i] = idx
        return bool_result, int_result

    @staticmethod
    def max_row_wise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row-wise max and index."""
        m = np.asarray(matrix, dtype=float)
        value_result = np.max(m, axis=1)
        index_result = np.argmax(m, axis=1) + 1  # 1-indexed
        return value_result, index_result

    @staticmethod
    def max_column_wise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Column-wise max and index."""
        m = np.asarray(matrix, dtype=float)
        value_result = np.max(m, axis=0)
        index_result = np.argmax(m, axis=0) + 1  # 1-indexed
        return value_result, index_result

    @staticmethod
    def min_row_wise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Row-wise min and index."""
        m = np.asarray(matrix, dtype=float)
        value_result = np.min(m, axis=1)
        index_result = np.argmin(m, axis=1) + 1  # 1-indexed
        return value_result, index_result

    @staticmethod
    def min_column_wise(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Column-wise min and index."""
        m = np.asarray(matrix, dtype=float)
        value_result = np.min(m, axis=0)
        index_result = np.argmin(m, axis=0) + 1  # 1-indexed
        return value_result, index_result

    @staticmethod
    def pdist2(matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        """Euclidean distance between two matrices."""
        m1 = np.asarray(matrix1, dtype=float)
        m2 = np.asarray(matrix2, dtype=float)
        if m1.ndim != 2 or m2.ndim != 2:
            raise ValueError("Inputs must be 2D matrices.")
        if m1.shape[1] != m2.shape[1]:
            raise ValueError("matrix1 and matrix2 must have the same number of columns.")
        return cdist(m1, m2, metric='euclidean')

    @staticmethod
    def regstats(x: np.ndarray, y: np.ndarray) -> "LinearRegressionResult | None":
        """Simple linear regression."""
        if x is None or y is None:
            return None
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        if x_arr.size == 0 or y_arr.size == 0 or x_arr.size != y_arr.size:
            return None

        result = LinearRegressionResult()
        l = x_arr.size
        xa = float(np.mean(x_arr))
        ya = float(np.mean(y_arr))

        lxx = float(np.sum((x_arr - xa) * (x_arr - xa)))
        lxy = float(np.sum((x_arr - xa) * (y_arr - ya)))

        result.coefficients.item2 = lxy / lxx if lxx != 0 else 0.0
        result.coefficients.item1 = ya - result.coefficients.item2 * xa

        for i in range(l):
            pred = result.coefficients.item1 + result.coefficients.item2 * x_arr[i]
            result.regression_sum_of_squares += (pred - ya) * (pred - ya)
            result.residual_sum_of_squares += (y_arr[i] - pred) * (y_arr[i] - pred)

        result.sum_of_squares_of_deviations = result.regression_sum_of_squares + result.residual_sum_of_squares
        result.regression_variance = result.regression_sum_of_squares
        try:
            result.residual_variance = result.residual_sum_of_squares / (l - 2) if l > 2 else 0.0
            if np.isinf(result.residual_variance) or np.isnan(result.residual_variance):
                result.residual_variance = 0.0
        except Exception:
            result.residual_variance = result.residual_sum_of_squares / l if l > 0 else 0.0

        result.standard_error = float(np.sqrt(result.residual_variance)) if result.residual_variance >= 0 else 0.0
        result.verification_f = result.regression_variance / result.residual_variance if result.residual_variance != 0 else np.inf
        result.correlation_coefficient = float(
            np.sqrt(result.regression_sum_of_squares / result.sum_of_squares_of_deviations)
        ) if result.sum_of_squares_of_deviations != 0 else 0.0
        return result

    @staticmethod
    def repmat(matrix: np.ndarray, m: int, n: int) -> np.ndarray:
        """Replicate matrix in block form."""
        mat = np.asarray(matrix, dtype=float)
        return np.tile(mat, (m, n))

    @staticmethod
    def sort_array(array: np.ndarray, ascend: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Sort array and return indices."""
        src = np.asarray(array, dtype=float).reshape(-1)
        order = np.argsort(src) if ascend else np.argsort(src)[::-1]
        value_result = src[order]
        index_result = order + 1  # 1-indexed
        return value_result, index_result

    @staticmethod
    def sort_column_wise(matrix: np.ndarray, ascend: bool = True) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Sort matrix columns."""
        m = np.asarray(matrix, dtype=float)
        row_num, col_num = m.shape
        value_result: list[np.ndarray] = []
        index_result: list[np.ndarray] = []

        for i in range(col_num):
            col = m[:, i].copy()
            values = np.sort(col) if ascend else np.sort(col)[::-1]
            idxs = np.zeros(row_num, dtype=int)
            source = col.copy()
            for j, v in enumerate(values):
                found = np.where(source == v)[0]
                if len(found) > 0:
                    idxs[j] = found[0] + 1  # 1-indexed
                    source[found[0]] = np.nan
            value_result.append(values)
            index_result.append(idxs)

        return value_result, index_result

    @staticmethod
    def sum_column_wise(matrix: np.ndarray) -> np.ndarray:
        """Column-wise sum."""
        m = np.asarray(matrix, dtype=float)
        if m.shape[0] == 1:
            return m.flatten()
        elif m.shape[1] == 1:
            return np.array([np.sum(m)])
        else:
            return np.sum(m, axis=0)

    @staticmethod
    def sum_row_wise(matrix: np.ndarray) -> np.ndarray:
        """Row-wise sum."""
        m = np.asarray(matrix, dtype=float)
        if m.shape[1] == 1:
            return m.flatten()
        elif m.shape[0] == 1:
            return np.array([np.sum(m)])
        else:
            return np.sum(m, axis=1)

    @staticmethod
    def unique_array(array: np.ndarray) -> np.ndarray:
        """Unique elements of array."""
        arr = np.asarray(array, dtype=float).reshape(-1)
        return np.unique(arr)

    @staticmethod
    def unique_row(matrix: np.ndarray) -> np.ndarray:
        """Unique rows of matrix."""
        m = np.asarray(matrix, dtype=float)
        if m.ndim != 2:
            raise ValueError("Input must be 2D matrix.")
        rows = [tuple(row.tolist()) for row in m]
        unique_rows = sorted(set(rows))
        result = np.array(unique_rows, dtype=float)
        return result

    @staticmethod
    def dense_of_column_major(row: int, col: int, data: np.ndarray) -> np.ndarray:
        """Create dense matrix from column-major data."""
        return np.asarray(data, dtype=float).reshape((row, col), order='F')


class Coefficients:
    def __init__(self, item1: float = 0.0, item2: float = 0.0):
        self.item1 = item1
        self.item2 = item2


class LinearRegressionResult:
    def __init__(self):
        self.regression_sum_of_squares: float = 0.0
        self.regression_variance: float = 0.0
        self.residual_sum_of_squares: float = 0.0
        self.residual_variance: float = 0.0
        self.sum_of_squares_of_deviations: float = 0.0
        self.standard_error: float = 0.0
        self.verification_f: float = 0.0
        self.correlation_coefficient: float = 0.0
        self.coefficients = Coefficients()