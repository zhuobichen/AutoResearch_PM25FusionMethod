"""
PM2.5 Downscaler v23

V23 - Conservative optimization based on V20 (stable version):
- Default parameters: numit=2500, burn=500, thin=1
- Core algorithm unchanged from V20
- Numerical stability floors preserved (tY2=0.1, residual variance=0.01)
- Sampling logic verified to match C# exactly

This version prioritizes stability over additional optimizations.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from common_setting import CommonSetting
from pm.calculation.pm25_downscaler_calculator import PM25DownscalerCalculator


class PM25Downscaler:
    """PM2.5 Downscaler implementation."""

    def __init__(self, setting: CommonSetting):
        self.setting = setting
        self.error_msg = ""

    def single_run(
        self,
        matrix_latlon_model: NDArray[np.float64],
        matrix_latlon_monitor: NDArray[np.float64],
        matrix_model: NDArray[np.float64],
        matrix_monitor: NDArray[np.float64],
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Single thread run of the downscaler algorithm."""
        if seed is not None:
            np.random.seed(seed)

        try:
            result = PM25DownscalerCalculator.run(
                matrix_latlon_model,
                matrix_latlon_monitor,
                matrix_model,
                matrix_monitor,
                self.setting,
                seed,
            )
            return result
        except Exception as e:
            import traceback
            self.error_msg = f"Single run DS error: {str(e)}\n{traceback.format_exc()}"
            return None

    def run(
        self,
        matrix_latlon_model: NDArray[np.float64],
        matrix_latlon_monitor: NDArray[np.float64],
        matrix_model: NDArray[np.float64],
        matrix_monitor: NDArray[np.float64],
        seed: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        """Run the downscaler algorithm.

        Args:
            matrix_latlon_model: Model lat/lon matrix (n_model x 2)
            matrix_latlon_monitor: Monitor lat/lon matrix (n_monitor x 2)
            matrix_model: Model concentration values (n_model x 1)
            matrix_monitor: Monitor concentration values (n_monitor x 1)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (prediction, sepred) arrays, or None on error
        """
        self.error_msg = ""

        # Validate inputs
        if matrix_latlon_model is None or matrix_latlon_model.size == 0:
            self.error_msg = "Model lat/lon data is null or empty."
            return None
        if matrix_latlon_monitor is None or matrix_latlon_monitor.size == 0:
            self.error_msg = "Monitor lat/lon data is null or empty."
            return None
        if matrix_model is None or matrix_model.size == 0:
            self.error_msg = "Model concentration data is null or empty."
            return None
        if matrix_monitor is None or matrix_monitor.size == 0:
            self.error_msg = "Monitor concentration data is null or empty."
            return None
        if self.setting is None:
            self.error_msg = "DS setting is null."
            return None

        return self.single_run(
            matrix_latlon_model,
            matrix_latlon_monitor,
            matrix_model,
            matrix_monitor,
            seed,
        )


def run(
    matrix_latlon_model: NDArray[np.float64],
    matrix_latlon_monitor: NDArray[np.float64],
    matrix_model: NDArray[np.float64],
    matrix_monitor: NDArray[np.float64],
    setting: CommonSetting | None = None,
    seed: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
    """Convenience function to run the downscaler.

    Args:
        matrix_latlon_model: Model lat/lon matrix (n_model x 2)
        matrix_latlon_monitor: Monitor lat/lon matrix (n_monitor x 2)
        matrix_model: Model concentration values (n_model x 1)
        matrix_monitor: Monitor concentration values (n_monitor x 1)
        setting: CommonSetting object (uses defaults if None)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (prediction, sepred) arrays, or None on error
    """
    if setting is None:
        setting = CommonSetting()
    downscaler = PM25Downscaler(setting)
    return downscaler.run(
        matrix_latlon_model,
        matrix_latlon_monitor,
        matrix_model,
        matrix_monitor,
        seed,
    )