"""
PM2.5 Downscaler v23 - Main Entry Point

Conservative optimization based on v20 (stable version).
Default parameters: numit=2500, burn=500, thin=1, neighbor=3, cmaqres=12

Usage:
    from downscale import downscale_pm25

    predictions, uncertainties = downscale_pm25(
        model_lat_lon, monitor_lat_lon,
        model_pm, monitor_pm
    )
"""
import numpy as np
from numpy.typing import NDArray
from common_setting import CommonSetting
from pm25_downscaler import run


def downscale_pm25(
    model_lat_lon: NDArray[np.float64],
    monitor_lat_lon: NDArray[np.float64],
    model_pm: NDArray[np.float64],
    monitor_pm: NDArray[np.float64],
    numit: int = 2500,
    burn: int = 500,
    thin: int = 1,
    neighbor: int = 3,
    cmaqres: int = 12,
    seed: int | None = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Downscale PM2.5 using MCMC method.

    Parameters:
    -----------
    model_lat_lon : ndarray (N, 2)
        Model grid latitude/longitude
    monitor_lat_lon : ndarray (M, 2)
        Monitor latitude/longitude
    model_pm : ndarray (N,) or (N, 1)
        Model PM2.5 values at grid points
    monitor_pm : ndarray (M,) or (M, 1)
        Monitor PM2.5 values at stations
    numit : int
        Number of MCMC iterations (default: 2500)
    burn : int
        Burn-in period (default: 500)
    thin : int
        Thinning factor (default: 1)
    neighbor : int
        Number of neighbors (default: 3)
    cmaqres : int
        CMAQ resolution in km (default: 12)
    seed : int or None
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    predictions : ndarray (N,)
        Downscaled PM2.5 values at model grid
    uncertainties : ndarray (N,)
        Standard errors at model grid
    """
    # Reshape inputs
    model_lat_lon = np.atleast_2d(model_lat_lon)
    monitor_lat_lon = np.atleast_2d(monitor_lat_lon)
    model_pm = np.atleast_1d(model_pm).reshape(-1, 1)
    monitor_pm = np.atleast_1d(monitor_pm).reshape(-1, 1)

    # Create setting
    setting = CommonSetting()
    setting.numit = numit
    setting.burn = burn
    setting.thin = thin
    setting.neighbor = neighbor
    setting.cmaqres = cmaqres

    # Run downscaler
    result = run(
        matrix_latlon_model=model_lat_lon,
        matrix_latlon_monitor=monitor_lat_lon,
        matrix_model=model_pm,
        matrix_monitor=monitor_pm,
        setting=setting,
        seed=seed,
    )

    if result is None:
        raise RuntimeError("Downscaler failed. Check error messages.")

    return result


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)

    # Model grid: 20x20 regular grid
    lat_model = np.linspace(20, 40, 20)
    lon_model = np.linspace(100, 120, 20)
    lon_grid, lat_grid = np.meshgrid(lon_model, lat_model)
    model_lat_lon = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    # Model PM2.5: simple spatial pattern
    model_pm = 50 + 10 * np.sin(lat_grid.ravel() * np.pi / 20) + \
               5 * np.cos(lon_grid.ravel() * np.pi / 20)

    # Monitor locations: 15 random locations
    monitor_indices = np.random.choice(len(model_lat_lon), 15, replace=False)
    monitor_lat_lon = model_lat_lon[monitor_indices]
    monitor_pm = model_pm[monitor_indices] + np.random.randn(15) * 2

    print("PM2.5 Downscaler v23")
    print("=" * 50)
    print(f"Model grid size: {len(model_lat_lon)}")
    print(f"Monitor count: {len(monitor_lat_lon)}")

    predictions, uncertainties = downscale_pm25(
        model_lat_lon, monitor_lat_lon,
        model_pm, monitor_pm,
        seed=42
    )

    print(f"\nPredictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Uncertainties range: [{uncertainties.min():.2f}, {uncertainties.max():.2f}]")