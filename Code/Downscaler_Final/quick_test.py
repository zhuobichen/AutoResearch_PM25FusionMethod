"""
Quick test for PM2.5 Downscaler v23
Tests the basic functionality with simple synthetic data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from common_setting import CommonSetting
from pm25_downscaler import run

# Set random seed for reproducibility
np.random.seed(42)

# Create simple test data
# Model grid: 20x20 regular grid
lat_model = np.linspace(20, 40, 20)
lon_model = np.linspace(100, 120, 20)
lon_grid, lat_grid = np.meshgrid(lon_model, lat_model)
model_lat_lon = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

# Model PM2.5: simple spatial pattern + noise
model_pm = 50 + 10 * np.sin(lat_grid.ravel() * np.pi / 20) + \
           5 * np.cos(lon_grid.ravel() * np.pi / 20) + \
           np.random.randn(len(model_lat_lon)) * 2

# Monitor locations: 15 random locations from model grid
monitor_indices = np.random.choice(len(model_lat_lon), 15, replace=False)
monitor_lat_lon = model_lat_lon[monitor_indices]
monitor_pm = model_pm[monitor_indices] + np.random.randn(15) * 2

print("PM2.5 Downscaler v23 Quick Test")
print("=" * 50)
print(f"Model grid size: {len(model_lat_lon)}")
print(f"Monitor count: {len(monitor_lat_lon)}")
print(f"Default parameters: numit=2500, burn=500, thin=1")

# Run downscaler
setting = CommonSetting()
predictions, uncertainties = run(
    model_lat_lon, monitor_lat_lon,
    model_pm.reshape(-1, 1), monitor_pm.reshape(-1, 1),
    setting=setting,
    seed=42
)

if predictions is not None:
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    print(f"Uncertainties range: [{uncertainties.min():.2f}, {uncertainties.max():.2f}]")

    # Check for numerical stability
    if np.any(np.abs(predictions) > 1e10):
        print("\nWARNING: Predictions contain extreme values!")
    else:
        print("\nPredictions appear numerically stable.")
else:
    print("\nError: Downscaler returned None")