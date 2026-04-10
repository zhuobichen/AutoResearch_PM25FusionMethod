"""
MultiScaleGWR Analysis for Day 1 (2020-01-01) with 10-Fold Cross-Validation
"""
import numpy as np
import pandas as pd
import netCDF4
from sklearn.neighbors import NearestNeighbors
from scipy.interpolate import RegularGridInterpolator

# ==================== Data Loading ====================
print("Loading data...")

# Load monitor data
monitor_df = pd.read_csv('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
fold_df = pd.read_csv('test_data/fold_split_table.csv')

# Merge fold information
monitor_df = monitor_df.merge(fold_df, on='Site', how='left')

# Filter for first day (2020-01-01)
day1_df = monitor_df[monitor_df['Date'] == '2020-01-01'].copy()
print(f"Day 1 data: {len(day1_df)} monitoring sites")
print(f"Fold distribution:\n{day1_df['fold'].value_counts().sort_index()}")

# Load CMAQ data
nc_file = 'test_data/raw/CMAQ/2020_PM25.nc'
ds = netCDF4.Dataset(nc_file, 'r')
cmaq_lat = ds.variables['lat'][:]
cmaq_lon = ds.variables['lon'][:]
cmaq_pred = ds.variables['pred_PM25'][:]  # shape: (365, 127, 172)
ds.close()

# CMAQ data for day 1 (index 0)
cmaq_day1 = cmaq_pred[0]  # shape: (127, 172)

print(f"CMAQ grid: lat ({cmaq_lat.shape}), lon ({cmaq_lon.shape})")
print(f"CMAQ day 1 range: {cmaq_day1.min():.2f} - {cmaq_day1.max():.2f}")

# ==================== CMAQ Interpolation ====================
print("\nInterpolating CMAQ to monitor locations...")

# Create interpolator for CMAQ data
# lat is decreasing (from north to south), lon is increasing (west to east)
lat_points = cmaq_lat[:, 0]  # 127 unique latitudes
lon_points = cmaq_lon[0, :]  # 172 unique longitudes

interp = RegularGridInterpolator(
    (lat_points, lon_points),
    cmaq_day1,
    method='linear',
    bounds_error=False,
    fill_value=None
)

# Interpolate CMAQ for each monitor location
cmaq_values = interp(np.column_stack([day1_df['Lat'].values, day1_df['Lon'].values]))
day1_df['CMAQ'] = cmaq_values

# Remove rows with invalid CMAQ interpolation
valid_mask = ~np.isnan(day1_df['CMAQ'])
day1_df = day1_df[valid_mask].reset_index(drop=True)
print(f"Valid sites after CMAQ interpolation: {len(day1_df)}")

# ==================== MultiScaleGWR ====================
def run_multiscale_gwr(year_df, bandwidths=[0.5, 1.0, 2.0]):
    """
    MultiScaleGWR: Multi-Scale Geographically Weighted Regression
    """
    results = {fold_id: {} for fold_id in range(1, 11)}

    for fold_id in range(1, 11):
        train_df = year_df[year_df['fold'] != fold_id].copy()
        test_df = year_df[year_df['fold'] == fold_id].copy()

        coords_train = train_df[['Lon', 'Lat']].values
        m_train = train_df['CMAQ'].values
        y_train = train_df['Conc'].values

        coords_test = test_df[['Lon', 'Lat']].values
        m_test = test_df['CMAQ'].values

        n_scales = len(bandwidths)
        nn_finder = NearestNeighbors(n_neighbors=min(30, len(coords_train)), algorithm='ball_tree')
        nn_finder.fit(coords_train)

        ms_pred = np.zeros((len(test_df), n_scales))

        for b_idx, bandwidth in enumerate(bandwidths):
            for i in range(len(test_df)):
                dists, indices = nn_finder.kneighbors([coords_test[i]], n_neighbors=min(25, len(coords_train)))

                weights = np.exp(-0.5 * (dists[0] / bandwidth)**2)
                weights = weights / (weights.sum() + 1e-6)

                X_local = np.column_stack([np.ones(len(indices[0])), m_train[indices[0]]])
                y_local = y_train[indices[0]]

                W = np.diag(weights)
                try:
                    XTWX = X_local.T @ W @ X_local + np.eye(2) * 1e-6
                    XTWy = X_local.T @ W @ y_local
                    beta = np.linalg.solve(XTWX, XTWy)
                    ms_pred[i, b_idx] = beta[0] + beta[1] * m_test[i]
                except:
                    ms_pred[i, b_idx] = y_train.mean()

        # Multi-scale fusion (equal weights)
        scale_weights = np.ones(n_scales) / n_scales
        final_pred = np.sum(ms_pred * np.array(scale_weights).reshape(1, -1), axis=1)

        results[fold_id] = {
            'y_true': test_df['Conc'].values,
            'msgwr': final_pred
        }

        print(f"  Fold {fold_id}: {len(test_df)} test samples")

    return results

# ==================== Run Analysis ====================
print("\nRunning MultiScaleGWR with 10-fold CV...")
print(f"Bandwidths: [0.5, 1.0, 2.0]")

results = run_multiscale_gwr(day1_df, bandwidths=[0.5, 1.0, 2.0])

# ==================== Calculate Metrics ====================
print("\nCalculating metrics...")

def calculate_metrics(y_true, y_pred):
    """Calculate R², MAE, RMSE, MB"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-6))

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # RMSE
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MB (Mean Bias)
    mb = np.mean(y_pred - y_true)

    return r2, mae, rmse, mb

# Aggregate all predictions
all_y_true = []
all_y_pred = []

for fold_id in range(1, 11):
    all_y_true.extend(results[fold_id]['y_true'])
    all_y_pred.extend(results[fold_id]['msgwr'])

all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)

# Calculate overall metrics
r2, mae, rmse, mb = calculate_metrics(all_y_true, all_y_pred)

print("\n" + "="*50)
print("MultiScaleGWR 10-Fold Cross-Validation Results")
print("="*50)
print(f"Date: 2020-01-01")
print(f"Total samples: {len(all_y_true)}")
print(f"\nOverall Metrics:")
print(f"  R2:   {r2:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  MB:   {mb:.4f}")
print("="*50)

# ==================== Save Results ====================
output_df = pd.DataFrame({
    'y_true': all_y_true,
    'msgwr_pred': all_y_pred
})
output_df.to_csv('test_result/MultiScaleGWR_day1_results.csv', index=False)
print(f"\nResults saved to: test_result/MultiScaleGWR_day1_results.csv")

# Per-fold metrics
print("\nPer-Fold Metrics:")
print(f"{'Fold':>6} {'R2':>10} {'MAE':>10} {'RMSE':>10} {'MB':>10}")
print("-" * 46)
for fold_id in range(1, 11):
    y_t = results[fold_id]['y_true']
    y_p = results[fold_id]['msgwr']
    r2_f, mae_f, rmse_f, mb_f = calculate_metrics(y_t, y_p)
    print(f"{fold_id:>6} {r2_f:>10.4f} {mae_f:>10.4f} {rmse_f:>10.4f} {mb_f:>10.4f}")