"""
VNA, eVNA, aVNA Multi-day 10-fold Cross-validation
"""

import pandas as pd
import numpy as np
from scipy.spatial import Voronoi
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load all required data"""
    # Load monitor data
    monitor = pd.read_csv('test_data/raw/Monitor/2020_DailyPM2.5Monitor.csv')
    monitor['Date'] = pd.to_datetime(monitor['Date'])

    # Load fold table
    fold_table = pd.read_csv('test_data/fold_split_table.csv')

    # Load CMAQ data
    import netCDF4 as nc
    ds = nc.Dataset('test_data/raw/CMAQ/2020_PM25.nc')

    # Get CMAQ grid info
    cmaq_time = ds.variables['time'][:]
    cmaq_lat = ds.variables['lat'][:]
    cmaq_lon = ds.variables['lon'][:]
    cmaq_pred = ds.variables['pred_PM25'][:]

    ds.close()

    # Create CMAQ date mapping (time is days since 2020-01-01)
    cmaq_dates = pd.date_range('2020-01-01', periods=len(cmaq_time), freq='D')

    return monitor, fold_table, cmaq_dates, cmaq_lat, cmaq_lon, cmaq_pred


def get_cmaq_at_location(date, lat, lon, cmaq_dates, cmaq_lat, cmaq_lon, cmaq_pred):
    """Get CMAQ prediction at a specific location and date"""
    # Find date index
    date_idx = np.where(cmaq_dates == date)[0]
    if len(date_idx) == 0:
        return np.nan
    date_idx = date_idx[0]

    # Find nearest grid point
    lat_diff = np.abs(cmaq_lat - lat)
    lon_diff = np.abs(cmaq_lon - lon)
    y_idx, x_idx = np.unravel_index(np.argmin(lat_diff + lon_diff), lat_diff.shape)

    return cmaq_pred[date_idx, y_idx, x_idx]


def voronoi_neighbor_avg(train_sites, test_site, train_values, test_lat, test_lon):
    """
    Calculate Voronoi neighborhood average for a test site
    using training sites as Voronoi generators
    """
    # Get training site coordinates
    train_lats = train_sites['Lat'].values
    train_lons = train_sites['Lon'].values

    # Combine train and test sites for Voronoi diagram
    all_lats = np.append(train_lats, test_lat)
    all_lons = np.append(train_lons, test_lon)
    points = np.column_stack([all_lons, all_lats])

    try:
        vor = Voronoi(points)

        # Find the Voronoi region that contains the test point (last point)
        test_point_idx = len(train_sites)  # Index of test site

        # Find neighbors (sites whose Voronoi regions share boundary with test region)
        neighbors = []
        neighbor_values = []

        for ridge_idx, ridge_points in enumerate(vor.ridge_points):
            if test_point_idx in ridge_points:
                other_idx = ridge_points[0] if ridge_points[1] == test_point_idx else ridge_points[1]
                if other_idx < len(train_sites):  # It's a training site
                    neighbors.append(other_idx)
                    neighbor_values.append(train_values[other_idx])

        if len(neighbor_values) > 0:
            return np.mean(neighbor_values)
        else:
            # Fallback: use all training sites
            return np.mean(train_values)

    except Exception as e:
        # Fallback: use all training sites
        return np.mean(train_values)


def calculate_metrics(y_true, y_pred):
    """Calculate R², MAE, RMSE, MB"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Remove NaN
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan, np.nan

    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    # MAE
    mae = np.mean(np.abs(y_true - y_pred))

    # RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MB (Mean Bias)
    mb = np.mean(y_pred - y_true)

    return r2, mae, rmse, mb


def run_cv_for_day(monitor_day, fold_table, cmaq_dates, cmaq_lat, cmaq_lon, cmaq_pred, date):
    """Run 10-fold CV for a single day with all three methods"""

    results = []

    for fold in range(1, 11):
        # Split data
        test_sites = fold_table[fold_table['fold'] == fold]['Site'].values
        train_sites = fold_table[fold_table['fold'] != fold]['Site'].values

        # Get monitor data for this day
        day_data = monitor_day[monitor_day['Site'].isin(fold_table['Site'])].copy()

        train_data = day_data[day_data['Site'].isin(train_sites)]
        test_data = day_data[day_data['Site'].isin(test_sites)]

        if len(test_data) == 0 or len(train_data) == 0:
            continue

        # Get CMAQ values for all sites
        train_data = train_data.copy()
        test_data = test_data.copy()

        train_data['CMAQ'] = train_data.apply(
            lambda row: get_cmaq_at_location(date, row['Lat'], row['Lon'],
                                             cmaq_dates, cmaq_lat, cmaq_lon, cmaq_pred), axis=1)
        test_data['CMAQ'] = test_data.apply(
            lambda row: get_cmaq_at_location(date, row['Lat'], row['Lon'],
                                              cmaq_dates, cmaq_lat, cmaq_lon, cmaq_pred), axis=1)

        # Remove sites without valid CMAQ
        train_data = train_data.dropna(subset=['CMAQ'])
        test_data = test_data.dropna(subset=['CMAQ'])

        if len(test_data) == 0 or len(train_data) == 0:
            continue

        # Calculate VNA predictions for test sites
        vna_preds = []
        evna_preds = []
        avna_preds = []

        for _, row in test_data.iterrows():
            test_lat = row['Lat']
            test_lon = row['Lon']
            test_cmaq = row['CMAQ']

            # Get VNA using training data (monitor values)
            vna_val = voronoi_neighbor_avg(
                train_data[['Site', 'Lat', 'Lon']],
                row['Site'],
                train_data['Conc'].values,
                test_lat, test_lon
            )

            # Get VNA for CMAQ (using training sites' CMAQ values)
            vna_cmaq = voronoi_neighbor_avg(
                train_data[['Site', 'Lat', 'Lon']],
                row['Site'],
                train_data['CMAQ'].values,
                test_lat, test_lon
            )

            # VNA: directly use monitor-based Voronoi interpolation
            vna_preds.append(vna_val)

            # eVNA: O = M * r_n, where r_n = VNA(O) / VNA(M)
            if vna_cmaq != 0 and not np.isnan(vna_cmaq):
                r_n = vna_val / vna_cmaq
                evna_preds.append(test_cmaq * r_n)
            else:
                evna_preds.append(vna_val)

            # aVNA: O = M + bias, where bias = VNA(O) - VNA(M)
            bias = vna_val - vna_cmaq
            avna_preds.append(test_cmaq + bias)

        test_data = test_data.copy()
        test_data['VNA_pred'] = vna_preds
        test_data['eVNA_pred'] = evna_preds
        test_data['aVNA_pred'] = avna_preds

        # Calculate metrics for each method
        for method in ['VNA', 'eVNA', 'aVNA']:
            pred_col = f'{method}_pred'
            r2, mae, rmse, mb = calculate_metrics(
                test_data['Conc'].values,
                test_data[pred_col].values
            )

            results.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Fold': fold,
                'Method': method,
                'R2': r2,
                'MAE': mae,
                'RMSE': rmse,
                'MB': mb,
                'N_test': len(test_data)
            })

    return results


def main():
    print("Loading data...")
    monitor, fold_table, cmaq_dates, cmaq_lat, cmaq_lon, cmaq_pred = load_data()

    # Target dates
    dates = pd.date_range('2020-01-01', '2020-01-05', freq='D')

    print(f"Processing {len(dates)} days...")

    all_results = []

    for date in dates:
        print(f"  Processing {date.strftime('%Y-%m-%d')}...")
        monitor_day = monitor[monitor['Date'] == date]

        day_results = run_cv_for_day(
            monitor_day, fold_table, cmaq_dates, cmaq_lat, cmaq_lon, cmaq_pred, date
        )
        all_results.extend(day_results)

    # Create results DataFrame
    results_df = pd.DataFrame(all_results)

    # Calculate summary statistics
    summary = results_df.groupby(['Date', 'Method']).agg({
        'R2': 'mean',
        'MAE': 'mean',
        'RMSE': 'mean',
        'MB': 'mean',
        'N_test': 'sum'
    }).reset_index()

    summary.columns = ['Date', 'Method', 'R2_mean', 'MAE_mean', 'RMSE_mean', 'MB_mean', 'N_test']

    # Save results
    output_dir = 'test_result'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results_df.to_csv(f'{output_dir}/VNA_eVNA_aVNA_multi_day_results.csv', index=False)
    summary.to_csv(f'{output_dir}/VNA_eVNA_aVNA_summary.csv', index=False)

    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)

    # Print detailed results
    print("\n--- Daily Results by Method ---")
    pivot_r2 = results_df.pivot_table(values='R2', index='Date', columns='Method')
    pivot_mae = results_df.pivot_table(values='MAE', index='Date', columns='Method')
    pivot_rmse = results_df.pivot_table(values='RMSE', index='Date', columns='Method')
    pivot_mb = results_df.pivot_table(values='MB', index='Date', columns='Method')

    print("\nR^2 (higher is better):")
    print(pivot_r2.round(4).to_string())

    print("\nMAE (lower is better):")
    print(pivot_mae.round(4).to_string())

    print("\nRMSE (lower is better):")
    print(pivot_rmse.round(4).to_string())

    print("\nMB (closer to 0 is better):")
    print(pivot_mb.round(4).to_string())

    # Overall summary
    print("\n--- Overall Summary (Mean across days) ---")
    overall = results_df.groupby('Method').agg({
        'R2': 'mean',
        'MAE': 'mean',
        'RMSE': 'mean',
        'MB': 'mean'
    }).round(4)
    print(overall.to_string())

    print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
