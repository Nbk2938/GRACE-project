"""GRACE ice-mass vs. temperature co-variability analysis.

This script extends the Greenland-focused GRACE workflow by blending in
regional temperature information stored in GRIB format. It produces
area-weighted mean anomaly time series for user-defined subregions, then
quantifies how strongly mass anomalies co-vary with regional warming.

Typical usage (executed from the project root):

    python grace_temp_covariability.py \
        --grace-file Data/GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc \
        --temp-file Data/temp_data.grib \
        --temp-var t2m

Dependencies: numpy, pandas, netCDF4, xarray, cfgrib, scipy, scikit-learn, matplotlib.
Install cfgrib via "pip install cfgrib" if it is missing.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Project-level defaults ----------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_GRACE_FILE = PROJECT_DIR / "Data" / "GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc"
DEFAULT_TEMP_FILE = PROJECT_DIR / "Data" / "temp_data.grib"
DEFAULT_TEMP_VAR = "t2m"  # Update if the GRIB file uses a different short name

# Five coarse Greenland subregions + whole ice sheet bounds (lon_deg, lat_deg)
SUBREGIONS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "Greenland": {"lon": (-75.0, -10.0), "lat": (58.0, 85.0)},
    "Northwest": {"lon": (-75.0, -45.0), "lat": (72.0, 85.0)},
    "Northeast": {"lon": (-45.0, -10.0), "lat": (72.0, 85.0)},
    "Southwest": {"lon": (-75.0, -35.0), "lat": (58.0, 72.0)},
    "Southeast": {"lon": (-35.0, -10.0), "lat": (58.0, 72.0)},
}

@dataclass
class GraceField:
    data: np.ma.MaskedArray  # (time, lat, lon)
    lat: np.ndarray
    lon: np.ndarray
    dates: pd.DatetimeIndex


# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantify GRACE mass vs. temperature co-variability over Greenland."
    )
    parser.add_argument("--grace-file", type=Path, default=DEFAULT_GRACE_FILE)
    parser.add_argument("--temp-file", type=Path, default=DEFAULT_TEMP_FILE)
    parser.add_argument(
        "--temp-var",
        type=str,
        default=DEFAULT_TEMP_VAR,
        help="Variable name (shortName) inside the GRIB file, e.g., t2m",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs",
        help="Folder for tables and plots",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=36,
        help="Rolling window (months) for co-variability diagnostics",
    )
    return parser.parse_args()


def check_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def shift_longitudes_xr(da: xr.DataArray) -> xr.DataArray:
    """Convert 0-360 lon to -180/180 for easier Greenland subsetting (xarray)."""
    lon = da["lon"]
    lon_shifted = (((lon + 180) % 360) - 180).sortby("lon")
    da = da.assign_coords(lon=lon_shifted)
    return da.sortby("lon")


def shift_longitudes_array(lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    lon_shift = ((lon + 180) % 360) - 180
    sort_idx = np.argsort(lon_shift)
    return lon_shift[sort_idx], sort_idx


def monthly_anomalies(series: pd.Series) -> pd.Series:
    climatology = series.groupby(series.index.month).transform("mean")
    return series - climatology

"""def monthly_anomalies(series: pd.Series, ref_start="2004-01", ref_end="2009-12") -> pd.Series:
    ref = series.loc[ref_start:ref_end]
    climatology = ref.groupby(ref.index.month).mean()
    return series - series.index.month.map(climatology)"""


def area_weighted_mean(da: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(weights).mean(("lat", "lon"))


def subset_region(da: xr.DataArray, region: Dict[str, Tuple[float, float]]) -> xr.DataArray:
    lon_bounds = tuple(region["lon"])
    lat_bounds = tuple(region["lat"])

    lon_coord = da["lon"].values
    lat_coord = da["lat"].values

    lon_sorted = sorted(lon_bounds)
    lat_sorted = sorted(lat_bounds)

    lon_increasing = np.all(np.diff(lon_coord) >= 0)
    lat_increasing = np.all(np.diff(lat_coord) >= 0)

    if lon_increasing:
        lon_slice = slice(lon_sorted[0], lon_sorted[1])
    else:
        lon_slice = slice(lon_sorted[1], lon_sorted[0])

    if lat_increasing:
        lat_slice = slice(lat_sorted[0], lat_sorted[1])
    else:
        lat_slice = slice(lat_sorted[1], lat_sorted[0])

    return da.sel(lon=lon_slice, lat=lat_slice)


def load_grace_field(path: Path) -> GraceField:
    with nc.Dataset(path) as ds:
        if "lwe_thickness" not in ds.variables:
            raise KeyError("GRACE file is expected to contain 'lwe_thickness'")

        lwe_var = ds.variables["lwe_thickness"]
        raw_data = lwe_var[:]
        fill = getattr(lwe_var, "_FillValue", None)
        data = (
            np.ma.masked_where(raw_data == fill, raw_data)
            if fill is not None
            else np.ma.masked_invalid(raw_data)
        )

        lat = ds.variables["lat"][:]
        lon = ds.variables["lon"][:]

        time_var = ds.variables["time"]
        time_values = time_var[:]
        time_units = time_var.units
        time_cal = getattr(time_var, "calendar", "standard")
        dates = nc.num2date(time_values, units=time_units, calendar=time_cal)

    lon_shifted, sort_idx = shift_longitudes_array(lon)
    data = data[:, :, sort_idx]
    dates_py = [datetime(dt.year, dt.month, getattr(dt, "day", 1)) for dt in dates]
    return GraceField(data=data, lat=lat, lon=lon_shifted, dates=pd.DatetimeIndex(dates_py))


def load_temperature_field(path: Path, var: str) -> xr.DataArray:
    ds = xr.open_dataset(path, engine="cfgrib")
    
    # Rename coordinates if they are full names
    rename_dict = {}
    if "longitude" in ds.coords:
        rename_dict["longitude"] = "lon"
    if "latitude" in ds.coords:
        rename_dict["latitude"] = "lat"
    if rename_dict:
        ds = ds.rename(rename_dict)
        
    if var not in ds:
        raise KeyError(f"Temperature variable '{var}' not found. Available: {list(ds.data_vars)}")
    da = ds[var].load()
    if da.attrs.get("units", "").lower() in {"k", "kelvin"}:
        da = da - 273.15
        da.attrs["units"] = "degC"
    da = shift_longitudes_xr(da)
    return da


def build_region_series_grace(field: GraceField, label: str) -> Dict[str, pd.Series]:
    region_series: Dict[str, pd.Series] = {}
    dates = field.dates
    for name, bounds in SUBREGIONS.items():
        lon_min, lon_max = bounds["lon"]
        lat_min, lat_max = bounds["lat"]
        lat_mask = (field.lat >= lat_min) & (field.lat <= lat_max)
        lon_mask = (field.lon >= lon_min) & (field.lon <= lon_max)
        if not lat_mask.any() or not lon_mask.any():
            continue

        region_cube = field.data[:, lat_mask][:, :, lon_mask]
        if region_cube.count() == 0:
            continue

        weights_lat = np.cos(np.deg2rad(field.lat[lat_mask]))
        weights_2d = np.repeat(weights_lat[:, None], lon_mask.sum(), axis=1)
        weights = np.broadcast_to(weights_2d, region_cube.shape)
        weights_masked = np.ma.array(weights, mask=np.ma.getmaskarray(region_cube))
        weighted_vals = np.ma.average(
            region_cube, axis=(1, 2), weights=weights_masked
        ).filled(np.nan)

        series = pd.Series(weighted_vals, index=dates).dropna()
        if series.empty:
            continue
        series = series.sort_index()
        series = monthly_anomalies(series)
        region_series[name] = series.rename(label)
    return region_series


def build_region_series_xarray(da: xr.DataArray, label: str) -> Dict[str, pd.Series]:
    region_series: Dict[str, pd.Series] = {}
    da = da.sortby("time")
    time_start = pd.to_datetime(da.time.min().values)
    time_end = pd.to_datetime(da.time.max().values)
    da = da.sel(time=slice(time_start, time_end))
    da = da.resample(time="1MS").mean()
    for name, bounds in SUBREGIONS.items():
        region_da = subset_region(da, bounds)
        if region_da.size == 0:
            continue
        series = area_weighted_mean(region_da).to_series()
        series = series.sort_index()
        series = monthly_anomalies(series)
        region_series[name] = series.rename(label)
    return region_series


def build_region_series_grace_raw(field: GraceField, label: str) -> Dict[str, pd.Series]:
    """Like build_region_series_grace but WITHOUT removing the seasonal cycle."""
    region_series: Dict[str, pd.Series] = {}
    dates = field.dates
    for name, bounds in SUBREGIONS.items():
        lon_min, lon_max = bounds["lon"]
        lat_min, lat_max = bounds["lat"]
        lat_mask = (field.lat >= lat_min) & (field.lat <= lat_max)
        lon_mask = (field.lon >= lon_min) & (field.lon <= lon_max)
        if not lat_mask.any() or not lon_mask.any():
            continue
        region_cube = field.data[:, lat_mask][:, :, lon_mask]
        if region_cube.count() == 0:
            continue
        weights_lat = np.cos(np.deg2rad(field.lat[lat_mask]))
        weights_2d = np.repeat(weights_lat[:, None], lon_mask.sum(), axis=1)
        weights = np.broadcast_to(weights_2d, region_cube.shape)
        weights_masked = np.ma.array(weights, mask=np.ma.getmaskarray(region_cube))
        weighted_vals = np.ma.average(
            region_cube, axis=(1, 2), weights=weights_masked
        ).filled(np.nan)
        series = pd.Series(weighted_vals, index=dates).dropna().sort_index()
        if series.empty:
            continue
        region_series[name] = series.rename(label)
    return region_series


def build_region_series_xarray_raw(da: xr.DataArray, label: str) -> Dict[str, pd.Series]:
    """Like build_region_series_xarray but WITHOUT removing the seasonal cycle."""
    region_series: Dict[str, pd.Series] = {}
    da = da.sortby("time")
    time_start = pd.to_datetime(da.time.min().values)
    time_end = pd.to_datetime(da.time.max().values)
    da = da.sel(time=slice(time_start, time_end))
    da = da.resample(time="1MS").mean()
    for name, bounds in SUBREGIONS.items():
        region_da = subset_region(da, bounds)
        if region_da.size == 0:
            continue
        series = area_weighted_mean(region_da).to_series().sort_index()
        region_series[name] = series.rename(label)
    return region_series


# ===================== ANALYSIS PIPELINE =====================


def build_aligned_matrix(
    region_series: Dict[str, pd.Series],
) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Stack regional time series into a (T x R) DataFrame, dropping NaN rows.

    Returns the aligned DataFrame (columns = region names) and its DatetimeIndex.
    """
    df = pd.DataFrame(region_series).dropna()
    return df, df.index


def run_pca(
    df: pd.DataFrame, n_components: int | None = None
) -> Tuple[PCA, np.ndarray, pd.DataFrame, StandardScaler]:
    """Standardise columns and run PCA.

    Returns
    -------
    pca : fitted PCA object (eigenvalues, loadings, etc.)
    scores : (T x n_components) PC time-series array
    loadings_df : DataFrame with loadings (columns = PCs, index = region names)
    scaler : fitted StandardScaler (for reference / inverse transform)
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(df.values)  # (T, R) standardised
    if n_components is None:
        n_components = min(X.shape)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)  # (T, n_components)
    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=df.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    return pca, scores, loadings_df, scaler


def cross_correlate_pcs(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_pcs: int,
    max_lag: int = 12,
) -> Tuple[pd.DataFrame, Dict[Tuple[int, int], pd.Series]]:
    """Pearson r at lag 0 (matrix) and lagged cross-correlation for each PC pair.

    Parameters
    ----------
    scores_a, scores_b : (T, >=n_pcs) arrays of PC scores
    n_pcs : number of PCs to correlate
    max_lag : maximum lag (months) in both directions

    Returns
    -------
    corr_matrix : DataFrame (n_pcs x n_pcs) of zero-lag Pearson r
    lag_series : dict mapping (i, j) -> pd.Series indexed by lag
    """
    rows = []
    lag_series: Dict[Tuple[int, int], pd.Series] = {}
    for i in range(n_pcs):
        row = []
        for j in range(n_pcs):
            a = scores_a[:, i]
            b = scores_b[:, j]
            r, _ = pearsonr(a, b)
            row.append(r)
            # lagged correlation: positive lag = b leads a
            lags = range(-max_lag, max_lag + 1)
            lag_corrs = []
            for lag in lags:
                if lag > 0:
                    r_lag, _ = pearsonr(a[lag:], b[:-lag])
                elif lag < 0:
                    r_lag, _ = pearsonr(a[:lag], b[-lag:])
                else:
                    r_lag, _ = pearsonr(a, b)
                lag_corrs.append(r_lag)
            lag_series[(i, j)] = pd.Series(lag_corrs, index=list(lags))
        rows.append(row)
    pc_labels = [f"PC{k+1}" for k in range(n_pcs)]
    corr_matrix = pd.DataFrame(rows, index=pc_labels, columns=pc_labels)
    return corr_matrix, lag_series


def _align_by_month(s1: pd.Series, s2: pd.Series) -> pd.DataFrame:
    """Inner-join two series on (year, month), handling mid-month vs 1st dates."""
    df1 = s1.to_frame("mass_mm")
    df1.index = df1.index.to_period("M")
    df1 = df1[~df1.index.duplicated(keep="first")]
    df2 = s2.to_frame("temp_degC")
    df2.index = df2.index.to_period("M")
    df2 = df2[~df2.index.duplicated(keep="first")]
    return df1.join(df2, how="inner").dropna()


def direct_regional_correlation(
    grace_series: Dict[str, pd.Series], temp_series: Dict[str, pd.Series]
) -> pd.DataFrame:
    """Per-region Pearson r between mass and temperature anomalies."""
    rows = []
    for region in SUBREGIONS:
        if region not in grace_series or region not in temp_series:
            continue
        combined = _align_by_month(
            grace_series[region], temp_series[region]
        )
        if combined.shape[0] < 10:
            continue
        r, p = pearsonr(combined["mass_mm"], combined["temp_degC"])
        rows.append({
            "region": region,
            "n_months": len(combined),
            "pearson_r": r,
            "p_value": p,
        })
    if not rows:
        return pd.DataFrame(columns=["region", "n_months", "pearson_r", "p_value"])
    return pd.DataFrame(rows).sort_values("region")


# ===================== PLOTTING =====================


def plot_raw_timeseries(
    region_series: Dict[str, pd.Series],
    title: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Plot raw (non-anomaly) time series for each region on a single panel."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for idx, (name, series) in enumerate(region_series.items()):
        ax.plot(series.index, series.values, label=name,
                color=colors[idx % len(colors)], alpha=0.85, linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)


def plot_raw_timeseries_subplots(
    region_series: Dict[str, pd.Series],
    title: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> None:
    """One subplot per region with a linear trend line overlaid."""
    regions = list(region_series.keys())
    n = len(regions)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, name in enumerate(regions):
        ax = axes[idx]
        s = region_series[name]
        # numeric time for trend fitting
        t_num = (s.index - s.index[0]).days.astype(float)
        coeffs = np.polyfit(t_num, s.values, 1)
        trend_line = np.polyval(coeffs, t_num)
        trend_per_decade = coeffs[0] * 365.25 * 10  # unit/decade

        ax.plot(s.index, s.values, color=colors[idx % len(colors)],
                alpha=0.7, linewidth=0.9, label=name)
        ax.plot(s.index, trend_line, color="k", linewidth=1.5, linestyle="--",
                label=f"trend: {trend_per_decade:+.2f} /decade")
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time")
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_seasonal_cycle(
    region_series: Dict[str, pd.Series],
    title: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Plot mean seasonal cycle (climatology) for each region."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    for idx, (name, series) in enumerate(region_series.items()):
        clim = series.groupby(series.index.month).mean()
        ax.plot(clim.index, clim.values, marker="o", label=name,
                color=colors[idx % len(colors)], linewidth=1.5)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_xlabel("Month")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)


def plot_annual_mean(
    region_series: Dict[str, pd.Series],
    title: str,
    ylabel: str,
    filename: str,
    output_dir: Path,
) -> None:
    """Bar/line plot of annual mean values per region across all years."""
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    width = 0.15
    regions = list(region_series.keys())

    # Compute annual means for each region
    annual = {}
    all_years = set()
    for name, series in region_series.items():
        yr_mean = series.groupby(series.index.year).mean()
        annual[name] = yr_mean
        all_years.update(yr_mean.index)
    years = sorted(all_years)
    x = np.arange(len(years))

    for idx, name in enumerate(regions):
        vals = [annual[name].get(y, np.nan) for y in years]
        offset = (idx - len(regions) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=name,
               color=colors[idx % len(colors)], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=300)
    plt.close(fig)


def plot_scree(pca: PCA, label: str, output_dir: Path) -> None:
    """Scree plot: explained variance ratio per component."""
    n = len(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(1, n + 1), pca.explained_variance_ratio_ * 100, color="steelblue")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained variance (%)")
    ax.set_title(f"Scree plot — {label}")
    ax.set_xticks(range(1, n + 1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"scree_{label.lower().replace(' ', '_')}.png", dpi=300)
    plt.close(fig)


def plot_loadings(loadings_df: pd.DataFrame, label: str, output_dir: Path) -> None:
    """Grouped bar chart of PCA loadings per region."""
    n_pcs = loadings_df.shape[1]
    x = np.arange(len(loadings_df))
    width = 0.8 / n_pcs
    fig, ax = plt.subplots(figsize=(8, 4))
    for k in range(n_pcs):
        ax.bar(x + k * width, loadings_df.iloc[:, k], width, label=f"PC{k+1}")
    ax.set_xticks(x + width * (n_pcs - 1) / 2)
    ax.set_xticklabels(loadings_df.index, rotation=30, ha="right")
    ax.set_ylabel("Loading")
    ax.set_title(f"PCA loadings — {label}")
    ax.legend()
    ax.axhline(0, color="k", linewidth=0.5)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"loadings_{label.lower().replace(' ', '_')}.png", dpi=300)
    plt.close(fig)


def plot_pc_timeseries(
    scores: np.ndarray, dates: pd.DatetimeIndex, label: str, output_dir: Path, n_plot: int = 3
) -> None:
    """Plot first n_plot PC time series."""
    n_plot = min(n_plot, scores.shape[1])
    fig, axes = plt.subplots(n_plot, 1, figsize=(11, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    for k in range(n_plot):
        axes[k].plot(dates, scores[:, k], color=colors[k % len(colors)])
        axes[k].set_ylabel(f"PC{k+1}")
        axes[k].grid(alpha=0.3)
        axes[k].axhline(0, color="k", linewidth=0.5)
    axes[-1].set_xlabel("Time")
    fig.suptitle(f"PC time series — {label}")
    fig.tight_layout()
    fig.savefig(output_dir / f"pc_timeseries_{label.lower().replace(' ', '_')}.png", dpi=300)
    plt.close(fig)


def plot_cross_corr_matrix(corr_matrix: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of zero-lag cross-correlations between GRACE PCs and Temp PCs."""
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(corr_matrix.shape[1]))
    ax.set_xticklabels([f"Temp {c}" for c in corr_matrix.columns])
    ax.set_yticks(range(corr_matrix.shape[0]))
    ax.set_yticklabels([f"GRACE {c}" for c in corr_matrix.index])
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Cross-correlation: GRACE PCs vs Temperature PCs")
    fig.tight_layout()
    fig.savefig(output_dir / "cross_corr_matrix.png", dpi=300)
    plt.close(fig)


def plot_lag_correlation(
    lag_series: Dict[Tuple[int, int], pd.Series],
    pairs: list,
    output_dir: Path,
) -> None:
    """Plot lagged cross-correlation for selected PC pairs.

    pairs: list of (i, j) tuples (0-indexed PC indices)
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, j in pairs:
        s = lag_series[(i, j)]
        ax.plot(s.index, s.values, marker="o", markersize=3, label=f"GRACE PC{i+1} vs Temp PC{j+1}")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Lag (months, positive = temperature leads)")
    ax.set_ylabel("Pearson r")
    ax.set_title("Lagged cross-correlation of PC time series")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "lag_cross_correlation.png", dpi=300)
    plt.close(fig)


def plot_regional_scatter(
    grace_series: Dict[str, pd.Series],
    temp_series: Dict[str, pd.Series],
    output_dir: Path,
) -> None:
    """One scatter subplot per subregion: temperature anomaly vs mass anomaly."""
    regions = [r for r in SUBREGIONS if r in grace_series and r in temp_series]
    n = len(regions)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    for idx, region in enumerate(regions):
        ax = axes[idx // ncols, idx % ncols]
        combined = _align_by_month(grace_series[region], temp_series[region])
        ax.scatter(
            combined["temp_degC"], combined["mass_mm"],
            alpha=0.5, s=15, edgecolor="k", linewidth=0.3,
        )
        ax.set_xlabel("Temp anomaly (°C)")
        ax.set_ylabel("Mass anomaly (mm w.e.)")
        ax.set_title(region)
        ax.grid(alpha=0.3)
    # hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)
    fig.suptitle("Regional mass vs temperature anomalies")
    fig.tight_layout()
    fig.savefig(output_dir / "regional_scatter.png", dpi=300)
    plt.close(fig)


# ===================== MAIN =====================


def main() -> None:
    """
    Research question
    -----------------
    How strongly do Greenland ice-mass anomalies co-vary with regional
    warming, and does this relationship vary across different sub-regions?

    Analysis pipeline
    -----------------
    After loading and preprocessing (area-weighted monthly anomalies for
    5 Greenland subregions), the analysis proceeds in 4 steps:

    Step 1 — PCA on GRACE subregional mass anomalies
        Decompose the (T × 5) matrix of regional mass anomalies into
        principal components.  This reveals the dominant *spatial modes*
        of mass variability (e.g. uniform loss vs. NW-SE contrast).

    Step 2 — PCA on temperature subregional anomalies
        Same decomposition for temperature.  Identifies leading modes of
        regional warming (e.g. coherent warming vs. coastal gradients).

    Step 3 — Cross-correlation of GRACE and temperature PC time series
        Correlate GRACE PC_k with Temp PC_j at lag 0 and at lags up to
        ±12 months.  Strong correlations indicate that a particular
        spatial mode of mass change is driven by / co-varies with a
        particular spatial mode of warming.

    Step 4 — Direct per-region correlation
        Simple Pearson r between mass and temperature for each subregion.
        Complements PCA: shows which individual regions exhibit the
        strongest mass–temperature coupling.
    """

    # ── Load data ──────────────────────────────────────────────────────
    args = parse_args()
    check_file(args.grace_file, "GRACE file")
    check_file(args.temp_file, "temperature file")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grace_field = load_grace_field(args.grace_file)
    temp_field = load_temperature_field(args.temp_file, args.temp_var)

    # ── Pre-processing ─────────────────────────────────────────────────
    # Each dict maps region name → pd.Series of monthly anomalies
    #   grace_regions : mm w.e., seasonal cycle removed (full-period climatology)
    #   temp_regions  : °C,      seasonal cycle removed (full-period climatology)
    grace_regions = build_region_series_grace(grace_field, "mass_mm")
    temp_regions = build_region_series_xarray(temp_field, "temp_degC")

    # Also build raw (non-anomaly) series for descriptive plots
    grace_raw = build_region_series_grace_raw(grace_field, "mass_mm")
    temp_raw = build_region_series_xarray_raw(temp_field, "temp_degC")

    print(f"GRACE regions: {list(grace_regions.keys())}")
    print(f"Temp  regions: {list(temp_regions.keys())}")

    # ── Pre-analysis: descriptive plots of raw data ────────────────────
    print("\n=== Pre-analysis: Raw data overview ===")

    # 1) All regions on one panel
    plot_raw_timeseries(
        temp_raw,
        "Mean monthly temperature by region",
        "Temperature (°C)",
        "raw_temp_timeseries.png",
        args.output_dir,
    )
    plot_raw_timeseries(
        grace_raw,
        "GRACE liquid water equivalent thickness by region",
        "LWE thickness (mm)",
        "raw_grace_timeseries.png",
        args.output_dir,
    )

    # 2) Individual subplots with trend lines
    plot_raw_timeseries_subplots(
        temp_raw,
        "Temperature trend by region",
        "Temperature (°C)",
        "raw_temp_trends.png",
        args.output_dir,
    )
    plot_raw_timeseries_subplots(
        grace_raw,
        "GRACE mass trend by region",
        "LWE thickness (mm)",
        "raw_grace_trends.png",
        args.output_dir,
    )

    # 3) Seasonal cycles
    plot_seasonal_cycle(
        temp_raw,
        "Mean seasonal cycle — Temperature",
        "Temperature (°C)",
        "seasonal_cycle_temp.png",
        args.output_dir,
    )
    plot_seasonal_cycle(
        grace_raw,
        "Mean seasonal cycle — GRACE mass",
        "LWE thickness (mm)",
        "seasonal_cycle_grace.png",
        args.output_dir,
    )

    print("Pre-analysis plots saved.")

    # ── Step 1: PCA on GRACE mass anomalies ────────────────────────────
    grace_df, grace_dates = build_aligned_matrix(grace_regions)
    grace_pca, grace_scores, grace_loadings, _ = run_pca(grace_df)

    print("\n=== Step 1: GRACE PCA ===")
    print("Explained variance ratio:", np.round(grace_pca.explained_variance_ratio_, 4))
    print("Loadings:\n", grace_loadings.round(3))

    plot_scree(grace_pca, "GRACE mass", args.output_dir)
    plot_loadings(grace_loadings, "GRACE mass", args.output_dir)
    plot_pc_timeseries(grace_scores, grace_dates, "GRACE mass", args.output_dir)

    # ── Step 2: PCA on temperature anomalies ───────────────────────────
    temp_df, temp_dates = build_aligned_matrix(temp_regions)
    temp_pca, temp_scores, temp_loadings, _ = run_pca(temp_df)

    print("\n=== Step 2: Temperature PCA ===")
    print("Explained variance ratio:", np.round(temp_pca.explained_variance_ratio_, 4))
    print("Loadings:\n", temp_loadings.round(3))

    plot_scree(temp_pca, "Temperature", args.output_dir)
    plot_loadings(temp_loadings, "Temperature", args.output_dir)
    plot_pc_timeseries(temp_scores, temp_dates, "Temperature", args.output_dir)

    # ── Step 3: Cross-correlation of PC time series ────────────────────
    # Align GRACE and temperature PCs to overlapping months.
    # GRACE dates are mid-month, temp dates are 1st-of-month, so we
    # join on year-month period to get exactly matching rows.
    n_grace_pcs = grace_scores.shape[1]
    n_temp_pcs = temp_scores.shape[1]
    grace_pc_df = pd.DataFrame(
        grace_scores,
        index=grace_dates.to_period("M"),
        columns=[f"g_PC{i+1}" for i in range(n_grace_pcs)],
    )
    temp_pc_df = pd.DataFrame(
        temp_scores,
        index=temp_dates.to_period("M"),
        columns=[f"t_PC{i+1}" for i in range(n_temp_pcs)],
    )
    # Drop duplicate months (keep first) then inner-join
    grace_pc_df = grace_pc_df[~grace_pc_df.index.duplicated(keep="first")]
    temp_pc_df = temp_pc_df[~temp_pc_df.index.duplicated(keep="first")]
    joined = grace_pc_df.join(temp_pc_df, how="inner")
    grace_scores_aligned = joined[[c for c in joined if c.startswith("g_")]].values
    temp_scores_aligned = joined[[c for c in joined if c.startswith("t_")]].values
    n_pcs = min(
        grace_scores_aligned.shape[1],
        temp_scores_aligned.shape[1], 3
    )
    print(f"\nAligned PC series: {joined.shape[0]} common months")

    corr_matrix, lag_series = cross_correlate_pcs(
        grace_scores_aligned, temp_scores_aligned, n_pcs=n_pcs, max_lag=12
    )

    print("\n=== Step 3: GRACE PC vs Temperature PC cross-correlation (lag 0) ===")
    print(corr_matrix.round(3))

    plot_cross_corr_matrix(corr_matrix, args.output_dir)

    # Plot lagged correlation for strongest pairs
    # Find top pairs by absolute correlation at lag 0
    flat = corr_matrix.abs().values.flatten()
    n_pairs = min(3, len(flat))
    top_indices = np.argsort(flat)[::-1][:n_pairs]
    top_pairs = [
        (idx // corr_matrix.shape[1], idx % corr_matrix.shape[1])
        for idx in top_indices
    ]
    plot_lag_correlation(lag_series, top_pairs, args.output_dir)

    # ── Step 4: Direct per-region correlation ──────────────────────────
    region_corr = direct_regional_correlation(grace_regions, temp_regions)

    print("\n=== Step 4: Direct regional correlation ===")
    print(region_corr.to_string(index=False))

    region_corr.to_csv(args.output_dir / "regional_correlation_summary.csv", index=False)
    plot_regional_scatter(grace_regions, temp_regions, args.output_dir)

    # ── Save PCA summaries ─────────────────────────────────────────────
    grace_loadings.to_csv(args.output_dir / "grace_pca_loadings.csv")
    temp_loadings.to_csv(args.output_dir / "temp_pca_loadings.csv")
    corr_matrix.to_csv(args.output_dir / "pc_cross_correlation_lag0.csv")

    pd.DataFrame({
        "PC": [f"PC{i+1}" for i in range(len(grace_pca.explained_variance_ratio_))],
        "GRACE_var_explained": grace_pca.explained_variance_ratio_,
        "Temp_var_explained": temp_pca.explained_variance_ratio_,
    }).to_csv(args.output_dir / "pca_variance_explained.csv", index=False)

    print(f"\nAll outputs saved to {args.output_dir}")


if __name__ == "__main__":
    main()
