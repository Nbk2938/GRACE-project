"""GRACE ice-mass vs. temperature co-variability analysis.

This script extends the Greenland-focused GRACE workflow by blending in
regional temperature information stored in GRIB format. It produces
area-weighted mean anomaly time series for user-defined subregions, then
quantifies how strongly mass anomalies co-vary with regional warming.

Typical usage (executed from the project root):

    python grace_temp_covariability.py \
        --grace-file \
        Data/GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc \
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

from animation_coimparison import animation_comparison

# Project-level defaults ----------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_GRACE_FILE = PROJECT_DIR / "Data" / "GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc"
DEFAULT_TEMP_FILE = PROJECT_DIR / "Data" / "temp_data.grib"
DEFAULT_TEMP_VAR = "t2m"  # Update if the GRIB file uses a different short name

# Greenland bounding box (lon_deg, lat_deg)
GREENLAND_BOUNDS: Dict[str, Tuple[float, float]] = {
    "lon": (-75.0, -10.0),
    "lat": (58.0, 85.0),
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
    field = GraceField(data=data, lat=lat, lon=lon_shifted, dates=pd.DatetimeIndex(dates_py))

    land_mask = build_greenland_land_mask(field.lon, field.lat)
    land_bool = np.nan_to_num(land_mask.values, nan=0.0).astype(bool)
    ocean_mask = ~land_bool
    existing_mask = np.ma.getmaskarray(field.data)
    land_mask_3d = np.broadcast_to(ocean_mask, field.data.shape)
    combined_mask = existing_mask | land_mask_3d
    field.data = np.ma.array(field.data, mask=combined_mask)

    return field


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



def build_greenland_land_mask(lon: np.ndarray, lat: np.ndarray) -> xr.DataArray:
    """Return a boolean mask (1 land, 0 ocean) for Greenland at the given grid."""
    try:
        from regionmask import defined_regions
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "regionmask is required for Greenland land masking."
            " Install it via `pip install regionmask`."
        ) from exc

    natural_earth = None
    for candidate in ("natural_earth_v5_1_2", "natural_earth_v5_0_0", "natural_earth_v4_1_0"):
        if hasattr(defined_regions, candidate):
            natural_earth = getattr(defined_regions, candidate)
            break
    if natural_earth is None:  # pragma: no cover - unexpected
        raise RuntimeError("No Natural Earth dataset found in regionmask")

    regions = natural_earth.countries_50
    try:
        greenland_idx = regions.names.index("Greenland")
    except ValueError as err:  # pragma: no cover - should not happen
        raise RuntimeError("Greenland polygon missing in Natural Earth data") from err

    mask = regions.mask(lon, lat)
    land = (mask == greenland_idx).astype(float)
    land = land.assign_coords({"lat": lat, "lon": lon})
    return land


# ===================== PDD COMPUTATION =====================


def compute_monthly_pdd_gridded(path: Path, var: str) -> xr.DataArray:
    """Compute monthly Positive Degree Days at every grid cell.

    PDD = sum of max(T_celsius, 0) * dt  for all sub-monthly time steps,
    where dt = 3 h / 24 h = 0.125 days (for 3-hourly ERA5 data).

    Returns
    -------
    xr.DataArray with dims (time_month, lat, lon) in units of °C·days.
    """
    print(f"  Loading temperature file: {path}")
    with xr.open_dataset(path, engine="cfgrib") as ds:
        rename = {}
        if "longitude" in ds.coords:
            rename["longitude"] = "lon"
        if "latitude" in ds.coords:
            rename["latitude"] = "lat"
        if rename:
            ds = ds.rename(rename)

        if var not in ds:
            raise KeyError(
                f"Temperature variable '{var}' not found in {path}. Available: {list(ds.data_vars)}"
            )

        da = ds[var]
        if da.attrs.get("units", "").lower() in {"k", "kelvin"}:
            da = da - 273.15
            da.attrs["units"] = "degC"

        da = shift_longitudes_xr(da)
    da = da.sortby("time")

    # Positive part only
    da_pos = da.where(da > 0, 0.0)

    # Determine time step in fraction of a day
    time_vals = da["time"].values
    dt_ns = np.diff(time_vals[:10]).mean()
    dt_days = dt_ns / np.timedelta64(1, "D")
    print(f"  Temperature time step: {dt_days:.4f} days "
          f"({dt_days * 24:.1f} hours)")

    temp_monthly = da.resample(time="1MS").mean("time")
    
    # Monthly sum → PDD (°C·days)
    pdd_monthly = (da_pos * dt_days).resample(time="1MS").sum("time")
    pdd_monthly.attrs["units"] = "degC days"
    pdd_monthly.attrs["long_name"] = "Positive Degree Days"

    land_mask = build_greenland_land_mask(
        pdd_monthly["lon"].values, pdd_monthly["lat"].values
    )
    pdd_monthly = pdd_monthly.where(land_mask == 1)
    temp_monthly = temp_monthly.where(land_mask == 1)

    return temp_monthly.load(), pdd_monthly.load()


@dataclass
class AllData:
    """Container for data prepared for independent PCA analysis.
    
    Note: GRACE and PDD have different spatial resolutions, so they are
    kept as separate grids. Only start/end times are aligned; missing
    months are preserved (as NaN/masked) for proper handling during PCA.
    """
    grace_data: np.ma.MaskedArray  # (time, lat, lon) - GRACE mass anomaly
    grace_dates: pd.DatetimeIndex
    grace_lat: np.ndarray
    grace_lon: np.ndarray

    grace_rate_data: np.ma.MaskedArray  # (time, lat, lon) - GRACE mass rate of change
    grace_rate_dates: pd.DatetimeIndex
    grace_rate_lat: np.ndarray
    grace_rate_lon: np.ndarray

    temp_data: xr.DataArray         # (time, lat, lon) - raw temperature 
    temp_dates: pd.DatetimeIndex
    temp_lat: np.ndarray
    temp_lon: np.ndarray

    pdd_data: xr.DataArray         # (time, lat, lon) - PDD anomaly
    pdd_dates: pd.DatetimeIndex
    pdd_lat: np.ndarray
    pdd_lon: np.ndarray

    pdd_anomaly_data: xr.DataArray  # (time, lat, lon) - PDD anomaly
    pdd_anomaly_dates: pd.DatetimeIndex
    pdd_anomaly_lat: np.ndarray
    pdd_anomaly_lon: np.ndarray
    
    time_range: Tuple[pd.Timestamp, pd.Timestamp]  # common start/end


def compute_grace_rate(
    grace_data: np.ma.MaskedArray,
    grace_dates: pd.DatetimeIndex,
) -> np.ma.MaskedArray:
    """Compute dM/dt from GRACE anomaly field on the same timestamps.

    Uses central differences for interior timestamps and one-sided
    forward/backward differences at the first/last timestamp. The time
    denominator uses actual spacing between available dates, so gaps are
    naturally accounted for.

    Returns
    -------
    np.ma.MaskedArray of shape (time, lat, lon) with units of anomaly per month.
    """
    arr = np.ma.asarray(grace_data)
    n_time = arr.shape[0]
    if n_time < 2:
        raise ValueError("Need at least two GRACE time steps to compute rate")

    dates = pd.DatetimeIndex(grace_dates)
    t_days = dates.view("i8") / (24 * 3600 * 1e9)

    # Convert day spacing to "months" using mean Gregorian month length.
    day_to_month = 1.0 / 30.436875

    rate = np.ma.masked_all(arr.shape, dtype=float)

    dt_fwd = (t_days[1] - t_days[0]) * day_to_month
    if dt_fwd != 0:
        rate[0] = (arr[1] - arr[0]) / dt_fwd

    for t in range(1, n_time - 1):
        dt_ctr = (t_days[t + 1] - t_days[t - 1]) * day_to_month
        if dt_ctr == 0:
            continue
        rate[t] = (arr[t + 1] - arr[t - 1]) / dt_ctr

    dt_bwd = (t_days[-1] - t_days[-2]) * day_to_month
    if dt_bwd != 0:
        rate[-1] = (arr[-1] - arr[-2]) / dt_bwd

    return rate


def load_prepare_data(grace_path: Path, temp_path: Path,temp_var: str = "t2m",) -> AllData:
    """Load GRACE and PDD data over Greenland for independent PCA.

    This function:
    1. Loads GRACE field with Greenland land mask applied
    2. Computes monthly PDD and its anomaly (also Greenland-masked)
    3. Aligns only the start and end times (keeps all months in between)
    4. Returns gridded data (not flattened) - each dataset keeps its resolution

    Parameters
    ----------
    grace_path : Path to GRACE netCDF file
    temp_path : Path to temperature GRIB file
    temp_var : Variable name in GRIB file (default: "t2m")

    Returns
    -------
    AllData with gridded arrays at their native resolutions
    """

    # RAW DATA LOADING ----------------------------------------------------------

    print("\n=== Loading GRACE field (Greenland-masked) ===")
    grace_field = load_grace_field(grace_path)
    print(f"  GRACE shape: {grace_field.data.shape} (time, lat, lon)")
    print(f"  GRACE time range: {grace_field.dates[0]} to {grace_field.dates[-1]}")
    print(f"  GRACE resolution: {len(grace_field.lat)} lat × {len(grace_field.lon)} lon")

    print("\n=== Computing gridded monthly PDD ===")
    temp_gridded, pdd_gridded = compute_monthly_pdd_gridded(temp_path, temp_var)
    print(f"  Temperature shape: {temp_gridded.shape}")
    print(f"  Temperature time range: {temp_gridded.time.values[0]} to {temp_gridded.time.values[-1]}")
    print(f"  Temperature resolution: {len(temp_gridded.lat)} lat × {len(temp_gridded.lon)} lon")
    print(f"  PDD shape: {pdd_gridded.shape}")
    print(f"  PDD time range: {pdd_gridded.time.values[0]} to {pdd_gridded.time.values[-1]}")
    print(f"  PDD resolution: {len(pdd_gridded.lat)} lat × {len(pdd_gridded.lon)} lon")

    # PROCESSING AND TIME ALIGNMENT ----------------------------------------------------------

    # Compute PDD anomaly (remove seasonal cycle)
    pdd_climatology = pdd_gridded.groupby("time.month").mean("time")
    pdd_anomaly = pdd_gridded.groupby("time.month") - pdd_climatology
    print("  PDD anomaly computed (seasonal cycle removed)")

    # Align only start and end times (keep all months in between, including gaps)
    grace_dates = grace_field.dates
    pdd_dates = pd.DatetimeIndex(pdd_gridded.time.values)
    temp_dates = pd.DatetimeIndex(temp_gridded.time.values)
    
    common_start = max(grace_dates.min(), pdd_dates.min(), temp_dates.min())
    common_end = min(grace_dates.max(), pdd_dates.max(), temp_dates.max())

    print("\n=== Aligning time range (start/end only) ===")
    print(f"  GRACE original: {grace_dates.min()} to {grace_dates.max()} ({len(grace_dates)} months)")
    print(f"  Temperature original: {temp_dates.min()} to {temp_dates.max()} ({len(temp_dates)} months)")
    print(f"  PDD original: {pdd_dates.min()} to {pdd_dates.max()} ({len(pdd_dates)} months)")
    print(f"  Common range: {common_start} to {common_end}")

    # Subset to common time range (but keep all months within that range)
    grace_time_mask = (grace_dates >= common_start) & (grace_dates <= common_end)
    grace_subset = grace_field.data[grace_time_mask, :, :]
    grace_dates_subset = grace_dates[grace_time_mask]
    grace_rate_subset = compute_grace_rate(grace_subset, grace_dates_subset)
    grace_rate_dates_subset = grace_dates_subset
    
    temp_subset = temp_gridded.sel(time=slice(common_start, common_end))
    temp_dates_subset = pd.DatetimeIndex(temp_subset.time.values)
    pdd_subset = pdd_gridded.sel(time=slice(common_start, common_end))
    pdd_dates_subset = pd.DatetimeIndex(pdd_subset.time.values)

    pdd_anomaly_subset = pdd_anomaly.sel(time=slice(common_start, common_end))
    pdd_anomaly_dates = pd.DatetimeIndex(pdd_anomaly_subset.time.values)

    print(f"  GRACE after trim: {len(grace_dates_subset)} months")
    print(f"  Temperature after trim: {len(temp_dates_subset)} months")
    print(f"  PDD after trim: {len(pdd_dates_subset)} months")
    print(f"  GRACE rate computed on: {len(grace_rate_dates_subset)} timestamps")
    
    # Count valid pixels in each dataset
    grace_valid_pixels = (~np.all(np.ma.getmaskarray(grace_subset), axis=0)).sum()
    temp_valid_pixels = (~np.all(np.isnan(temp_subset.values), axis=0)).sum()
    pdd_valid_pixels = (~np.all(np.isnan(pdd_subset.values), axis=0)).sum()
    print(f"  GRACE valid pixels: {grace_valid_pixels}")
    print(f"  Temperature valid pixels: {temp_valid_pixels}")
    print(f"  PDD valid pixels: {pdd_valid_pixels}")

    print("\n=== Data ready for independent PCA ===")
    print(f"  GRACE shape: {grace_subset.shape} (time, lat, lon)")
    print(f"  Temperature shape: {temp_subset.shape} (time, lat, lon)")
    print(f"  PDD shape: {pdd_subset.shape} (time, lat, lon)")


    return AllData(
        grace_data=grace_subset,
        grace_dates=grace_dates_subset,
        grace_lat=grace_field.lat,
        grace_lon=grace_field.lon,

        grace_rate_data=grace_rate_subset,  
        grace_rate_dates=grace_rate_dates_subset,
        grace_rate_lat=grace_field.lat,
        grace_rate_lon=grace_field.lon,

        temp_data=temp_subset,
        temp_dates=temp_dates_subset,
        temp_lat=temp_subset.lat.values,
        temp_lon=temp_subset.lon.values,

        pdd_data=pdd_subset,
        pdd_dates=pdd_dates_subset,
        pdd_lat=pdd_subset.lat.values,
        pdd_lon=pdd_subset.lon.values,

        pdd_anomaly_data=pdd_anomaly_subset,
        pdd_anomaly_dates=pdd_anomaly_dates,
        pdd_anomaly_lat=pdd_anomaly_subset.lat.values,
        pdd_anomaly_lon=pdd_anomaly_subset.lon.values,

        time_range=(common_start, common_end),
    )


# ===================== PLOTTING FUNCTIONS =====================

def plot_pdd_map(pdd: xr.DataArray,target_month: str,output_dir: Path, ) -> None:

    """Plot a map of PDD over Greenland for a specific month.

    Parameters
    ----------
    pdd : monthly gridded PDD (time, lat, lon)
    target_month : e.g. "2012-07" or "2019-08"
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    sel = pdd.sel(time=target_month)
    if sel.ndim == 3:
        sel = sel.isel(time=0)

    lat = sel["lat"].values
    lon = sel["lon"].values
    data = np.ma.masked_invalid(sel.values)

    if has_cartopy:
        proj = ccrs.NorthPolarStereo(central_longitude=-45)
        fig, ax = plt.subplots(
            figsize=(8, 8),
            subplot_kw={"projection": proj},
        )
        ax.set_extent([-75, -10, 58, 85], crs=ccrs.PlateCarree())
        im = ax.pcolormesh(
            lon, lat, data,
            transform=ccrs.PlateCarree(),
            cmap="YlOrRd", shading="auto",
        )
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle=":")
        ax.gridlines(draw_labels=True, alpha=0.3)
    else:
        # Fallback: plain pcolormesh without cartopy
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.pcolormesh(
            lon, lat, data,
            cmap="YlOrRd", shading="auto",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")

    cb = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.05)
    cb.set_label("PDD (°C·days)")
    ax.set_title(f"Positive Degree Days — {target_month}")
    fig.tight_layout()
    safe_name = target_month.replace("-", "")
    fig.savefig(
        output_dir / f"pdd_map_{safe_name}.png",
        dpi=300, bbox_inches="tight",
    )

    plt.close(fig)
    print(f"  Saved PDD map for {target_month}")

def plot_pdd_map_multi(pdd: xr.DataArray, months: list[str], output_dir: Path, ) -> None:
    """Plot a grid of PDD maps for multiple months."""
    
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    n = len(months)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))

    if has_cartopy:
        proj = ccrs.NorthPolarStereo(central_longitude=-45)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 6 * nrows),
            subplot_kw={"projection": proj}, squeeze=False,
        )
    else:
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(6 * ncols, 5 * nrows),
            squeeze=False,
        )

    # Global vmax for consistent color scale
    vmax = 0
    for m in months:
        sel = pdd.sel(time=m)
        if sel.ndim == 3:
            sel = sel.isel(time=0)
        vmax = max(vmax, float(np.nanmax(sel.values)))

    for idx, month in enumerate(months):
        ax = axes[idx // ncols, idx % ncols]
        sel = pdd.sel(time=month)
        if sel.ndim == 3:
            sel = sel.isel(time=0)

        lat = sel["lat"].values
        lon = sel["lon"].values
        data = np.ma.masked_invalid(sel.values)

        if has_cartopy:
            ax.set_extent([-75, -10, 58, 85], crs=ccrs.PlateCarree())
            im = ax.pcolormesh(
                lon, lat, data,
                transform=ccrs.PlateCarree(),
                cmap="YlOrRd", shading="auto",
                vmin=0, vmax=vmax,
            )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
            ax.gridlines(alpha=0.3)
        else:
            im = ax.pcolormesh(
                lon, lat, data,
                cmap="YlOrRd", shading="auto",
                vmin=0, vmax=vmax,
            )
            ax.set_aspect("equal")

        ax.set_title(month)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.colorbar(im, ax=axes, shrink=0.6, label="PDD (°C·days)")
    fig.suptitle(
        "Positive Degree Days by month", fontsize=14, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(
        output_dir / "pdd_maps_multi.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)
    print(f"  Saved multi-panel PDD map ({len(months)} months)")

def plot_triplet(
    data: AllData,
    target_month: str,
    output_dir: Path,
    selected_fields: tuple[str, str, str] = ("grace", "temperature", "pdd"),
) -> None:
    """Plot a 3-panel comparison for any selected Greenland fields.

    Valid field names are: "grace", "grace_rate", "temperature", "pdd", "pdd_anomaly".
    """

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        has_cartopy = True
    except ImportError:
        has_cartopy = False

    if len(selected_fields) != 3:
        raise ValueError("selected_fields must contain exactly 3 field names")

    target_ts = pd.Timestamp(target_month)

    field_catalog = {
        "grace": {
            "values": np.ma.asarray(data.grace_data),
            "dates": data.grace_dates,
            "lat": data.grace_lat,
            "lon": data.grace_lon,
            "cmap": "RdBu_r",
            "label": "LWE thickness anomaly",
            "title": "GRACE mass anomaly",
            "symmetric": True,
        },
        "grace_rate": {
            "values": np.ma.asarray(data.grace_rate_data),
            "dates": data.grace_rate_dates,
            "lat": data.grace_rate_lat,
            "lon": data.grace_rate_lon,
            "cmap": "RdBu_r",
            "label": "GRACE mass rate (anomaly/month)",
            "title": "GRACE mass change rate",
            "symmetric": True,
        },
        "temperature": {
            "values": np.asarray(data.temp_data.values),
            "dates": data.temp_dates,
            "lat": data.temp_lat,
            "lon": data.temp_lon,
            "cmap": "RdBu_r",
            "label": "Temperature (°C)",
            "title": "Monthly mean temperature",
            "symmetric": False,
        },
        "pdd": {
            "values": np.asarray(data.pdd_data.values),
            "dates": data.pdd_dates,
            "lat": data.pdd_lat,
            "lon": data.pdd_lon,
            "cmap": "YlOrRd",
            "label": "PDD (°C·days)",
            "title": "Monthly PDD",
            "symmetric": False,
        },
        "pdd_anomaly": {
            "values": np.asarray(data.pdd_anomaly_data.values),
            "dates": data.pdd_anomaly_dates,
            "lat": data.pdd_anomaly_lat,
            "lon": data.pdd_anomaly_lon,
            "cmap": "RdBu_r",
            "label": "PDD anomaly (°C·days)",
            "title": "Monthly PDD anomaly",
            "symmetric": True,
        },
    }

    invalid = [name for name in selected_fields if name not in field_catalog]
    if invalid:
        valid_fields = ", ".join(field_catalog.keys())
        raise ValueError(f"Unknown selected_fields {invalid}. Valid options: {valid_fields}")

    if has_cartopy:
        proj = ccrs.NorthPolarStereo(central_longitude=-45)
        fig, axes = plt.subplots(
            1,
            3,
            figsize=(18, 6),
            subplot_kw={"projection": proj},
            constrained_layout=True,
        )
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    panels = []
    for axis, field_name in zip(axes, selected_fields):
        config = field_catalog[field_name]
        dates = config["dates"]
        idx = int(np.argmin(np.abs(dates - target_ts)))
        actual_date = dates[idx]

        field_slice = np.ma.masked_invalid(np.asarray(config["values"][idx, :, :]))
        vals = field_slice.compressed() if np.ma.isMaskedArray(field_slice) else np.asarray(field_slice).ravel()
        vals = vals[np.isfinite(vals)]

        if vals.size == 0:
            vmin, vmax = -1.0, 1.0
        elif config["symmetric"]:
            vabs = float(np.nanmax(np.abs(vals)))
            vabs = vabs if np.isfinite(vabs) and vabs > 0 else 1.0
            vmin, vmax = -vabs, vabs
        else:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
            if vmax <= vmin:
                vmax = vmin + 1.0

        panels.append(
            {
                "ax": axis,
                "lon": config["lon"],
                "lat": config["lat"],
                "field": field_slice,
                "cmap": config["cmap"],
                "vmin": vmin,
                "vmax": vmax,
                "title": f"{config['title']}\n{actual_date.strftime('%Y-%m')}",
                "label": config["label"],
            }
        )

    for panel in panels:
        ax = panel["ax"]
        if has_cartopy:
            ax.set_extent([-75, -10, 58, 85], crs=ccrs.PlateCarree())
            image = ax.pcolormesh(
                panel["lon"],
                panel["lat"],
                panel["field"],
                transform=ccrs.PlateCarree(),
                cmap=panel["cmap"],
                shading="auto",
                vmin=panel["vmin"],
                vmax=panel["vmax"],
            )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
            ax.gridlines(alpha=0.3)
        else:
            image = ax.pcolormesh(
                panel["lon"],
                panel["lat"],
                panel["field"],
                cmap=panel["cmap"],
                shading="auto",
                vmin=panel["vmin"],
                vmax=panel["vmax"],
            )
            ax.set_aspect("equal")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        ax.set_title(panel["title"])
        colorbar = fig.colorbar(image, ax=ax, shrink=0.75, pad=0.03)
        colorbar.set_label(panel["label"])

    fig.suptitle(
        f"Greenland triplet comparison around {target_ts.strftime('%Y-%m')}"
        f" ({selected_fields[0]} | {selected_fields[1]} | {selected_fields[2]})",
        y=1.04,
    )

    out_name = (
        f"triplet_{selected_fields[0]}_{selected_fields[1]}_{selected_fields[2]}_"
        f"{target_ts.strftime('%Y%m')}.png"
    )
    fig.savefig(output_dir / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved triplet plot: {output_dir / out_name}")


def prepare_pca_inputs(
    field: np.ndarray | np.ma.MaskedArray,
    dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, tuple[int, int], dict[str, int]]:
    """Prepare a 3D field (time, lat, lon) for PCA.

    Strategy:
    1) Keep only spatial pixels with at least one finite value.
    2) Drop months containing any missing value across retained pixels.
    3) Drop pixels that still contain missing values after month filtering.
    """
    arr = np.asarray(np.ma.filled(field, np.nan), dtype=float)
    if arr.ndim != 3:
        raise ValueError("PCA input field must be 3D: (time, lat, lon)")

    n_time, n_lat, n_lon = arr.shape
    flat = arr.reshape(n_time, -1)

    spatial_any = np.any(np.isfinite(flat), axis=0)
    flat_any = flat[:, spatial_any]
    if flat_any.shape[1] == 0:
        raise ValueError("No valid spatial pixels available for PCA")

    valid_time = np.all(np.isfinite(flat_any), axis=1)
    flat_time = flat_any[valid_time, :]
    dates_time = dates[valid_time]
    if flat_time.shape[0] < 2:
        raise ValueError("Not enough valid months for PCA after dropping missing months")

    spatial_complete = np.all(np.isfinite(flat_time), axis=0)
    matrix = flat_time[:, spatial_complete]
    if matrix.shape[1] == 0:
        raise ValueError("No fully valid spatial pixels remain after month filtering")

    all_idx_any = np.where(spatial_any)[0]
    selected_flat_idx = all_idx_any[spatial_complete]

    meta = {
        "original_months": int(n_time),
        "kept_months": int(matrix.shape[0]),
        "dropped_months": int(n_time - matrix.shape[0]),
        "original_pixels": int(n_lat * n_lon),
        "kept_pixels": int(matrix.shape[1]),
        "dropped_pixels": int(n_lat * n_lon - matrix.shape[1]),
    }
    return matrix, pd.DatetimeIndex(dates_time), selected_flat_idx, (n_lat, n_lon), meta


def run_and_plot_pca(
    field: np.ndarray | np.ma.MaskedArray,
    dates: pd.DatetimeIndex,
    lat: np.ndarray,
    lon: np.ndarray,
    name: str,
    cmap:str,
    output_dir: Path,
    n_modes: int = 5,
) -> pd.DataFrame:
    """Run PCA and save one figure with EOF maps + PC time series."""
    matrix, kept_dates, flat_idx, grid_shape, meta = prepare_pca_inputs(field, dates)

    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)
    if lon_arr.ndim not in (1, 2) or lat_arr.ndim not in (1, 2):
        raise ValueError(
            f"Invalid coordinate dimensions for {name}: "
            f"lon ndim={lon_arr.ndim}, lat ndim={lat_arr.ndim}. Expected 1D or 2D."
        )

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    scaler = StandardScaler(with_mean=True, with_std=True)
    matrix_scaled = scaler.fit_transform(matrix)

    max_modes = min(n_modes, matrix_scaled.shape[0], matrix_scaled.shape[1])
    if max_modes < 1:
        raise ValueError(f"Cannot compute PCA modes for {name}")

    pca = PCA(n_components=max_modes)
    pcs = pca.fit_transform(matrix_scaled)

    eof_flat = np.full((max_modes, grid_shape[0] * grid_shape[1]), np.nan, dtype=float)
    eof_flat[:, flat_idx] = pca.components_
    eof_maps = eof_flat.reshape(max_modes, grid_shape[0], grid_shape[1])

    fig = plt.figure(figsize=(14, 3.9 * max_modes))
    grid = fig.add_gridspec(max_modes, 2)

    if has_cartopy:
        map_proj = ccrs.NorthPolarStereo(central_longitude=-45)
        map_transform = ccrs.PlateCarree()
    lon_min, lon_max = GREENLAND_BOUNDS["lon"]
    lat_min, lat_max = GREENLAND_BOUNDS["lat"]

    for mode in range(max_modes):
        var_exp = pca.explained_variance_ratio_[mode] * 100.0

        if has_cartopy:
            ax_map = fig.add_subplot(grid[mode, 0], projection=map_proj)
        else:
            ax_map = fig.add_subplot(grid[mode, 0])

        eof_mode = np.ma.masked_invalid(eof_maps[mode])
        eof_abs = np.nanmax(np.abs(eof_mode.filled(np.nan)))
        eof_vlim = float(eof_abs) if np.isfinite(eof_abs) and eof_abs > 0 else 1.0

        if has_cartopy:
            ax_map.set_extent([lon_min, lon_max, lat_min, lat_max], crs=map_transform)
            im = ax_map.pcolormesh(
                lon_arr,
                lat_arr,
                eof_mode,
                transform=map_transform,
                cmap=cmap,
                shading="auto",
                vmin=-eof_vlim,
                vmax=eof_vlim,
            )
            ax_map.add_feature(cfeature.COASTLINE, linewidth=0.7)
            ax_map.gridlines(alpha=0.3)
        else:
            im = ax_map.pcolormesh(
                lon_arr,
                lat_arr,
                eof_mode,
                cmap=cmap,
                shading="auto",
                vmin=-eof_vlim,
                vmax=eof_vlim,
            )
            ax_map.set_xlim(lon_min, lon_max)
            ax_map.set_ylim(lat_min, lat_max)
            ax_map.set_aspect("equal")
            ax_map.set_xlabel("Longitude")
            ax_map.set_ylabel("Latitude")

        ax_map.set_title(f"{name} EOF {mode + 1} ({var_exp:.1f}% var)")
        cbar = fig.colorbar(im, ax=ax_map, shrink=0.8, pad=0.02)
        cbar.set_label("EOF loading")

        ax_pc = fig.add_subplot(grid[mode, 1])
        ax_pc.plot(kept_dates, pcs[:, mode], color="black", linewidth=1.0)
        ax_pc.axhline(0.0, color="gray", linewidth=0.8)
        ax_pc.set_title(f"{name} PC {mode + 1}")
        ax_pc.set_xlabel("Time")
        ax_pc.set_ylabel("Standardized amplitude")

    fig.suptitle(
        f"{name} PCA (kept {meta['kept_months']}/{meta['original_months']} months, "
        f"{meta['kept_pixels']}/{meta['original_pixels']} pixels)",
        y=1.0,
    )
    fig.tight_layout()

    safe_name = name.lower().replace(" ", "_")
    fig.savefig(output_dir / f"{safe_name}_pca_eof_pc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(
        f"  Saved PCA figure for {name}: {output_dir / f'{safe_name}_pca_eof_pc.png'} "
        f"(dropped {meta['dropped_months']} months)"
    )

    variance_df = pd.DataFrame(
        {
            "mode": np.arange(1, max_modes + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_percent": pca.explained_variance_ratio_ * 100.0,
        }
    )
    variance_df.to_csv(output_dir / f"{safe_name}_pca_variance.csv", index=False)
    return variance_df


# ===================== ANALYSIS PIPELINE ======================

def run_pca_diagnostics(data: AllData, output_dir: Path, n_modes: int = 5) -> None:
    """Run PCA diagnostics for GRACE, monthly mean temperature, and PDD anomaly."""
    print("\n=== PCA diagnostics (first 5 modes) ===")

    run_and_plot_pca(
        field=data.grace_rate_data,
        dates=data.grace_rate_dates,
        lat=data.grace_rate_lat,
        lon=data.grace_rate_lon,
        name="GRACE-rate of change",
        cmap="RdBu_r",
        output_dir=output_dir,
        n_modes=n_modes,
    )
    run_and_plot_pca(
        field=data.temp_data.values,
        dates=data.temp_dates,
        lat=data.temp_lat,
        lon=data.temp_lon,
        name="Monthly Mean Temperature",
        cmap="RdBu_r",
        output_dir=output_dir,
        n_modes=n_modes,
    )
    run_and_plot_pca(
        field=data.pdd_data.values,
        dates=data.pdd_dates,
        lat=data.pdd_lat,
        lon=data.pdd_lon,
        name="Monthly PDD",
        cmap="YlOrRd",
        output_dir=output_dir,
        n_modes=n_modes,
    )


def run_direct_correlation_grace_rate_pdd(
    data: AllData,
    output_dir: Path,
    min_samples: int = 24,
) -> pd.DataFrame:
    """Correlate GRACE rate with monthly PDD at lag-0 on a common grid.

    Steps:
    1) Time-align to exact common months (handles GRACE missing months).
    2) Interpolate PDD from temperature grid to GRACE-rate grid.
    3) Compute per-pixel Pearson r/p with sample threshold.
    4) Compute regional area-weighted correlation and save diagnostics.
    """
    grace_dates = pd.DatetimeIndex(data.grace_rate_dates)
    pdd_dates = pd.DatetimeIndex(data.pdd_dates)

    # Align by calendar month (not exact day), because GRACE timestamps are
    # typically mid-month while PDD monthly products are month-start.
    grace_months = grace_dates.to_period("M")
    pdd_months = pdd_dates.to_period("M")

    grace_pos = pd.Series(np.arange(len(grace_months)), index=grace_months)
    pdd_pos = pd.Series(np.arange(len(pdd_months)), index=pdd_months)
    grace_pos = grace_pos[~grace_pos.index.duplicated(keep="first")]
    pdd_pos = pdd_pos[~pdd_pos.index.duplicated(keep="first")]

    common_months = grace_pos.index.intersection(pdd_pos.index)
    if len(common_months) < min_samples:
        raise ValueError(
            f"Not enough overlapping months for correlation: {len(common_months)} "
            f"(min_samples={min_samples})"
        )

    grace_idx = grace_pos.loc[common_months].to_numpy(dtype=int)
    pdd_idx = pdd_pos.loc[common_months].to_numpy(dtype=int)
    common_dates = common_months.to_timestamp(how="start")

    grace_common = np.asarray(
        np.ma.filled(data.grace_rate_data[grace_idx, :, :], np.nan),
        dtype=float,
    )
    grace_da = xr.DataArray(
        grace_common,
        coords={"time": common_dates, "lat": data.grace_rate_lat, "lon": data.grace_rate_lon},
        dims=("time", "lat", "lon"),
        name="grace_rate",
    )

    pdd_common = data.pdd_data.isel(time=pdd_idx)
    pdd_common = pdd_common.assign_coords(time=common_dates)
    pdd_on_grace = pdd_common.interp(
        lat=data.grace_rate_lat,
        lon=data.grace_rate_lon,
        method="linear",
    )

    grace_flat = grace_common.reshape(len(common_dates), -1)
    pdd_flat = np.asarray(pdd_on_grace.values, dtype=float).reshape(len(common_dates), -1)

    n_pixels = grace_flat.shape[1]
    r_flat = np.full(n_pixels, np.nan, dtype=float)
    p_flat = np.full(n_pixels, np.nan, dtype=float)
    n_flat = np.zeros(n_pixels, dtype=int)

    for pixel in range(n_pixels):
        valid = np.isfinite(grace_flat[:, pixel]) & np.isfinite(pdd_flat[:, pixel])
        n_valid = int(valid.sum())
        n_flat[pixel] = n_valid
        if n_valid < min_samples:
            continue
        r_val, p_val = pearsonr(grace_flat[valid, pixel], pdd_flat[valid, pixel])
        r_flat[pixel] = r_val
        p_flat[pixel] = p_val

    ny, nx = grace_common.shape[1:]
    r_map = r_flat.reshape(ny, nx)
    p_map = p_flat.reshape(ny, nx)
    n_map = n_flat.reshape(ny, nx)
    r2_percent_map = (r_map ** 2) * 100.0

    # Regional area-weighted lag-0 correlation on overlapping months
    grace_reg = area_weighted_mean(grace_da)
    pdd_reg = area_weighted_mean(pdd_on_grace)
    reg_df = pd.concat(
        [
            pd.Series(grace_reg.values, index=common_dates, name="grace_rate"),
            pd.Series(pdd_reg.values, index=common_dates, name="pdd"),
        ],
        axis=1,
    ).dropna()

    if len(reg_df) >= min_samples:
        reg_r, reg_p = pearsonr(reg_df["grace_rate"].values, reg_df["pdd"].values)
        reg_r2 = reg_r ** 2 * 100.0
    else:
        reg_r, reg_p, reg_r2 = np.nan, np.nan, np.nan

    valid_pixels = np.isfinite(r_map)
    sig_pixels = valid_pixels & (p_map < 0.05)
    summary = pd.DataFrame(
        {
            "common_months": [len(common_dates)],
            "regional_r": [reg_r],
            "regional_p": [reg_p],
            "regional_r2_percent": [reg_r2],
            "valid_pixels": [int(valid_pixels.sum())],
            "significant_pixels_p_lt_0_05": [int(sig_pixels.sum())],
            "significant_fraction_percent": [
                float(100.0 * sig_pixels.sum() / valid_pixels.sum()) if valid_pixels.sum() > 0 else np.nan
            ],
            "median_r": [float(np.nanmedian(r_map)) if valid_pixels.sum() > 0 else np.nan],
            "median_r2_percent": [float(np.nanmedian(r2_percent_map)) if valid_pixels.sum() > 0 else np.nan],
            "min_samples": [min_samples],
        }
    )

    summary.to_csv(output_dir / "grace_rate_vs_pdd_correlation_summary.csv", index=False)

    corr_ds = xr.Dataset(
        data_vars={
            "corr_r": (("lat", "lon"), r_map),
            "corr_p": (("lat", "lon"), p_map),
            "samples_n": (("lat", "lon"), n_map),
            "r2_percent": (("lat", "lon"), r2_percent_map),
        },
        coords={"lat": data.grace_rate_lat, "lon": data.grace_rate_lon},
    )
    corr_ds.to_netcdf(output_dir / "grace_rate_vs_pdd_correlation_maps.nc")

    # Compact table for strongest-matching pixels
    top_df = (
        corr_ds["corr_r"]
        .to_series()
        .dropna()
        .rename("corr_r")
        .reset_index()
    )
    if not top_df.empty:
        top_df["abs_r"] = top_df["corr_r"].abs()
        p_lookup = corr_ds["corr_p"].to_series().rename("corr_p").reset_index()
        n_lookup = corr_ds["samples_n"].to_series().rename("samples_n").reset_index()
        top_df = top_df.merge(p_lookup, on=["lat", "lon"], how="left")
        top_df = top_df.merge(n_lookup, on=["lat", "lon"], how="left")
        top_df = top_df.sort_values("abs_r", ascending=False).head(200)
        top_df.to_csv(output_dir / "grace_rate_vs_pdd_top_pixels.csv", index=False)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    if has_cartopy:
        proj = ccrs.NorthPolarStereo(central_longitude=-45)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 6),
            subplot_kw={"projection": proj},
            constrained_layout=True,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    panels = [
        {
            "ax": axes[0],
            "field": np.ma.masked_invalid(r_map),
            "cmap": "RdBu_r",
            "vmin": -1.0,
            "vmax": 1.0,
            "label": "Pearson r",
            "title": "GRACE-rate vs monthly PDD correlation (lag 0)",
        },
        {
            "ax": axes[1],
            "field": np.ma.masked_invalid(r2_percent_map),
            "cmap": "YlOrRd",
            "vmin": 0.0,
            "vmax": 100.0,
            "label": "Variance explained (r², %)",
            "title": "GRACE-rate vs monthly PDD variance explained",
        },
    ]

    sig_mask = np.isfinite(p_map) & (p_map < 0.05)
    nonsig_mask = np.isfinite(p_map) & ~sig_mask
    sig_overlay = np.where(sig_mask, 1.0, np.nan)
    nonsig_overlay = np.where(nonsig_mask, 1.0, np.nan)

    for panel in panels:
        ax = panel["ax"]
        if has_cartopy:
            ax.set_extent([-75, -10, 58, 85], crs=ccrs.PlateCarree())
            im = ax.pcolormesh(
                data.grace_rate_lon,
                data.grace_rate_lat,
                panel["field"],
                transform=ccrs.PlateCarree(),
                cmap=panel["cmap"],
                shading="auto",
                vmin=panel["vmin"],
                vmax=panel["vmax"],
            )
            if np.any(nonsig_mask):
                ax.contourf(
                    data.grace_rate_lon,
                    data.grace_rate_lat,
                    nonsig_overlay,
                    levels=[0.5, 1.5],
                    colors=["lightgray"],
                    alpha=0.4,
                    transform=ccrs.PlateCarree(),
                )
            if np.any(sig_mask):
                ax.contourf(
                    data.grace_rate_lon,
                    data.grace_rate_lat,
                    sig_overlay,
                    levels=[0.5, 1.0],
                    colors="none",
                    hatches=["...."],
                    transform=ccrs.PlateCarree(),
                )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
            ax.gridlines(alpha=0.3)
        else:
            im = ax.pcolormesh(
                data.grace_rate_lon,
                data.grace_rate_lat,
                panel["field"],
                cmap=panel["cmap"],
                shading="auto",
                vmin=panel["vmin"],
                vmax=panel["vmax"],
            )
            if np.any(nonsig_mask):
                ax.contourf(
                    data.grace_rate_lon,
                    data.grace_rate_lat,
                    nonsig_overlay,
                    levels=[0.5, 1.5],
                    colors=["lightgray"],
                    alpha=0.4,
                )
            if np.any(sig_mask):
                ax.contourf(
                    data.grace_rate_lon,
                    data.grace_rate_lat,
                    sig_overlay,
                    levels=[0.5, 1.0],
                    colors="none",
                    hatches=["...."],
                )
            ax.set_aspect("equal")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        ax.set_title(panel["title"] + "\n(hatched: p < 0.05, gray: non-significant)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label(panel["label"])

    fig.savefig(output_dir / "grace_rate_vs_pdd_correlation_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        "  Saved direct-correlation outputs: "
        "grace_rate_vs_pdd_correlation_summary.csv, "
        "grace_rate_vs_pdd_correlation_maps.nc, "
        "grace_rate_vs_pdd_correlation_maps.png"
    )
    return summary


def run_lag_analysis_grace_rate_pdd(
    data: AllData,
    output_dir: Path,
    max_lag: int = 6,
    min_samples: int = 24,
) -> pd.DataFrame:
    """Run lagged correlation of GRACE-rate vs monthly PDD.

    Lag convention used here:
    - lag L means corr(PDD_t, GRACE-rate_(t+L)), so PDD leads GRACE-rate.
    """
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")

    grace_dates = pd.DatetimeIndex(data.grace_rate_dates)
    pdd_dates = pd.DatetimeIndex(data.pdd_dates)

    grace_months = grace_dates.to_period("M")
    pdd_months = pdd_dates.to_period("M")

    grace_pos = pd.Series(np.arange(len(grace_months)), index=grace_months)
    pdd_pos = pd.Series(np.arange(len(pdd_months)), index=pdd_months)
    grace_pos = grace_pos[~grace_pos.index.duplicated(keep="first")]
    pdd_pos = pdd_pos[~pdd_pos.index.duplicated(keep="first")]

    common_months = grace_pos.index.intersection(pdd_pos.index)
    if len(common_months) < (min_samples + max_lag):
        raise ValueError(
            f"Not enough overlapping months for lag analysis: {len(common_months)} "
            f"(need >= {min_samples + max_lag})"
        )

    grace_idx = grace_pos.loc[common_months].to_numpy(dtype=int)
    pdd_idx = pdd_pos.loc[common_months].to_numpy(dtype=int)
    common_dates = common_months.to_timestamp(how="start")

    grace_common = np.asarray(
        np.ma.filled(data.grace_rate_data[grace_idx, :, :], np.nan),
        dtype=float,
    )
    grace_da = xr.DataArray(
        grace_common,
        coords={"time": common_dates, "lat": data.grace_rate_lat, "lon": data.grace_rate_lon},
        dims=("time", "lat", "lon"),
        name="grace_rate",
    )

    pdd_common = data.pdd_data.isel(time=pdd_idx).assign_coords(time=common_dates)
    pdd_on_grace = pdd_common.interp(
        lat=data.grace_rate_lat,
        lon=data.grace_rate_lon,
        method="linear",
    )

    grace_flat = grace_common.reshape(len(common_dates), -1)
    pdd_flat = np.asarray(pdd_on_grace.values, dtype=float).reshape(len(common_dates), -1)
    n_pixels = grace_flat.shape[1]

    lags = np.arange(max_lag + 1, dtype=int)
    r_cube = np.full((len(lags), n_pixels), np.nan, dtype=float)
    p_cube = np.full((len(lags), n_pixels), np.nan, dtype=float)
    n_cube = np.zeros((len(lags), n_pixels), dtype=int)

    regional_rows: list[dict[str, float | int]] = []
    grace_reg = area_weighted_mean(grace_da).to_series()
    pdd_reg = area_weighted_mean(pdd_on_grace).to_series()

    for lag_idx, lag in enumerate(lags):
        if lag == 0:
            x_full = pdd_flat
            y_full = grace_flat
            pdd_reg_l = pdd_reg
            grace_reg_l = grace_reg
        else:
            x_full = pdd_flat[:-lag, :]
            y_full = grace_flat[lag:, :]
            pdd_reg_l = pdd_reg.iloc[:-lag]
            grace_reg_l = grace_reg.iloc[lag:]

        for pixel in range(n_pixels):
            x = x_full[:, pixel]
            y = y_full[:, pixel]
            valid = np.isfinite(x) & np.isfinite(y)
            n_valid = int(valid.sum())
            n_cube[lag_idx, pixel] = n_valid
            if n_valid < min_samples:
                continue
            x_valid = x[valid]
            y_valid = y[valid]
            if np.nanstd(x_valid) == 0 or np.nanstd(y_valid) == 0:
                continue
            r_val, p_val = pearsonr(x_valid, y_valid)
            r_cube[lag_idx, pixel] = r_val
            p_cube[lag_idx, pixel] = p_val

        reg_df = pd.concat(
            [pdd_reg_l.rename("pdd"), grace_reg_l.rename("grace_rate")],
            axis=1,
        ).dropna()
        if len(reg_df) >= min_samples and reg_df["pdd"].std() > 0 and reg_df["grace_rate"].std() > 0:
            reg_r, reg_p = pearsonr(reg_df["pdd"].values, reg_df["grace_rate"].values)
            reg_r2 = reg_r ** 2 * 100.0
        else:
            reg_r, reg_p, reg_r2 = np.nan, np.nan, np.nan
        """regional_rows.append(
            {
                "lag_months": int(lag),
                "regional_r": reg_r,
                "regional_p": reg_p,
                "regional_r2_percent": reg_r2,
                "regional_n": int(len(reg_df)),
            }
        )"""

    ny, nx = grace_common.shape[1:]
    r_maps = r_cube.reshape(len(lags), ny, nx)
    p_maps = p_cube.reshape(len(lags), ny, nx)
    n_maps = n_cube.reshape(len(lags), ny, nx)

    valid_any = np.isfinite(r_cube).any(axis=0)
    abs_r = np.abs(r_cube)
    abs_r_fill = np.where(np.isfinite(abs_r), abs_r, -np.inf)
    best_idx = np.argmax(abs_r_fill, axis=0)

    best_lag_flat = np.full(n_pixels, np.nan, dtype=float)
    best_r_flat = np.full(n_pixels, np.nan, dtype=float)
    best_p_flat = np.full(n_pixels, np.nan, dtype=float)
    best_n_flat = np.full(n_pixels, np.nan, dtype=float)
    for pixel in np.where(valid_any)[0]:
        idx = int(best_idx[pixel])
        best_lag_flat[pixel] = float(lags[idx])
        best_r_flat[pixel] = r_cube[idx, pixel]
        best_p_flat[pixel] = p_cube[idx, pixel]
        best_n_flat[pixel] = float(n_cube[idx, pixel])

    best_lag_map = best_lag_flat.reshape(ny, nx)
    best_r_map = best_r_flat.reshape(ny, nx)
    best_p_map = best_p_flat.reshape(ny, nx)
    best_n_map = best_n_flat.reshape(ny, nx)
    best_r2_map = (best_r_map ** 2) * 100.0

    lag_ds = xr.Dataset(
        data_vars={
            "corr_r": (("lag", "lat", "lon"), r_maps),
            "corr_p": (("lag", "lat", "lon"), p_maps),
            "samples_n": (("lag", "lat", "lon"), n_maps),
            "best_lag": (("lat", "lon"), best_lag_map),
            "best_r": (("lat", "lon"), best_r_map),
            "best_p": (("lat", "lon"), best_p_map),
            "best_samples_n": (("lat", "lon"), best_n_map),
            "best_r2_percent": (("lat", "lon"), best_r2_map),
        },
        coords={"lag": lags, "lat": data.grace_rate_lat, "lon": data.grace_rate_lon},
    )
    lag_ds.to_netcdf(output_dir / "grace_rate_vs_pdd_lag_analysis_maps.nc")

    #regional_df = pd.DataFrame(regional_rows)
    #regional_df.to_csv(output_dir / "grace_rate_vs_pdd_lag_analysis_regional.csv", index=False)

    # Plot regional lag curve
    """fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(regional_df["lag_months"], regional_df["regional_r"], marker="o", color="black")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("Lag (months): PDD leads GRACE-rate")
    ax.set_ylabel("Regional Pearson r")
    ax.set_title("Regional lag correlation: GRACE-rate vs monthly PDD")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "grace_rate_vs_pdd_lag_analysis_regional.png", dpi=300, bbox_inches="tight")
    plt.close(fig)"""

    # Plot best-lag and best-r maps
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    if has_cartopy:
        proj = ccrs.NorthPolarStereo(central_longitude=-45)
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(14, 6),
            subplot_kw={"projection": proj},
            constrained_layout=True,
        )
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    panels = [
        {
            "ax": axes[0],
            "field": np.ma.masked_invalid(best_lag_map),
            "cmap": "viridis",
            "vmin": 0,
            "vmax": max_lag,
            "label": "Best lag (months)",
            "title": "Best lag of PDD leading GRACE-rate",
        },
        {
            "ax": axes[1],
            "field": np.ma.masked_invalid(best_r_map),
            "cmap": "RdBu_r",
            "vmin": -1.0,
            "vmax": 1.0,
            "label": "Correlation at best lag (r)",
            "title": "Best-lag correlation strength",
        },
    ]

    sig_best = np.isfinite(best_p_map) & (best_p_map < 0.05)
    sig_overlay = np.where(sig_best, 1.0, np.nan)

    for panel in panels:
        ax = panel["ax"]
        if has_cartopy:
            ax.set_extent([-75, -10, 58, 85], crs=ccrs.PlateCarree())
            im = ax.pcolormesh(
                data.grace_rate_lon,
                data.grace_rate_lat,
                panel["field"],
                transform=ccrs.PlateCarree(),
                cmap=panel["cmap"],
                shading="auto",
                vmin=panel["vmin"],
                vmax=panel["vmax"],
            )
            if np.any(sig_best):
                ax.contour(
                    data.grace_rate_lon,
                    data.grace_rate_lat,
                    sig_overlay,
                    levels=[0.5],
                    colors=["black"],
                    linewidths=0.4,
                    transform=ccrs.PlateCarree(),
                )
            ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
            ax.gridlines(alpha=0.3)
        else:
            im = ax.pcolormesh(
                data.grace_rate_lon,
                data.grace_rate_lat,
                panel["field"],
                cmap=panel["cmap"],
                shading="auto",
                vmin=panel["vmin"],
                vmax=panel["vmax"],
            )
            if np.any(sig_best):
                ax.contour(
                    data.grace_rate_lon,
                    data.grace_rate_lat,
                    sig_overlay,
                    levels=[0.5],
                    colors=["black"],
                    linewidths=0.4,
                )
            ax.set_aspect("equal")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")

        ax.set_title(panel["title"] + "\n(black outline: p < 0.05)")
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
        cbar.set_label(panel["label"])

    fig.savefig(output_dir / "grace_rate_vs_pdd_lag_analysis_maps.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(
        "  Saved lag-analysis outputs: "
        "grace_rate_vs_pdd_lag_analysis_maps.nc, "
        "grace_rate_vs_pdd_lag_analysis_maps.png"
    )
    #return regional_df




# ===================== MAIN =====================

def main() -> None:
    """
    Research question
    -----------------
    How strongly do Greenland ice-mass anomalies co-vary with regional
    warming (as measured by Positive Degree Days)?

    Analysis pipeline
    -----------------
    1. Load GRACE mass anomaly field (Greenland-masked)
    2. Compute monthly PDD anomaly (Greenland-masked)
    3. Align both datasets and flatten for PCA
    4. Ready for joint PCA or correlation analysis
    """

    # ── Parse arguments and check files ───────────────────────────────
    args = parse_args()
    check_file(args.grace_file, "GRACE file")
    check_file(args.temp_file, "temperature file")
    args.output_dir.mkdir(parents=True, exist_ok=True)


    # ── Load and prepare data for PCA ─────────────────────────────────
    pca_data = load_prepare_data(
        grace_path=args.grace_file,
        temp_path=args.temp_file,
        temp_var=args.temp_var,
    )

    # Data:
    # - pca_data.grace_data: (time, lat, lon) GRACE mass anomaly at native resolution
    # - pca_data.temp_data: (time, lat, lon) MONTHLY MEAN temperature data at native resolution
    # - pca_data.pdd_data: (time, lat, lon) PDD anomaly at native resolution
    # - Each dataset keeps its own dates (may have gaps/missing months)
    # - Only start/end times are aligned

    """plot_triplet(
        pca_data,
        "2012-06",
        args.output_dir,
        selected_fields=("grace","grace_rate", "pdd"),
    )"""

    print("\n=== Data Summary ===")
    print(f"  GRACE: {pca_data.grace_data.shape} over {len(pca_data.grace_dates)} months")
    print(f"  PDD: {pca_data.pdd_data.shape} over {len(pca_data.pdd_dates)} months")
    print(f"  Time range: {pca_data.time_range[0]} to {pca_data.time_range[1]}")

    #run_pca_diagnostics(pca_data, args.output_dir, n_modes=5)

    """run_direct_correlation_grace_rate_pdd(
        pca_data,
        args.output_dir,
        min_samples=24,
    )"""

    run_lag_analysis_grace_rate_pdd(
        pca_data,
        args.output_dir,
        max_lag=7,
        min_samples=24,
    )






if __name__ == "__main__":
    main()
