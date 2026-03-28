
"""Side-by-side animated GIF: cumulative GRACE mass change vs. monthly PDD.

Called from ``grace_temp_covariability_try2.main()`` after both the GRACE
field and monthly PDD grid have been loaded.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import gaussian_filter


# ── Greenland bounding box (same as SUBREGIONS["Greenland"]) ──────────────
LON_MIN, LON_MAX = -75, -10
LAT_MIN, LAT_MAX = 58, 85


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def gaussian_filter_nan(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian smoothing that ignores NaN pixels."""
    valid = np.isfinite(data).astype(float)
    data0 = np.nan_to_num(data, nan=0.0)
    smooth_data = gaussian_filter(data0, sigma=sigma)
    smooth_valid = gaussian_filter(valid, sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        result = smooth_data / smooth_valid
    result[smooth_valid < 1e-6] = np.nan
    return result


def _ensure_increasing(lat, lon, cube):
    """Return (lat, lon, cube) with both axes sorted in ascending order."""
    lat = np.asarray(lat)
    lon = np.asarray(lon)
    data = np.array(cube, copy=True)
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[:, ::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        data = data[:, :, ::-1]
    return lat, lon, data


def _interpolate_to_fine_grid(
    cube: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    upsample: int = 4,
    sigma: float = 10.0,
):
    """Cubic-spline upsample + light Gaussian smooth, preserving the land mask."""
    lat, lon, cube = _ensure_increasing(lat, lon, cube)

    lat_fine = np.linspace(lat.min(), lat.max(), len(lat) * upsample)
    lon_fine = np.linspace(lon.min(), lon.max(), len(lon) * upsample)
    Lon2D, Lat2D = np.meshgrid(lon_fine, lat_fine)

    cube_fine = np.empty((cube.shape[0], len(lat_fine), len(lon_fine)))
    for t in range(cube.shape[0]):
        frame = cube[t]
        if np.isnan(frame).all():
            cube_fine[t] = np.nan
            continue

        frame_filled = frame.copy()
        if np.isnan(frame_filled).any():
            frame_filled = np.nan_to_num(frame_filled, nan=np.nanmean(frame_filled))

        spline = RectBivariateSpline(lat, lon, frame_filled, kx=3, ky=3)
        frame_fine = spline(lat_fine, lon_fine)

        # Restore the land-mask boundary
        mask_coarse = np.isfinite(frame).astype(float)
        mask_spline = RectBivariateSpline(lat, lon, mask_coarse, kx=1, ky=1)
        mask_fine = mask_spline(lat_fine, lon_fine)
        frame_fine[mask_fine < 0.5] = np.nan

        frame_fine = gaussian_filter_nan(frame_fine, sigma=sigma)
        cube_fine[t] = frame_fine

    return cube_fine, lat_fine, lon_fine, Lon2D, Lat2D


def _cumulative_trapezoid(anomaly: np.ndarray, time_days: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration along axis-0 (time)."""
    cumulative = np.zeros_like(anomaly)
    for k in range(1, len(time_days)):
        dt = time_days[k] - time_days[k - 1]
        cumulative[k] = (
            cumulative[k - 1]
            + 0.5 * dt * (anomaly[k - 1] + anomaly[k])
        )
    return cumulative


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def animation_comparison(grace_field, pdd_gridded: xr.DataArray, output_dir: Path) -> None:
    """Create a side-by-side animated GIF of cumulative GRACE mass change
    and monthly PDD maps over Greenland.

    Parameters
    ----------
    grace_field : GraceField
        Loaded GRACE mascon data (masked array with .data, .lat, .lon, .dates).
    pdd_gridded : xr.DataArray
        Monthly gridded PDD (time, lat, lon) in °C·days, Greenland-masked.
    output_dir : Path
        Directory where the GIF will be saved.
    """
    print("\n=== Building side-by-side GRACE vs PDD animation ===")

    # ------------------------------------------------------------------
    # 1. Subset GRACE to the Greenland bounding box
    # ------------------------------------------------------------------
    lat_mask = (grace_field.lat >= LAT_MIN) & (grace_field.lat <= LAT_MAX)
    lon_mask = (grace_field.lon >= LON_MIN) & (grace_field.lon <= LON_MAX)
    grace_sub = grace_field.data[:, lat_mask][:, :, lon_mask]
    grace_lat = grace_field.lat[lat_mask]
    grace_lon = grace_field.lon[lon_mask]
    grace_dates = grace_field.dates

    # Fill masked → NaN so the interpolation helpers work uniformly
    grace_cube = np.ma.filled(grace_sub, np.nan).astype(float)

    # ------------------------------------------------------------------
    # 2. Cumulative GRACE mass anomaly (relative to first field)
    # ------------------------------------------------------------------
    reference = grace_cube[0]
    grace_anomaly = grace_cube - reference[None, :, :]
    grace_time_days = np.array(
        [(d - grace_dates[0]).days for d in grace_dates], dtype=float
    )
    grace_cumulative = _cumulative_trapezoid(grace_anomaly, grace_time_days)

    # Upsample + smooth
    grace_fine, g_lat_fine, g_lon_fine, g_Lon2D, g_Lat2D = (
        _interpolate_to_fine_grid(grace_cumulative, grace_lat, grace_lon,
                                  upsample=4, sigma=10.0)
    )

    # ------------------------------------------------------------------
    # 3. Align PDD to GRACE time stamps (nearest-month matching)
    # ------------------------------------------------------------------
    pdd_sorted = pdd_gridded.sortby("time")
    pdd_times = pd.DatetimeIndex(pdd_sorted.time.values)

    overlap_start = max(grace_dates.min(), pdd_times.min())
    overlap_end = min(grace_dates.max(), pdd_times.max())
    grace_overlap = (grace_dates >= overlap_start) & (grace_dates <= overlap_end)

    if not grace_overlap.any():
        print("  WARNING: no overlapping period between GRACE and PDD — skipping animation.")
        return

    common_dates = grace_dates[grace_overlap]
    grace_fine_common = grace_fine[grace_overlap]

    # Reindex PDD to the common GRACE dates (nearest month, ≤20 days tolerance)
    pdd_reindexed = pdd_sorted.reindex(
        time=common_dates, method="nearest", tolerance="20D"
    )
    pdd_vals = pdd_reindexed.values  # (T_common, lat_pdd, lon_pdd)

    # Drop frames that are fully NaN
    valid = ~np.isnan(pdd_vals).all(axis=(1, 2))
    if not valid.any():
        print("  WARNING: PDD data is entirely NaN in the overlap window — skipping.")
        return
    common_dates = common_dates[valid]
    grace_fine_common = grace_fine_common[valid]
    pdd_vals = pdd_vals[valid]

    pdd_lat = pdd_reindexed["lat"].values
    pdd_lon = pdd_reindexed["lon"].values

    # ------------------------------------------------------------------
    # 4. Upsample + smooth the PDD monthly field (no cumulative integral,
    #    just the raw monthly PDD so the seasonal pulse is visible)
    # ------------------------------------------------------------------
    pdd_fine, p_lat_fine, p_lon_fine, p_Lon2D, p_Lat2D = (
        _interpolate_to_fine_grid(pdd_vals, pdd_lat, pdd_lon,
                                  upsample=4, sigma=6.0)
    )

    # ------------------------------------------------------------------
    # 5. Build the side-by-side animation
    # ------------------------------------------------------------------
    n_frames = len(common_dates)
    mass_vmax = np.nanmax(np.abs(grace_fine_common))
    pdd_vmax = max(np.nanmax(np.abs(pdd_fine)), 1e-6)

    proj = ccrs.PlateCarree()
    fig, (ax_mass, ax_pdd) = plt.subplots(
        1, 2, figsize=(14, 6),
        subplot_kw={"projection": proj},
    )

    # --- Left panel: cumulative GRACE mass anomaly ---
    mass_mesh = ax_mass.pcolormesh(
        g_Lon2D, g_Lat2D, grace_fine_common[0],
        cmap="RdBu", vmin=-mass_vmax, vmax=mass_vmax,
        shading="auto", transform=proj,
    )
    # --- Right panel: monthly PDD ---
    pdd_mesh = ax_pdd.pcolormesh(
        p_Lon2D, p_Lat2D, pdd_fine[0],
        cmap="YlOrRd", vmin=0, vmax=pdd_vmax,
        shading="auto", transform=proj,
    )

    for ax in (ax_mass, ax_pdd):
        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=proj)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
        ax.coastlines(linewidth=0.5)
        ax.set_xlim(LON_MIN, LON_MAX)
        ax.set_ylim(LAT_MIN, LAT_MAX)
        mean_lat = 0.5 * (LAT_MIN + LAT_MAX)
        ax.set_aspect(1.0 / np.cos(np.deg2rad(mean_lat)))
        ax.set_xlabel("Longitude (°)")

    ax_mass.set_ylabel("Latitude (°)")
    ax_pdd.set_ylabel("Latitude (°)")

    date_str = common_dates[0].strftime("%Y-%m-%d")
    mass_title = ax_mass.set_title(
        f"Cumulative GRACE mass anomaly\n{date_str}"
    )
    pdd_title = ax_pdd.set_title(
        f"Monthly PDD\n{date_str}"
    )

    cbar_mass = fig.colorbar(mass_mesh, ax=ax_mass, shrink=0.75, pad=0.03)
    cbar_mass.set_label("Cumulative LWE anomaly [cm·day]")
    cbar_pdd = fig.colorbar(pdd_mesh, ax=ax_pdd, shrink=0.75, pad=0.03)
    cbar_pdd.set_label("PDD [°C·days]")

    fig.tight_layout()

    def _update(frame):
        ts = common_dates[frame].strftime("%Y-%m-%d")
        mass_mesh.set_array(grace_fine_common[frame].ravel())
        pdd_mesh.set_array(pdd_fine[frame].ravel())
        mass_title.set_text(f"Cumulative GRACE mass anomaly\n{ts}")
        pdd_title.set_text(f"Monthly PDD\n{ts}")
        return mass_mesh, pdd_mesh, mass_title, pdd_title

    ani = FuncAnimation(fig, _update, frames=n_frames, interval=250, blit=False)

    out_path = output_dir / "greenland_mass_vs_pdd.gif"
    ani.save(str(out_path), writer=PillowWriter(fps=10), dpi=180)
    plt.close(fig)
    print(f"  Saved animation to: {out_path}")