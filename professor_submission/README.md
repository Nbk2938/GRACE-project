# GRACE-Temperature Covariability (Greenland)

This project analyzes the relationship between Greenland ice-mass change (GRACE/GRACE-FO) and regional temperature variability, with PCA, lag correlation, and detrended regional validation.

## Repository Contents

- `grace_temp_covariability_try1.py`: main analysis pipeline
- `outputs/`: generated figures and CSV summaries
- `main.tex`, `script.tex`: report manuscript

## Data Requirements

Create a `Data/` folder in the project root and place these files:

1. GRACE mascon NetCDF:
	- `Data/GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc`
2. Temperature data (either format is supported):
	- preferred: `Data/temp_data.nc`
	- optional: `Data/temp_data.grib`

Notes:
- The script default temperature variable is `t2m`.
- If temperature is in Kelvin, it is converted to degC automatically.

## Environment

Use Python 3.10+ with:

- `numpy`
- `pandas`
- `xarray`
- `netCDF4`
- `scipy`
- `scikit-learn`
- `matplotlib`
- `cfgrib` (only needed when using GRIB input)

## Run

From project root:

```bash
python grace_temp_covariability_try1.py
```

Optional arguments:

```bash
python grace_temp_covariability_try1.py \
  --grace-file Data/GRCTellus.JPL.200204_202512.GLO.RL06.3M.MSCNv04CRI.nc \
  --temp-file Data/temp_data.nc \
  --temp-var t2m \
  --output-dir outputs \
  --clim-ref-start 2004-01 \
  --clim-ref-end 2009-12
```

Add `--run-preanalysis-plots` to generate extra raw/seasonal diagnostic plots.

## Main Outputs

Key CSV files in `outputs/`:

- `validation_trend_detrended_correlation.csv`
- `regional_correlation_summary.csv`
- `pc_cross_correlation_lag0.csv`
- `grace_pca_variance.csv`
- `monthly_mean_temperature_pca_variance.csv`

Key figures in `outputs/`:

- `grace_pca_eof_pc.png`
- `monthly_mean_temperature_pca_eof_pc.png`
- `lag_cross_correlation.png`
- `regional_scatter.png`
- `scree_grace_mass.png`
- `scree_temperature.png`

## Report Build

Compile the report with:

```bash
pdflatex main.tex
```

`main.tex` includes `script.tex` and the figures from `outputs/`.
