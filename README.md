

# NonLinLoc (NLLoc) Regional Earthquake Location Pipeline

This repository provides an end-to-end, production-ready workflow to perform regional earthquake location with **NonLinLoc (NLLoc)**, starting from scattered velocity models and REAL pick outputs, and ending with a clean, filterable event catalog.



## 1. Pipeline Overview

```
Scattered velocity model
        │
        ▼
[Step 1] Interpolate to regular (lon, lat, depth) Vp/Vs grid
        │
        ▼
[Step 2] Convert to NLLoc SLOW_LEN grids (Cartesian, P/S)
        │
        ▼
[Step 3] Grid2Time → station travel-time tables
        │
        ▼
[Step 4] REAL → NLLoc inputs (STATION / OBS / IN)
        │
        ▼
[Step 5] Parallel NLLoc location (event-by-event)
        │
        ▼
[Step 6] Parse *.hyp → export catalog (CSV/TSV/TXT)
```



## 2. Software Requirements

### Required Software

* **NonLinLoc** (including `NLLoc` and `Grid2Time` binaries)
* **Python ≥ 3.9**

### Python Dependencies

```bash
pip install numpy scipy
```

> Note: `step2.make_nll_grid.py` uses `nllgrid` (a Python helper for NLLoc grids).
> Ensure it is available in your environment or installed according to your local setup.



## 3. Recommended Directory Layout

```
nlloc_script/
  demo_data/
    velo.txt                         # Step 1 input: scattered velocity samples
    velocity_grid_*.npz              # Step 1 output
    slow_P_cubic.*.mod.hdr/.buf      # Step 2 output (P)
    slow_S_cubic.*.mod.hdr/.buf      # Step 2 output (S)
    location.txt                     # Station list
    time/
      tt_PS.*.mod.hdr/.buf           # Step 3 travel-time grids
  out_dir/
    sc.stations                      # Step 4 station file
    real.obs                         # Step 4 observation file
    nlloc.in                         # Step 4 control file
    nlloc.temp.in                    # Step 4 template variant
    work/<event_id>/                 # Step 5 per-event work dirs
    out/*.hyp                        # Step 5 NLLoc outputs
```



## 4. Step-by-Step Usage (Minimal Working Example)

### Step 1 — Interpolate Scattered Velocity Model

```bash
python step1.interp_vel_data.py
```

**Input**
`demo_data/velo.txt`, comma-separated:

```
lon, lat, depth_km, vp_km_s, vs_km_s
```

**Output**

* `velocity_grid_*.npz`
  (`lon_grid`, `lat_grid`, `depth_grid`, `Vp_grid`, `Vs_grid`)

**Validation Checklist**

* No NaNs in `Vp_grid` / `Vs_grid`
* Grid bounds fully cover the target region



### Step 2 — Build NLLoc SLOW_LEN Grids

```bash
python step2.make_nll_grid.py
```

**Purpose**

* Convert geographic Vp/Vs grids to **local Cartesian cubic grids**
* Export NLLoc-compatible **SLOW_LEN** (`(1/v)*dx`) models

**Output**

* `slow_P_cubic.P.mod.hdr/.buf`
* `slow_S_cubic.S.mod.hdr/.buf`

**Critical Parameters**

* `lon0`, `lat0` — local tangent-plane reference
* `grid_step_km` — cubic grid spacing (`dx=dy=dz`)

> **Must remain consistent** with Step 3 and Step 4 `TRANS SIMPLE`.



### Step 3 — Grid2Time: Generate Travel-Time Tables

```bash
python step3.grid2time.py
```

**Input**

* `location.txt`:

  ```
  net sta lon lat elev_km
  ```
* Slow grids from Step 2

**Output**

* Travel-time grids under `demo_data/time/`
* Default root: `tt_PS`

#### ⚠ Important: P/S Overwrite Risk

The script writes **P and S travel times to the same root** (`tt_PS`).
Depending on your Grid2Time build, this **may overwrite** one phase.

**Recommended Safe Options**

1. Use **separate roots**: `tt_P` and `tt_S`
2. Verify filenames include explicit phase tags (`.P.` / `.S.`)



### Step 4 — Generate NLLoc Input Files

```bash
python step4.make_nll_in_file.py
```

**Input**

* `real.txt` (REAL format)
* `location.txt`
* Travel-time grids (`time/tt_PS`)

**Output**

* `sc.stations` — NLLoc `STATION` file
* `real.obs` — NLLoc `NLLOC_OBS`
* `nlloc.in`, `nlloc.temp.in`

**Key Assumptions**

* REAL header line:

  ```
  event_id YYYY MM DD HH:MM:SS[.sss]
  ```
* Pick line:

  ```
  net sta phase ... rel_t_sec
  ```

  (`rel_t_sec` at column index 4)

**Critical Consistency Rules**

1. **Station naming**

   * Grid2Time: `GTSRCE label = NET_STA`
   * OBS: station field = `NET_STA`
2. **Coordinate reference**

   * `TRANS SIMPLE lat0 lon0` must match Steps 2 & 3
3. **Travel-time root**

   * Must exactly match Step 3 output



### Step 5 — Parallel Event Location

```bash
python step5.parallel_locate.py
```

**Input**

* `out_dir/real.obs`
* `out_dir/nlloc.in`
* Travel-time grids (`tt_PS`)

**Output**

* `out_dir/out/*.hyp`
* `out_dir/work/<event_id>/`
* `parallel_loc_summary.json`

#### ⚠ Recommended Patch (Prevent Output Overwrite)

Currently, all events share the same `outputFileRoot`.

**Recommended change**
Use an **event-specific output prefix**, e.g.:

```
LOCFILES <obs> NLLOC_OBS <tt_root> <out_dir>/out/<event_id>_
```

This guarantees isolation of `.hyp`, `.sum`, `.scat` files per event.



### Step 6 — Export Location Catalog

```bash
python step.export_pyp_catalog.py \
  --in-dir nlloc_script/out_dir \
  --recursive \
  --out all.locfiles.csv \
  --format csv \
  --header \
  --prefer auto \
  --min-nphs 6 \
  --max-rms 0.5
```

**Supported Outputs**

* `csv`, `tsv`, `txt`

**Extracted Fields**

```
event_id, origin_time, lat, lon, depth_km,
horizontal_error_km, vertical_error_km,
RMS, Nphs, Gap
```

**Filtering Options**

* Minimum phases (`--min-nphs`)
* Maximum RMS (`--max-rms`)
* Maximum azimuthal gap (`--max-gap`)



## 5. Design Conventions

* **SLOW_LEN**: `(1 / velocity_km_s) * grid_step_km`
* **Depth (Z)**: positive downward
* **Projection**: local tangent-plane approximation
* **LOCGRID**: automatically derived from travel-time `.hdr` (recommended)



## 6. Common Errors & Troubleshooting

### “Station not found” / “No travel time”

* Check `NET_STA` naming consistency
* Verify `LOCFILES` travel-time root

### Systematic location offset

* `TRANS SIMPLE lat0 lon0` mismatch across steps
* Mixing grids generated with different reference points

### Only P or S travel times exist

* Grid2Time overwrite issue → split roots

### Parallel outputs overwritten

* Use event-specific `outputFileRoot` (see Step 5)



## 7. Reproducibility Checklist

* Fixed `lon0 / lat0`
* Fixed `grid_step_km`
* Fixed travel-time root naming
* Fixed EventID generation logic
* Preserve `parallel_loc_summary.json`



## 8. Notes for Extension

* Easily extended to **P-only**, **S-only**, or **multi-model** experiments
* REAL parser can be adapted to other pick formats (modify Step 4 only)
* Compatible with Bayesian post-processing using `.hyp` statistics
