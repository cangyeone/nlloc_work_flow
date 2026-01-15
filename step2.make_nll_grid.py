"""
step2.make_nll_grid.py

Purpose
-------
Convert the regular geographic velocity grids (lon/lat/depth, Vp/Vs) generated in
Step 1 into NonLinLoc (NLLoc) compatible 3D grid files (*.hdr + *.buf) with type
"SLOW_LEN" for both P and S waves.

Why NLLoc needs this
--------------------
NLLoc’s grid-based travel-time engines work in a local Cartesian coordinate system
(X, Y, Z) with constant spacing (dx, dy, dz). A common input grid type is:

    SLOW_LEN = (1 / velocity) * grid_step_length

Units:
- velocity: km/s
- grid_step_length: km
- SLOW_LEN: seconds (s)

Interpreting coordinates
------------------------
- X, Y are local kilometers (km) derived from lon/lat via a simple local tangent-plane
  approximation around (lon0, lat0).
- Z is depth in km (positive downward).
- The grid origin (x_orig, y_orig, z_orig) corresponds to the first grid node.

Inputs
------
1) NPZ file from Step 1:
   - lon_grid   : (Nlon,)
   - lat_grid   : (Nlat,)
   - depth_grid : (Ndepth,)
   - Vp_grid    : (Nlon, Nlat, Ndepth)  [km/s]
   - Vs_grid    : (Nlon, Nlat, Ndepth)  [km/s]

Outputs
-------
Two NLLoc grid datasets written as:
- <out_basename_P>.hdr and <out_basename_P>.buf   (P-wave SLOW_LEN)
- <out_basename_S>.hdr and <out_basename_S>.buf   (S-wave SLOW_LEN)

Example output basenames:
- nlloc_script/demo_data/slow_P_cubic
- nlloc_script/demo_data/slow_S_cubic

Implementation notes
--------------------
1) Interpolation: we re-sample the Step-1 grid onto a *cubic* Cartesian grid
   (dx = dy = dz = grid_step_km) using scipy.griddata:
   - linear interpolation inside the convex hull
   - nearest-neighbor fill for NaNs (outside hull)

2) Projection: lon/lat -> km uses a small-region approximation:
   x = (lon - lon0) * R * cos(lat0)
   y = (lat - lat0) * R
   where angles are in radians.

3) Safety: velocities are clamped by eps to avoid division by zero.

Caveats
-------
- The lon/lat -> km conversion is a local approximation (not a true map projection).
  For large regions, consider an actual projection (e.g., UTM via pyproj).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.interpolate import griddata
from nllgrid import NLLGrid


# =============================================================================
# Configuration container (cleaner I/O and reproducible parameters)
# =============================================================================

@dataclass
class NLLGridConfig:
    # Input from Step 1
    input_npz: str = "nlloc_script/demo_data/velocity_grid_lon97_108_lat21_34_dz2p5.npz"

    # Local reference point for lon/lat -> km conversion (tangent-plane approx)
    lon0: float = 102.5
    lat0: float = 27.5
    earth_radius_km: float = 6371.0

    # Target cubic grid definition (km)
    grid_step_km: float = 5.0
    z_min_km: float = 0.0
    z_max_km: float = 70.0

    # Output basenames (without extension)
    out_basename_p: str = "nlloc_script/demo_data/slow_P_cubic"
    out_basename_s: str = "nlloc_script/demo_data/slow_S_cubic"

    # Numerical safety
    eps_vel: float = 1e-6

    # Interpolation methods
    interp_linear: str = "linear"
    interp_fill: str = "nearest"


# =============================================================================
# Utilities
# =============================================================================

def lonlat_to_xy_km(
    lon: np.ndarray,
    lat: np.ndarray,
    lon0: float,
    lat0: float,
    R_earth_km: float = 6371.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert lon/lat in degrees to local X/Y in kilometers using a tangent-plane
    approximation around (lon0, lat0).

    Parameters
    ----------
    lon, lat : arrays in degrees
    lon0, lat0 : reference point in degrees
    R_earth_km : Earth radius in km

    Returns
    -------
    x_km, y_km : arrays in km

    Notes
    -----
    This assumes small geographic extent. For large extents, use a real projection.
    """
    deg2rad = np.pi / 180.0
    lon_rad = lon * deg2rad
    lat_rad = lat * deg2rad
    lon0_rad = lon0 * deg2rad
    lat0_rad = lat0 * deg2rad

    # X: scaled by cos(lat0) to account for meridian convergence
    x_km = (lon_rad - lon0_rad) * R_earth_km * np.cos(lat0_rad)
    y_km = (lat_rad - lat0_rad) * R_earth_km
    return x_km, y_km


def interpolate_to_cubic_grid(
    points_xyz: np.ndarray,
    values: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
    z_new: np.ndarray,
    method_linear: str = "linear",
    method_fill: str = "nearest",
) -> np.ndarray:
    """
    Interpolate scattered 3D values onto a regular cubic grid and fill NaNs.

    Parameters
    ----------
    points_xyz : (N, 3) array of source points [x_km, y_km, z_km]
    values     : (N,) array of values at source points (e.g., Vp or Vs)
    x_new, y_new, z_new : 1D arrays defining the target grid nodes
    method_linear : interpolation method for interior (usually 'linear')
    method_fill   : fill method for NaNs (usually 'nearest')

    Returns
    -------
    grid_values : (Nx, Ny, Nz) float32 array interpolated on the cubic grid
    """
    Nx, Ny, Nz = len(x_new), len(y_new), len(z_new)

    # Build (Nx,Ny,Nz) coordinate mesh, then flatten to (M,3) target points
    X3, Y3, Z3 = np.meshgrid(x_new, y_new, z_new, indexing="ij")
    target_xyz = np.vstack([X3.ravel(), Y3.ravel(), Z3.ravel()]).T  # (M,3)

    # Linear interpolation (NaNs outside convex hull)
    out = griddata(points_xyz, values, target_xyz, method=method_linear)

    # Fill NaNs with nearest-neighbor extrapolation
    mask_nan = np.isnan(out)
    if np.any(mask_nan):
        out[mask_nan] = griddata(points_xyz, values, target_xyz[mask_nan], method=method_fill)

    return out.reshape(Nx, Ny, Nz).astype(np.float32)


def velocity_to_slow_len(
    vel_grid_km_s: np.ndarray,
    grid_step_km: float,
    eps_vel: float = 1e-6,
) -> np.ndarray:
    """
    Convert velocity (km/s) to NLLoc SLOW_LEN (s) per cell step:
        SLOW_LEN = (1 / vel) * grid_step_km

    Parameters
    ----------
    vel_grid_km_s : (Nx,Ny,Nz) array in km/s
    grid_step_km  : scalar in km
    eps_vel       : clamp to avoid division by zero

    Returns
    -------
    slow_len_s : (Nx,Ny,Nz) float32 array in seconds
    """
    vel_safe = np.maximum(vel_grid_km_s, eps_vel)
    slow_len = (1.0 / vel_safe) * grid_step_km
    return slow_len.astype(np.float32)


def write_nll_grid_slow_len(
    slow_len: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
    z_new: np.ndarray,
    grid_step_km: float,
    basename: str,
    wave_type: str = "P",
) -> None:
    """
    Write a 3D grid to NLLoc format via NLLGrid.

    Parameters
    ----------
    slow_len : (Nx,Ny,Nz) array (float32) representing SLOW_LEN in seconds
    x_new, y_new, z_new : 1D arrays of grid node coordinates (km)
    grid_step_km : grid spacing (km), used as dx=dy=dz
    basename : output path prefix without extension (writes .hdr and .buf)

    Notes
    -----
    - NLLGrid expects:
        nx, ny, nz
        x_orig, y_orig, z_orig  (origin coordinate of grid index (0,0,0))
        dx, dy, dz              (grid spacing)
      plus the array and metadata fields type/float_type/basename.
    """
    Nx, Ny, Nz = slow_len.shape

    x_orig = float(x_new[0])
    y_orig = float(y_new[0])
    z_orig = float(z_new[0])

    grd = NLLGrid(
        nx=Nx, ny=Ny, nz=Nz,
        x_orig=x_orig, y_orig=y_orig, z_orig=z_orig,
        dx=grid_step_km, dy=grid_step_km, dz=grid_step_km,
    )
    grd.array = slow_len
    grd.type = "SLOW_LEN"
    grd.float_type = "FLOAT"
    grd.basename = basename + f".{wave_type.upper()}.mod"

    # Write NLLoc headers and binary buffers
    grd.write_hdr_file()
    grd.write_buf_file()


# =============================================================================
# Main pipeline
# =============================================================================

def main(cfg: NLLGridConfig) -> None:
    # -------------------------------------------------------------------------
    # 1) Load geographic (lon/lat/depth) velocity grids produced by Step 1
    # -------------------------------------------------------------------------
    npz_path = Path(cfg.input_npz)
    data = np.load(npz_path)

    lon_grid = data["lon_grid"]      # (Nlon,)
    lat_grid = data["lat_grid"]      # (Nlat,)
    depth_grid = data["depth_grid"]  # (Ndepth,) [km]
    Vp_grid = data["Vp_grid"]        # (Nlon, Nlat, Ndepth) [km/s]
    Vs_grid = data["Vs_grid"]        # (Nlon, Nlat, Ndepth) [km/s]

    # -------------------------------------------------------------------------
    # 2) Flatten the Step-1 regular grid into scattered samples for interpolation
    #    We treat each grid node as a "point" with known Vp/Vs.
    # -------------------------------------------------------------------------
    Lon3d, Lat3d, Dep3d = np.meshgrid(lon_grid, lat_grid, depth_grid, indexing="ij")

    lon_flat = Lon3d.ravel()
    lat_flat = Lat3d.ravel()
    dep_flat = Dep3d.ravel()  # depth in km
    Vp_flat = Vp_grid.ravel()
    Vs_flat = Vs_grid.ravel()

    # -------------------------------------------------------------------------
    # 3) Convert lon/lat (degrees) to local Cartesian X/Y (km), keep Z=depth (km)
    # -------------------------------------------------------------------------
    x_flat, y_flat = lonlat_to_xy_km(
        lon_flat, lat_flat,
        lon0=cfg.lon0, lat0=cfg.lat0,
        R_earth_km=cfg.earth_radius_km,
    )
    z_flat = dep_flat.astype(np.float64)  # km

    points_xyz = np.vstack([x_flat, y_flat, z_flat]).T  # (N,3)

    # -------------------------------------------------------------------------
    # 4) Define the target cubic grid (dx=dy=dz=grid_step_km)
    #    X/Y ranges are derived from input coverage; Z range is explicitly set.
    # -------------------------------------------------------------------------
    grid_step = float(cfg.grid_step_km)

    x_min, x_max = float(x_flat.min()), float(x_flat.max())
    y_min, y_max = float(y_flat.min()), float(y_flat.max())
    z_min, z_max = float(cfg.z_min_km), float(cfg.z_max_km)

    x_new = np.arange(x_min, x_max + 1e-6, grid_step)
    y_new = np.arange(y_min, y_max + 1e-6, grid_step)
    z_new = np.arange(z_min, z_max + 1e-6, grid_step)

    Nx, Ny, Nz = len(x_new), len(y_new), len(z_new)
    print(f"Cubic grid: Nx={Nx}, Ny={Ny}, Nz={Nz}, step={grid_step:.3f} km")
    print(f"Origin (x_orig,y_orig,z_orig): {x_new[0]:.3f}, {y_new[0]:.3f}, {z_new[0]:.3f}")

    # -------------------------------------------------------------------------
    # 5) Interpolate Vp/Vs to the cubic grid
    # -------------------------------------------------------------------------
    Vp_new = interpolate_to_cubic_grid(
        points_xyz, Vp_flat, x_new, y_new, z_new,
        method_linear=cfg.interp_linear,
        method_fill=cfg.interp_fill,
    )
    Vs_new = interpolate_to_cubic_grid(
        points_xyz, Vs_flat, x_new, y_new, z_new,
        method_linear=cfg.interp_linear,
        method_fill=cfg.interp_fill,
    )

    # -------------------------------------------------------------------------
    # 6) Convert Vp/Vs (km/s) to NLLoc SLOW_LEN (s)
    # -------------------------------------------------------------------------
    slow_P = velocity_to_slow_len(Vp_new, grid_step_km=grid_step, eps_vel=cfg.eps_vel)
    slow_S = velocity_to_slow_len(Vs_new, grid_step_km=grid_step, eps_vel=cfg.eps_vel)

    # -------------------------------------------------------------------------
    # 7) Write NLLoc grids (.hdr + .buf) for P and S
    # -------------------------------------------------------------------------
    # Ensure output directories exist
    Path(cfg.out_basename_p).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.out_basename_s).parent.mkdir(parents=True, exist_ok=True)

    write_nll_grid_slow_len(slow_P, x_new, y_new, z_new, grid_step, cfg.out_basename_p, wave_type="P")
    print(f"Wrote {cfg.out_basename_p}.hdr/.buf (type=SLOW_LEN, float=FLOAT)")

    write_nll_grid_slow_len(slow_S, x_new, y_new, z_new, grid_step, cfg.out_basename_s, wave_type="S")
    print(f"Wrote {cfg.out_basename_s}.hdr/.buf (type=SLOW_LEN, float=FLOAT)")


if __name__ == "__main__":
    # In a manual, you can expose these knobs as CLI args later (argparse),
    # but keeping a config object is already a big improvement for readability.
    cfg = NLLGridConfig(
        input_npz="nlloc_script/demo_data/velocity_grid_lon97_108_lat21_34_dz2p5.npz",
        lon0=102.5,
        lat0=27.5,
        grid_step_km=5.0,
        z_min_km=0.0,
        z_max_km=70.0,
        out_basename_p="nlloc_script/demo_data/slow_P_cubic",
        out_basename_s="nlloc_script/demo_data/slow_S_cubic",
    )
    main(cfg)
