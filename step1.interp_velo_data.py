import numpy as np
from scipy.interpolate import griddata


def build_velocity_grid(
    filename,
    lon_min=97.0, lon_max=108.0, dlon=0.2,
    lat_min=21.0, lat_max=34.0, dlat=0.2,
    depth_min=0.0, depth_max=70.0, ddepth=2.5,
):
    """
    Build a regular 3D velocity grid (Vp/Vs) from an irregular (scattered) velocity model.

    Context (for NLLoc manuals)
    --------------------------
    NonLinLoc (NLLoc) typically consumes gridded 3D models (e.g., for travel-time
    computation and subsequent hypocenter location). In practice, many regional models
    are provided as scattered samples (lon, lat, depth, Vp, Vs). This function converts
    such scattered samples into a *regular* 3D grid by interpolation.

    Input file format
    -----------------
    `filename` should be a text file where each row contains 5 columns:
        lon, lat, depth_km, vp_km_s, vs_km_s

    - lon / lat: geographic longitude/latitude in degrees
    - depth_km: depth in kilometers (positive downward is assumed)
    - vp_km_s: P-wave velocity in km/s
    - vs_km_s: S-wave velocity in km/s

    Target grid definition
    ----------------------
    The output grid is defined as a Cartesian product of three 1D coordinate arrays:

        lon_grid   = [lon_min, lon_min+dlon, ..., lon_max]
        lat_grid   = [lat_min, lat_min+dlat, ..., lat_max]
        depth_grid = [depth_min, depth_min+ddepth, ..., depth_max]

    where the inclusive upper bound is enforced by adding a small epsilon (1e-6)
    in `np.arange(...)`.

    Array ordering / indexing convention
    ------------------------------------
    We build the 3D mesh using:
        np.meshgrid(lon_grid, lat_grid, depth_grid, indexing='ij')

    With `indexing='ij'`, the resulting velocity arrays follow:
        Vp_grid.shape == (Nx, Ny, Nz)
        Vs_grid.shape == (Nx, Ny, Nz)

    and indices map as:
        x-index -> longitude (lon_grid)
        y-index -> latitude  (lat_grid)
        z-index -> depth     (depth_grid)

    This is important for downstream conversion to NLLoc grid formats:
    you must keep the axis ordering consistent when writing buffers/headers.

    Interpolation strategy (critical)
    ---------------------------------
    We use `scipy.interpolate.griddata` in 3D (lon, lat, depth) space.

    Step A) Linear interpolation:
        method='linear'
    - Produces smooth, piecewise-linear interpolation.
    - Only defined inside the convex hull of the input points.
    - Points outside the convex hull become NaN (no extrapolation).

    Step B) Nearest-neighbor fill for NaNs:
        method='nearest' on NaN locations only
    - Fills the undefined regions (typically outside the convex hull) by
      assigning each target point the value of the closest input sample.
    - Acts as a pragmatic "extrapolation" to guarantee the final grid has no NaNs.
    - This is often necessary because many grid formats / travel-time engines
      do not accept missing values.

    Returns
    -------
    lon_grid : (Nx,) 1D array of longitude grid nodes
    lat_grid : (Ny,) 1D array of latitude grid nodes
    depth_grid : (Nz,) 1D array of depth grid nodes
    Vp_grid : (Nx, Ny, Nz) 3D array, P-wave velocity on the grid (km/s)
    Vs_grid : (Nx, Ny, Nz) 3D array, S-wave velocity on the grid (km/s)

    Notes / practical considerations
    --------------------------------
    1) Coordinates:
       This interpolation treats (lon, lat, depth) as Euclidean coordinates.
       For large regions, geographic curvature can introduce distortions.
       For typical regional scales and moderate grid spacing, this is often acceptable,
       but if you need higher fidelity, consider projecting lon/lat to a local Cartesian
       coordinate system (e.g., UTM) before interpolation.

    2) Performance:
       The number of target grid points is Nx * Ny * Nz.
       Memory/time can grow quickly for fine grids.

    3) Validation:
       Always inspect slices of Vp/Vs and check for artifacts (especially near boundaries),
       since nearest-neighbor filling can create sharp discontinuities outside the convex hull.
    """

    # -------------------------------------------------------------------------
    # 1) Read scattered velocity samples.
    #    Assumption: each row is "lon,lat,depth_km,vp_km_s,vs_km_s" (comma-separated).
    # -------------------------------------------------------------------------
    data = np.loadtxt(filename, delimiter=",")

    # Column extraction for clarity and to make shape assumptions explicit.
    lon = data[:, 0]    # (N,) longitude samples [deg]
    lat = data[:, 1]    # (N,) latitude samples  [deg]
    depth = data[:, 2]  # (N,) depth samples     [km]
    vp = data[:, 3]     # (N,) Vp samples        [km/s]
    vs = data[:, 4]     # (N,) Vs samples        [km/s]

    # -------------------------------------------------------------------------
    # 2) Construct regular 1D grid coordinates.
    #    Add a small epsilon to include the upper bound when it falls on the step.
    # -------------------------------------------------------------------------
    lon_grid = np.arange(lon_min, lon_max + 1e-6, dlon)        # (Nx,)
    lat_grid = np.arange(lat_min, lat_max + 1e-6, dlat)        # (Ny,)
    depth_grid = np.arange(depth_min, depth_max + 1e-6, ddepth)  # (Nz,)

    Nx = lon_grid.size
    Ny = lat_grid.size
    Nz = depth_grid.size
    print(f"Grid size: Nx={Nx}, Ny={Ny}, Nz={Nz}, total={Nx * Ny * Nz}")

    # -------------------------------------------------------------------------
    # 3) Build the 3D mesh for the target grid coordinates.
    #    indexing='ij' ensures the axis order is (lon, lat, depth) -> (x, y, z).
    # -------------------------------------------------------------------------
    Lon3d, Lat3d, Dep3d = np.meshgrid(
        lon_grid, lat_grid, depth_grid, indexing="ij"
    )
    # Lon3d/Lat3d/Dep3d each have shape (Nx, Ny, Nz)

    # -------------------------------------------------------------------------
    # 4) Interpolate Vp and Vs from scattered points to the target grid points.
    #
    #    griddata expects:
    #      - points: (N, D) array of input coordinates (here D=3)
    #      - values: (N,) array of data values corresponding to each point
    #      - xi:     (M, D) array of target coordinates (M = Nx*Ny*Nz)
    # -------------------------------------------------------------------------
    points = np.vstack([lon, lat, depth]).T  # (N, 3) input coordinates

    # Flatten the 3D mesh to a list of target points (M, 3).
    target_points = np.vstack([
        Lon3d.ravel(),
        Lat3d.ravel(),
        Dep3d.ravel()
    ]).T  # (M, 3) where M = Nx*Ny*Nz

    # (A) Linear interpolation (smooth, but undefined outside convex hull -> NaN)
    vp_lin = griddata(points, vp, target_points, method="linear")  # (M,)
    vs_lin = griddata(points, vs, target_points, method="linear")  # (M,)

    # (B) Identify undefined locations (NaNs) produced by linear interpolation.
    mask_vp_nan = np.isnan(vp_lin)
    mask_vs_nan = np.isnan(vs_lin)

    # (C) Fill NaNs using nearest-neighbor "extrapolation" (only where needed).
    if np.any(mask_vp_nan):
        vp_nn = griddata(points, vp, target_points[mask_vp_nan], method="nearest")
        vp_lin[mask_vp_nan] = vp_nn

    if np.any(mask_vs_nan):
        vs_nn = griddata(points, vs, target_points[mask_vs_nan], method="nearest")
        vs_lin[mask_vs_nan] = vs_nn

    # -------------------------------------------------------------------------
    # 5) Reshape the flattened arrays back into 3D grids with (Nx, Ny, Nz).
    # -------------------------------------------------------------------------
    Vp_grid = vp_lin.reshape(Nx, Ny, Nz)
    Vs_grid = vs_lin.reshape(Nx, Ny, Nz)

    return lon_grid, lat_grid, depth_grid, Vp_grid, Vs_grid


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Example usage:
    #   1) Read scattered velocity samples from a demo file
    #   2) Interpolate to a regular 3D grid
    #   3) Save as .npz for later conversion to NLLoc grid/buffer formats
    # -------------------------------------------------------------------------
    (
        lon_grid,
        lat_grid,
        depth_grid,
        Vp_grid,
        Vs_grid,
    ) = build_velocity_grid("nlloc_script/demo_data/velo.txt")

    # Save intermediate result as NumPy .npz.
    # This is convenient for debugging/visualization before exporting to NLLoc formats.
    np.savez(
        "nlloc_script/demo_data/velocity_grid_lon97_108_lat21_34_dz2p5.npz",
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        depth_grid=depth_grid,
        Vp_grid=Vp_grid,
        Vs_grid=Vs_grid,
    )
    print("Saved to velocity_grid_lon97_108_lat21_34_dz2p5.npz")
