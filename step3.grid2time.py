#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Grid2TimeConfig:
    # 固定：所有文件都放这里；Grid2Time 也在这里运行
    work_dir: str = "nlloc_script/demo_data"

    # ---------- Inputs（相对 work_dir） ----------
    station_file: str = "location.txt"

    # 输入慢度网格 root（不带 .hdr/.buf）
    # 建议用你已经修正过的命名：*.P.mod / *.S.mod
    slow_p_root: str = "slow_P_cubic"
    slow_s_root: str = "slow_S_cubic"

    # ---------- Outputs（相对 work_dir） ----------
    out_dir: str = "time"
    out_ps_root: str = "time/tt_PS"

    # 单一输入文件：包含两条 GTFILES（P 和 S）
    grid2time_ps_in: str = "grid2time_PS.in"

    # ---------- NLLoc Grid2Time Common Controls ----------
    control_1: int = 1
    control_seed: int = 54321
    lat0: float = 27.5
    lon0: float = 102.5
    z0: float = 0.0

    gtmode: str = "GRID3D"
    angles: str = "ANGLES_NO"
    gt_plfd_1: float = 1.0e-3
    gt_plfd_2: int = 0

    # ---------- Station handling ----------
    z_srce_km: float = 0.0
    keep_elev_sign: bool = True

    # ---------- Execution ----------
    grid2time_bin: str = "/Users/yuziye/machinelearning/location/NonLinLoc/src/bin/Grid2Time"
    iswap: int = 0
    run_grid2time: bool = True

    # ---------- Safety checks ----------
    # 如果输出文件看起来被覆盖（只剩一套），给出告警
    verify_outputs: bool = True


def _parse_station_lines(station_file: str) -> List[Tuple[str, str, float, float, float]]:
    stations = []
    p = Path(station_file)
    if not p.is_file():
        raise FileNotFoundError(f"Station file not found: {station_file}")

    for ln, raw in enumerate(p.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 5:
            raise ValueError(f"Bad station line (need 5 columns): line {ln}: {raw}")
        net, sta = parts[0], parts[1]
        lon = float(parts[2])
        lat = float(parts[3])
        elev_km = float(parts[4])
        stations.append((net, sta, lon, lat, elev_km))

    if not stations:
        raise ValueError(f"No stations parsed from: {station_file}")
    return stations


def _format_grid2time_ps_in(
    cfg: Grid2TimeConfig,
    slow_p_root: str,
    slow_s_root: str,
    out_ps_root: str,
    stations: List[Tuple[str, str, float, float, float]],
) -> str:
    """
    Single Grid2Time .in containing TWO GTFILES lines (P and S) writing to the SAME out root.
    """
    lines = []
    lines.append("# ========= 通用控制参数 =========")
    lines.append(f"CONTROL {cfg.control_1} {cfg.control_seed}")
    lines.append(f"TRANS SIMPLE {cfg.lat0:.6f} {cfg.lon0:.6f} {cfg.z0:.1f}")
    lines.append("")
    lines.append("# P, S 都从各自 slow 模型生成，只是 waveType 不同")
    lines.append("# GTFILES inputFileRoot outputFileRoot waveType iSwapBytesOnInput")
    lines.append(f"GTFILES {slow_p_root} {out_ps_root} P {cfg.iswap}")
    lines.append(f"GTFILES {slow_s_root} {out_ps_root} S {cfg.iswap}")
    lines.append("")
    lines.append(f"GTMODE {cfg.gtmode} {cfg.angles}")
    lines.append(f"GT_PLFD {cfg.gt_plfd_1:.1e} {cfg.gt_plfd_2}")
    lines.append("")
    lines.append("# ========= 台站列表 =========")
    lines.append("# GTSRCE label LATLON  lat  lon  z(km, 正向向下)  elev(km)")
    lines.append("")

    for net, sta, lon, lat, elev_km in stations:
        label = f"{net}_{sta}"  # 不能有空格
        elev_out = elev_km if cfg.keep_elev_sign else (-elev_km)
        lines.append(
            f"GTSRCE {label:<10s} LATLON  {lat:8.4f} {lon:9.4f}  {cfg.z_srce_km:4.1f}  {elev_out:7.3f}"
        )

    lines.append("")
    lines.append("END")
    lines.append("")
    return "\n".join(lines)


def _run_grid2time_in_place(grid2time_bin: str, in_dir: Path, in_name: str) -> str:
    cmd = [grid2time_bin, in_name]
    print("[RUN]", " ".join(cmd), f"(cwd={in_dir})")
    r = subprocess.run(cmd, cwd=str(in_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"Grid2Time failed with code {r.returncode}: {' '.join(cmd)} (cwd={in_dir})")
    return r.stdout


def _verify_tt_ps_outputs(time_dir: Path, out_ps_root_rel: str) -> None:
    """
    Heuristic check:
    - ensure there are multiple files starting with tt_PS
    - ensure both P and S seem present by filename tokens (common patterns: .P., .S., _P, _S)
    This is not perfect but catches the 'S overwrote P' failure mode.
    """
    root_name = Path(out_ps_root_rel).name  # tt_PS
    files = sorted([p.name for p in time_dir.glob(f"{root_name}*") if p.is_file()])
    if not files:
        raise RuntimeError(f"No output files found under {time_dir} for root '{root_name}'")

    has_p = any((".P" in fn) or ("_P" in fn) or ("P." in fn) for fn in files)
    has_s = any((".S" in fn) or ("_S" in fn) or ("S." in fn) for fn in files)

    # Many builds don't encode phase in filename; in that case, this heuristic can't prove both exist.
    # We still warn if it looks like only one set exists.
    if not (has_p and has_s):
        print("[WARN] Cannot confirm BOTH P and S outputs by filename pattern.")
        print("       Files:", files[:20], ("..." if len(files) > 20 else ""))
        print("       If you later see only one phase available, it likely means the second GTFILES overwrote the first.")
        print("       In that case, revert to tt_PS.P / tt_PS.S two-root output (the safe method).")
    else:
        print("[OK] Detected both P-like and S-like outputs for tt_PS.")


def main():
    cfg = Grid2TimeConfig()

    work_dir = Path(cfg.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    station_path = (work_dir / cfg.station_file).resolve()
    out_dir_path = (work_dir / cfg.out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    ps_in_path = (work_dir / cfg.grid2time_ps_in).resolve()

    # CLI override station file (optional)
    if len(sys.argv) >= 2:
        station_path = Path(sys.argv[1]).expanduser()
        if not station_path.is_absolute():
            station_path = (Path.cwd() / station_path).resolve()

    stations = _parse_station_lines(str(station_path))

    # Use relative paths inside .in (Grid2Time runs in work_dir)
    slow_p_rel = os.path.relpath(str((work_dir / cfg.slow_p_root).resolve()), str(work_dir))
    slow_s_rel = os.path.relpath(str((work_dir / cfg.slow_s_root).resolve()), str(work_dir))
    out_ps_rel = os.path.relpath(str((work_dir / cfg.out_ps_root).resolve()), str(work_dir))

    txt_ps = _format_grid2time_ps_in(cfg, "P", "S", [], [])  # placeholder to satisfy linter


    txt_ps = _format_grid2time_ps_in(
        cfg=cfg,
        slow_p_root=slow_p_rel,
        slow_s_root=slow_s_rel,
        out_ps_root=out_ps_rel,
        stations=stations,
    )

    ps_in_path.write_text(txt_ps, encoding="utf-8")

    print(f"[OK] wrote: {ps_in_path}")
    print(f"[OK] stations: {len(stations)}")
    print(f"[OK] output dir: {out_dir_path}")
    print(f"[OK] tt root: {(work_dir / cfg.out_ps_root).resolve()}  (single root in .in)")

    if not cfg.run_grid2time:
        return

    _run_grid2time_in_place(cfg.grid2time_bin, work_dir, ps_in_path.name)

    if cfg.verify_outputs:
        _verify_tt_ps_outputs(out_dir_path, out_ps_rel)

    print("[DONE] Grid2Time finished.")


if __name__ == "__main__":
    main()
