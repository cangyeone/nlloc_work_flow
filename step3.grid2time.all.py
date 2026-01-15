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

    # 注意：NonLinLoc/Grid2Time 读取慢度网格 root（不带 .hdr/.buf）
    # 你前面已经确认需要显式 .P.mod / .S.mod
    slow_p_root: str = "slow_P_cubic"
    slow_s_root: str = "slow_S_cubic"

    # ---------- Outputs（相对 work_dir） ----------
    out_dir: str = "time"

    # 统一 root：生成 time/tt_PS.P.* 和 time/tt_PS.S.*
    out_ps_root: str = "time/tt_PS"

    grid2time_p_in: str = "grid2time_P.in"
    grid2time_s_in: str = "grid2time_S.in"

    # ---------- NLLoc Grid2Time Common Controls ----------
    control_1: int = 1
    control_seed: int = 12345
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


def _format_grid2time_in(
    cfg: Grid2TimeConfig,
    wave_type: str,
    slow_root: str,
    out_root: str,
    stations: List[Tuple[str, str, float, float, float]],
) -> str:
    lines = []
    lines.append("# ========= 通用控制参数 =========")
    lines.append(f"CONTROL {cfg.control_1} {cfg.control_seed}")
    lines.append(f"TRANS SIMPLE {cfg.lat0:.6f} {cfg.lon0:.6f} {cfg.z0:.1f}")
    lines.append("")
    lines.append("# ========= 输入模型网格 & 输出走时表 =========")
    lines.append("# GTFILES inputFileRoot outputFileRoot waveType iSwapBytesOnInput")
    lines.append(f"GTFILES {slow_root} {out_root} {wave_type} {cfg.iswap}")
    lines.append("")
    lines.append("# 3D 网格 + 不计算射线角度")
    lines.append(f"GTMODE {cfg.gtmode} {cfg.angles}")
    lines.append(f"GT_PLFD {cfg.gt_plfd_1:.1e} {cfg.gt_plfd_2}")
    lines.append("")
    lines.append("# ========= 台站列表 =========")
    lines.append("# GTSRCE label LATLON  lat  lon  z(km, 正向向下)  elev(km)")
    lines.append("")

    for net, sta, lon, lat, elev_km in stations:
        label = f"{net}_{sta}"
        elev_out = elev_km if cfg.keep_elev_sign else (-elev_km)
        lines.append(
            f"GTSRCE {label:<10s} LATLON  {lat:8.4f} {lon:9.4f}  {cfg.z_srce_km:4.1f}  {elev_out:7.3f}"
        )

    lines.append("")
    lines.append("END")
    lines.append("")
    return "\n".join(lines)


def _run_grid2time_in_place(grid2time_bin: str, in_dir: Path, in_name: str) -> None:
    cmd = [grid2time_bin, in_name]
    print("[RUN]", " ".join(cmd), f"(cwd={in_dir})")
    r = subprocess.run(cmd, cwd=str(in_dir), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(r.stdout)
    if r.returncode != 0:
        raise RuntimeError(f"Grid2Time failed with code {r.returncode}: {' '.join(cmd)} (cwd={in_dir})")


def main():
    cfg = Grid2TimeConfig()

    # 固定工作目录：无论从哪里运行脚本，所有 IO 都落到 demo_data
    work_dir = Path(cfg.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # 把所有路径解析为 work_dir 下的绝对路径（确保写入位置固定）
    station_path = (work_dir / cfg.station_file).resolve()
    out_dir_path = (work_dir / cfg.out_dir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)

    p_in_path = (work_dir / cfg.grid2time_p_in).resolve()
    s_in_path = (work_dir / cfg.grid2time_s_in).resolve()

    # 允许命令行覆盖 station 文件（可选）
    if len(sys.argv) >= 2:
        station_path = Path(sys.argv[1]).expanduser()
        if not station_path.is_absolute():
            station_path = (Path.cwd() / station_path).resolve()

    stations = _parse_station_lines(str(station_path))

    # 输入慢度 root（相对 work_dir）
    slow_p_rel = os.path.relpath(str((work_dir / cfg.slow_p_root).resolve()), str(work_dir))
    slow_s_rel = os.path.relpath(str((work_dir / cfg.slow_s_root).resolve()), str(work_dir))

    # 输出 root：统一前缀 tt_PS，分别输出 tt_PS.P / tt_PS.S
    out_ps_abs = (work_dir / cfg.out_ps_root).resolve()
    out_p_abs = Path(str(out_ps_abs) + ".P")
    out_s_abs = Path(str(out_ps_abs) + ".S")
    out_p_rel = os.path.relpath(str(out_p_abs), str(work_dir))
    out_s_rel = os.path.relpath(str(out_s_abs), str(work_dir))

    txt_p = _format_grid2time_in(cfg, "P", slow_p_rel, out_p_rel, stations)
    txt_s = _format_grid2time_in(cfg, "S", slow_s_rel, out_s_rel, stations)

    p_in_path.write_text(txt_p, encoding="utf-8")
    s_in_path.write_text(txt_s, encoding="utf-8")

    print(f"[OK] wrote: {p_in_path}")
    print(f"[OK] wrote: {s_in_path}")
    print(f"[OK] stations: {len(stations)}")
    print(f"[OK] output dir: {out_dir_path}")
    print(f"[OK] tt root (P): {out_p_abs}.*")
    print(f"[OK] tt root (S): {out_s_abs}.*")

    if not cfg.run_grid2time:
        return

    # 在 work_dir 运行 Grid2Time（参数只传文件名）
    _run_grid2time_in_place(cfg.grid2time_bin, work_dir, p_in_path.name)
    _run_grid2time_in_place(cfg.grid2time_bin, work_dir, s_in_path.name)

    print("[DONE] Travel-time grids generated: tt_PS.P.* and tt_PS.S.*")


if __name__ == "__main__":
    main()
