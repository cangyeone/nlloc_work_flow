
# NonLinLoc（NLLoc）区域定位流水线 — 交付级 README

本仓库提供一套**从区域速度模型到并行定位、再到目录导出**的完整 NLLoc 工程化流程，适用于 REAL 拾取结果或相近格式的事件数据。



## 1. 流程总览

```
散点速度模型
   │
   ▼
[Step1] 插值为规则经纬深网格 (Vp/Vs)
   │
   ▼
[Step2] 转换为 NLLoc 慢度网格 (SLOW_LEN, P/S)
   │
   ▼
[Step3] Grid2Time 生成台站走时表 (tt grids)
   │
   ▼
[Step4] REAL → NLLoc 输入 (STATION / OBS / IN)
   │
   ▼
[Step5] 并行运行 NLLoc（事件级）
   │
   ▼
[Step6] 解析 .hyp → 目录导出（csv/tsv/txt）
```



## 2. 环境与依赖

### 必需软件

* **NonLinLoc**（含 `NLLoc` 与 `Grid2Time` 可执行文件）
* **Python ≥ 3.9**

### Python 依赖

```bash
pip install numpy scipy
```

> 说明：`step2.make_nll_grid.py` 使用了 `nllgrid`（NLLoc Python 辅助类）。请确保你的环境中已安装/可 import（或按你现有方式放置）。



## 3. 目录结构（推荐）

```
nlloc_script/
  demo_data/
    velo.txt
    velocity_grid_*.npz
    slow_P_cubic.*.mod.hdr/.buf
    slow_S_cubic.*.mod.hdr/.buf
    location.txt
    time/
      tt_PS.*.mod.hdr/.buf
  out_dir/
    sc.stations
    real.obs
    nlloc.in
    nlloc.temp.in
    work/<event_id>/
    out/*.hyp
```



## 4. 逐步使用指南（最小可运行）

### Step 1 — 插值散点速度模型

```bash
python step1.interp_vel_data.py
```

**输入**：`demo_data/velo.txt`（`lon,lat,depth_km,vp,vs`）
**输出**：`velocity_grid_*.npz`

**检查要点**

* `.npz` 中 `Vp_grid/Vs_grid` 是否无 NaN
* 网格范围是否覆盖目标区域



### Step 2 — 生成 NLLoc 慢度网格（SLOW_LEN）

```bash
python step2.make_nll_grid.py
```

**输出**

* `slow_P_cubic.P.mod.hdr/.buf`
* `slow_S_cubic.S.mod.hdr/.buf`

**关键参数**

* `lon0/lat0`：本地切平面参考点
* `grid_step_km`：立方网格步长（dx=dy=dz）

> **必须保持一致**：`lon0/lat0` 将在 Step 3/4 的 `TRANS SIMPLE` 中复用。



### Step 3 — Grid2Time 生成走时表

```bash
python step3.grid2time.py
```

**输入**

* `location.txt`：`net sta lon lat elev_km`
* 慢度网格 root：`slow_P_cubic` / `slow_S_cubic`

**输出**

* `demo_data/time/tt_PS.*.mod.*`

#### ⚠ 重要：P/S 覆盖风险

默认脚本**使用同一个输出 root（`tt_PS`）同时写 P 和 S**。某些 Grid2Time 构建可能发生覆盖。

**推荐安全做法（二选一）**

1. **拆分 root**：`tt_P` 与 `tt_S`
2. **确认文件名含相位标识**（`.P.` / `.S.`）



### Step 4 — 生成 NLLoc 输入（STATION / OBS / IN）

```bash
python step4.make_nll_in_file.py
```

**输入**

* `real.txt`（REAL 格式）
* `location.txt`
* 走时表目录与 root（如 `time/tt_PS`）

**输出**

* `out_dir/sc.stations`
* `out_dir/real.obs`
* `out_dir/nlloc.in`、`out_dir/nlloc.temp.in`

**关键一致性要求（最常见错误源）**

* **台站名**：`NET_STA` 在 Grid2Time（`GTSRCE`）、OBS、NLLoc 中必须一致
* **坐标系**：`TRANS SIMPLE lat0 lon0` 必须与 Step2/3 一致
* **REAL 格式**：相对到时列默认在第 5 列（`parts[4]`）



### Step 5 — 并行定位（事件级）

```bash
python step5.parallel_locate.py
```

**输入**

* `out_dir/real.obs`
* `out_dir/nlloc.in`
* `time/tt_PS`（或你采用的 root）

**输出**

* `out_dir/out/*.hyp`
* `out_dir/work/<event_id>/`
* `parallel_loc_summary.json`

#### ⚠ 推荐修订（防止事件间覆盖）

当前 patch 使用 `outputFileRoot = out_dir/out`（目录）。
**建议改为事件唯一前缀**，例如：

```text
LOCFILES <obs> NLLOC_OBS <tt_root> <out_dir>/out/<event_id>_
```

> 这样可确保每个事件的 `.hyp/.scat/.sum` 文件不互相覆盖。



### Step 6 — 导出目录（catalog）

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

**支持**

* 输出格式：`csv | tsv | txt`
* 方案选择：`--prefer auto|exp|ml`
* 质量过滤：`Nphs / RMS / Gap`



## 5. 参数约定与设计选择

* **SLOW_LEN**：`(1 / velocity) * grid_step`（秒）
* **Z 正方向**：向下为正（km）
* **局部投影**：切平面近似（适用于中等区域）
* **LOCGRID**：默认从走时表 `.hdr` 自动派生（推荐）



## 6. 常见报错与排查

### 1) “找不到台站 / 走时”

* 检查 **台站名一致性**（`NET_STA`）
* 检查 `LOCFILES` 的走时 root 是否正确

### 2) 定位结果明显偏移

* `TRANS SIMPLE lat0 lon0` 是否与 Step2/3 一致
* 是否混用了不同参考点的慢度/走时

### 3) 只生成 P 或 S 的走时

* 检查 Grid2Time 是否发生覆盖
* 采用拆分 root 的安全方案

### 4) 并行输出互相覆盖

* 采用 **事件唯一 outputFileRoot**（见 Step5 推荐修订）



## 7. 复现实验的最小清单

* 固定 `lon0/lat0`
* 固定 `grid_step_km`
* 固定走时 root 命名
* 固定 EventID 生成规则
* 保存 `parallel_loc_summary.json`



## 8. 联系与扩展

* 可无缝扩展到 **P-only / S-only / 多模型对比**
* 可替换 REAL 为其他拾取格式（仅需改 Step4 解析）
