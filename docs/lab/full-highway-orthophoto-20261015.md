# 整条主路正射影像替换结果

生成于 2026-10-15。结果仍标为候选，不能声称所有残影和接缝已完全解决。

## 结果与核查

- 完整文件：`CANDIDATE-main-highway-orthophoto.tif`，24945 × 24970，EPSG:32650，约 3 cm 像素间距。
- 原坐标系、仿射变换、范围、尺寸、波段与像素间距完全一致。
- 替换范围外变动像素 0；替换范围内与道路候选不一致的像素 0。
- 声明道路核心替换覆盖率 99.127722%；未获得有效覆盖的道路像素 32541。
- 34 个路面分段中 11 个通过几何检查，23 个未通过。未通过检查的模型及训练支持区外外推明确保留在报告中，混合后的连续高度场也没有独立精度认证。
- 车身保护区切割计数是遮罩上的检测结果，不等于人工确认零半车，也不等于消除比例。保留前景像素包含误检和配准差异，不能当成车辆数。
- JPEG 用于查看，不能代替带地理信息的 GeoTIFF。打开 `index.html` 查看整条道路和 6 处局部滑块。

## 方法

1. 读取既有 OpenSfM 内参、相机位置、方向、去畸变照片和地理偏移。
2. 从真实照片核对主路范围。固定的场景范围种子不是地面控制点，不能视为新场景通用自动分割。
3. 在静态路面上建立多视图匹配和三角化，按训练/独立检查划分拟合路面高度。稀疏/未通过区域作为诊断估计保留。
4. 把实际照片投到原有 3 cm 世界网格。车辆使用 GroundingDINO + SAM 遮罩；低置信度保留完整检测框。
5. 只从有效、未被车辆遮罩覆盖的真实照片选取背景。无可用背景时保留原有前景，不生成虚构路面。不跨照片做车身透明混合。
6. 整块车身优先使用同一来源；必要时以多源图割联合选取背景。接缝检查两侧车身遮罩，绕开车身。时序颜色差也可能把错位标线当成车辆：硬禁选不可行时，该类候选改作接缝代价，车身检测仍保持硬约束。
7. 在完整栅格中只替换声明区域，逐块重新读取验证像素和地理信息，原图 SHA-256 再次核对。

## 文件与修改入口

服务器根目录：`/volume/wzhang/mesh-research-workspace`。
仓库：`mesh-research`；数据与中间件：`runs/mangrove-077-20261013`。

| 文件 | 作用 / 修改入口 |
| --- | --- |
| `app/model_engine/ogs2_highway_background.py` | 时序异常候选、车身禁选、真实背景来源、单源/联合图割、色差门槛 |
| `app/model_engine/ogs2_highway_join.py` | 可横向绕行的硬接缝；检查实际切边两端的车辆/有效像素 |
| `scripts/lab/highway_full_plan.py` | 原世界网格、相机与道路运行范围规划 |
| `scripts/lab/highway_full_source_scaffold.py` | 用照片生成道路搜索底图；其临时 3 m 高度不是最终几何 |
| `scripts/lab/highway_full_reviewed_scope.py` | 本场景道路范围种子与边界；新场景必须重新核对 |
| `scripts/lab/highway_full_geometry.py` | 静态匹配、三角化、训练与独立检查；没有通过的测量不能包装为通过 |
| `scripts/lab/highway_full_render.py` | 高度场、照片投影、原始 6 cm DSM 按世界坐标采样、车身与来源缓存、分段拼接、道路边界 |
| `scripts/lab/export_full_highway_orthophoto.py` | 合成完整 GeoTIFF，逐块核验替换范围及坐标，导出局部对比和校验清单 |
| `scripts/lab/compose_full_highway_review.py` | 离线查看页面及本文档 |
| `source-photo-provenance.tif` | 道路范围内的真实照片编号；0 表示保留原图/无新来源 |
| `retained-foreground.tif` | 保留前景/异常候选；不是车辆计数或真值 |
| `road-geometry-measurements.json` | 路面三角化、训练与独立检查的全部记录 |
| `quality-report.json` | 本次拼接、覆盖、几何限制和坐标核验 |
| `manifest.json` | 原始文件不变的 SHA-256 和结果文件逐项校验 |

## 从当前保存的中间结果复现

在服务器执行；导出使用新的目录名，避免覆盖现有证据。

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
export PYTHONPATH="$PWD/app"
export LD_LIBRARY_PATH=/volume/wzhang/mesh-research-workspace/runtime/vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/volume/wzhang/mesh-research-workspace/runtime/vehicle-env/bin/python
JOB=/volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013
"$PY" scripts/lab/export_full_highway_orthophoto.py \
  --job "$JOB" \
  --candidate "$JOB/highway-full-20261014-01/full-render-scale1-09" \
  --output "$JOB/orthophoto-full-highway-replay-01"
"$PY" scripts/lab/compose_full_highway_review.py "$JOB/orthophoto-full-highway-replay-01"
"$PY" -m pytest -q tests/test_highway_background.py tests/test_highway_join.py tests/test_road_patch.py
```

从照片重新执行的完整阶段顺序、缓存规则及修复记录见仓库 `docs/lab/full-highway-orthophoto-20261015.md`。现有实验目录不能原地重跑；先修改运行目录名，在副本中运行。流程仍复用原项目 SfM 和去畸变照片，未引入中间尝试过的单目/SfM 优化。

## Git 回退

本轮前：`checkpoint/before-full-highway-20261014`。
本轮后：`checkpoint/full-highway-orthophoto-20261015`。

```bash
git switch --detach checkpoint/before-full-highway-20261014
# 查看本轮完成状态
git switch --detach checkpoint/full-highway-orthophoto-20261015
# 返回开发分支
git switch feature/by-dev
```

切换前先检查 `git status`，保留后来产生的未提交修改。未推送 GitLab，未合并主分支。原影像、中间候选和失败记录均保留。


## 本轮完整阶段顺序和保存的输入

本次是在已有 2.5D + OGS2 基础上运行。没有重跑原 SfM/MVS，也没有恢复中间试过的单目或 MP-SfM 算法。

1. `highway_full_plan.py` 读取旧 OGS2 GeoTIFF 和 `datasets/project/opensfm/undistorted/reconstruction.json`，输出 `highway-full-20261014-01/camera-corridor-plan.json`。道路运行窗口为原图 `[0, 17000, 24945, 4500]`。
2. `highway_full_source_scaffold.py` 用去畸变照片在临时 3 m 高度生成 12 cm 搜索底图 `source-scaffold-02`。这个高度仅供核对道路范围，不是最终测量结果。
3. `highway_full_reviewed_scope.py` 输出 `road-mask-08`。保留 9 个按照片核对的范围种子；自动范围提议曾误选辅路和屋顶，没有采用。渲染中的南侧范围再限制为中央分隔带以南 160 个 12 cm 搜索像素，排除平行绿化隔离带。这是本场景范围约束，不是通用自动道路识别的完成证明。
4. `highway_full_geometry.py` 输出 `geometry-04`。沿道路分 17 段，南北车道分别测量，共 34 个检查单元。每段 66 m，步长 42 m；采用真实静态路面 SIFT 匹配、至少 3 视图三角化。5 m 空间格中固定一部分作独立检查，不用独立检查误差删点。
5. `highway_full_render.py` 先生成 6 cm 候选并在实际投影图上运行 GroundingDINO + SAM；`full-render-scale2-06/source-body-masks` 保存车身遮罩。使用原相机参数、连续路面高度场和原世界网格重新投影原始照片，保持相机读取器每次按照片取值，避免复用会更新姿态的共享相机对象。
6. `full-render-scale1-01` 生成原始 3 cm 全线来源图。`full-render-scale1-04` 重新生成西段前三段并与不变的东段拼接。所有过程中旧图和源照片均只读。
7. `full-render-scale1-09` 完成边界闭合。把缓存缺少上下文的像素与真实车身遮罩分开：缺少上下文可以禁选为补洞来源，但不能连成虚构的车辆。外边界无法连续绕开前景时，包围整块已观测的路面前景；不能完整观测或超出允许范围的整块区域保留原图。
8. `export_full_highway_orthophoto.py` 把道路候选按掩膜写回原整幅栅格；重新读取 2,401 个输出块，核对区域内像素、区域外像素、坐标参数和原始文件 SHA-256。`compose_full_highway_review.py` 输出整条道路和 6 个局部的对比页面。

原始 DSM 为 6 cm 网格，正式栅格为 3 cm 网格，不能直接把 3 cm 的像素窗口用于 DSM。渲染器先计算世界坐标范围，再用 DSM 的 `from_bounds` 取样。本轮保存了这项修复。

## 几何与前景的关键限制

路面训练要求内点比例至少 0.65、坡度范数不超过 0.2；独立高度检查 P90 <= 0.12 m、最大值 <= 0.4 m；照片投影误差 P90 <= 2.5 px、最大值 <= 5 px。34 单元只有 11 个通过，23 个未通过。连续高度场使用训练支持的诊断估计和局部外推，并没有通过完整道路的独立精度认证。

真实背景选取至少需要 3 个其他有效观测；时序差异候选不直接写入影像。照度只作有限常数修正，单通道限制为 ±12 灰度级。接缝和图割优先保留整块车身；没有真实可用背景就保留前景。输出中的保留前景和补丁数不能当成唯一车辆数或消除比例。

本轮对比基线是本项目原有的 `vehicle-ogs2/ogs2_orthophoto.tif`。人工局部核查显示标线、车身完整性明显改善，但西段色差接缝和少量道路阴影/色块仍在；车辆遮罩可能漏检或误检。本轮没有证明全线车辆重复和残影全部消除，也没有建立绝对地面位置误差的真值。

道路核心替换覆盖率不是去车成功率。正射影像保存了原始 CRS、仿射变换和像素网格，仍是带坐标的投影栅格；3 cm 是像素间距，不是 3 cm 测量精度。

## 从保存的来源图重做最终边界

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
export PYTHONPATH="$PWD/app"
export LD_LIBRARY_PATH=/volume/wzhang/mesh-research-workspace/runtime/vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
PY=/volume/wzhang/mesh-research-workspace/runtime/vehicle-env/bin/python
BASE=/volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013/highway-full-20261014-01
"$PY" scripts/lab/highway_full_render.py --scale 1 --run-id replay-01 \
  --finalize-from "$BASE/full-render-scale1-04" \
  --repair-west-from "$BASE/full-render-scale1-01" --diagnostic
```

之后把前文导出命令的 `--candidate` 改为 `"$BASE/full-render-scale1-replay-01"`。`--diagnostic` 明确保留未通过的几何检查，不会把结果改成已验收。重跑初始阶段时应先在新运行目录副本中修改脚本的 `base`，避免覆盖现有中间件。

## 软件核查

本轮道路背景、整块前景保护、缓存上下文和接缝的 32 项专项测试通过；本轮相关代码通过 Ruff 格式与静态检查。整体工程的旧代码仍有既有 Ruff 问题；完整工程测试在当前正射环境中因缺少 `vtk` 和 `typer` 收集失败。因此没有把专项测试通过包装成整个工程 CI 通过。原始错误记录保存在 `highway-full-20261014-01/code-quality`。

## 保存范围

服务器保留原始输入、原图、所有候选和失败记录。D 盘另存最终 GeoTIFF、局部对比、质量/几何记录和完整 Git 仓库备份。仓库备份包含 `.git`，但不包含仓库同级 `runtime`、`runs` 的全部大模型、照片和缓存；这些不能被误认为已全部备份。
