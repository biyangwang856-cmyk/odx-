# 当前道路正射候选导出（2026-10-14）

本次输出用于查看当前进展，未通过全幅验收。原 OGS2 正射保持不变；没有 push 或合并主分支。

## 采用的高度和照片

- 西段：`multiview-fit-02/west-lanes-fit.json`，道路高度留出误差 P90 0.1141 m，照片重投影误差 P90 2.3061 px。
- 中段：`multiview-fit-01/centre-lanes-fit.json`，道路高度留出误差 P90 0.1149 m，照片重投影误差 P90 2.3002 px。
- 这些是相对于 SfM 留出点的误差，不是测量控制点验证的绝对精度。道路语义仍是候选，支撑范围也不覆盖全部道路。
- 新的掩膜稠密点云拟合两段道路未通过：高度误差 P90 分别为 0.1372 m、0.1297 m，未采用这些新平面。
- 西段采用照片 0016 的道路投影，保留已验证的 0013 背景替换及新的源照片禁选保护；中段使用照片 0040 的单源投影，没有采用未通过的跨照片配准。
- 车辆像素投影到道路平面，不代表重建了真实车顶曲面。不同区域的源照片时刻不同；不能把车辆数量差当作消重率。

## 输出与质量状态

服务器输出目录：`/volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013/orthophoto-review-20261014-01`。

- `west/centre-current-context.tif`：带坐标的局部诊断影像；道路之外仍是平面投影，不能用于树木或建筑的几何结论。
- `west/centre-supported-road.tif`：仅显示道路、有效照片覆盖与已检查几何支撑范围的交集。透明区域表示没有覆盖这个验证范围，不是新生成的道路空洞。
- `west/centre-quality-scope.tif`：0 无效；1 道路外诊断上下文；2 道路平面外推；3 道路与已检查几何支撑范围交集。任何类别都不表示完成了独立语义验收。
- `west/centre-closed-patch-candidate.tif`、`closed-seam.jpg`、`closed-patch-provenance.npz`：闭合接缝的候选及失败证据。接缝避开了提议的车体、阴影和标线，但颜色/位置不一致仍不能忽略。
- `summary.json`：严格验收结果。两段都没有通过接缝阈值 P90≤25、max≤80；严格流程未输出通过的全幅成品。
- `UNACCEPTED-current-full-orthophoto-review.tif`：明确通过诊断开关生成的整幅预览。只替换西段和中段选定区域，其他像素保持 OGS2 原样；它不是验收通过的成品。
- `whole-review.json`：未验收整幅预览的生成与完整性检查。原图 SHA 未变、替换范围之外像素变化为 0、额外非 RGB 通道变化为 0。
- `current-full-provenance.tif`：0 保留原图；1 西段候选；2 中段候选。

西段接缝 RGB 差值 P90=33.0、max=158.0；中段 P90=37.6667、max=134.3333。离散图割可行不等于接缝质量合格。没有放宽阈值或把失败状态改成通过。

## 修改入口与复现

`app/model_engine/ogs2_road_surface.py` 负责道路平面和固定空间留出检查；`ogs2_source_guard.py` 负责源照片不确定车辆禁选；`ogs2_road_patch.py` 负责不可跨越对象的闭合硬接缝；`ogs2_road_color.py` 负责有界色差校正。

`scripts/lab/export_current_road_orthophoto.py` 绑定原图、道路拟合、照片及掩膜，导出局部候选，执行接缝验收。`scripts/lab/compose_orthophoto_review.py` 仅负责明确未验收的全幅预览与逐像素保留检查。后者的诊断开关不能改变验收结论。

在服务器上：

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
export PYTHONPATH="$PWD/app"
export LD_LIBRARY_PATH=/volume/wzhang/mesh-research-workspace/runtime/vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/volume/wzhang/mesh-research-workspace/runtime/vehicle-env/bin/python scripts/lab/export_current_road_orthophoto.py --job /volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013 --output /volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013/orthophoto-review-repeat-01
/volume/wzhang/mesh-research-workspace/runtime/vehicle-env/bin/python scripts/lab/compose_orthophoto_review.py --inputs /volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013/orthophoto-review-repeat-01 --include-rejected-seams-for-review
```

复现输出目录必须不存在；避免覆盖任何结果。原图固定，候选全幅另存。

本次导出前的回退点：`checkpoint/source-guard-hard-depth-20261014`（5a3b6ee）。查看旧代码：`git switch --detach checkpoint/source-guard-hard-depth-20261014`；返回开发分支：`git switch feature/by-dev`。没有 reset 或历史改写。
