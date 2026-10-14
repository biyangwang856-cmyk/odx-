# 原生车辆掩膜深度机制验收

本阶段继承 `c9e7a6b`。2026-10-14 已在服务器读取到两组原生 OpenMVS CPU 对照成功完成，随后验证了全部 20 对原生 DR 深度文件。

## 同输入对照

目录：`runs/mangrove-077-20261013/vehicle-source-masks-20261014/geometry-ab-02`。

`prepare_vehicle_mask_dense_ab.py` 准备同一组 20 张照片，保存相同相机、轨迹和照片输入；掩膜来自已完成的 `raw-mask-pilot-03`。原生 OpenSfM 导出两组 scene.mvs。`run_vehicle_mask_dense_ab.py` 使用独立、无缓存的深度目录，固定 8 CPU 线程；掩膜组使用 `--ignore-mask-label 0`。PatchMatch 的随机差异不能完全排除，所以点数变化不能单独解释为质量收益。

未加掩膜组运行约 1072 秒，生成 19,427,998 个融合点；掩膜组约 971 秒，生成 18,949,206 个点。两组各有 20 张原生深度图，均成功退出。没有运行最终三维网格交付，也没有覆盖正式正射。

## 实测机制

`inspect_vehicle_mask_depth_ab.py` 只读解析 OpenMVS 的 DR 格式，检查两组分辨率、K/R/C 一致，并验证原照片相机内参、灰度标签和掩膜坐标。按照原生深度分辨率采样掩膜，向内去除一个像素边界，避免把缩放边缘当作可靠内部。

20 张照片合计 3,476,319 个排除内核像素。按像素数加权，其深度有效率：未加掩膜 **56.82%**，掩膜组 **0.280%**。单张照片最大残留有效率 **1.127%**。说明排除掩膜确实参与了重建；残留深度尚需确认是否来自后处理填洞、掩膜缩放或其他阶段，未确认原因。

掩膜内包含车辆/局部阴影提议、误检和无效照片像素，不是人工标注的纯车辆真值。因此这些数字不是车辆检测准确率、消重率或正射提升率。

可用掩膜区域的深度覆盖率中位数从 **75.15%** 到 **74.48%**。该区域也没有全部通过静态语义认证，不能据此证明道路精度提升；需要继续检查误删的静态结构、白车遗漏、投影标线和全幅接缝。

## 输出与继续入口

- `vehicle-source-masks-20261014/depth-review-01/report.json`：逐照片深度有效率、可用区域变化、原生文件哈希、未经认证的质量标记。
- `depth-review-01/summary.json`：上述汇总。
- `depth-review-01/*.jpg`：原照片上显示原生有效深度的对比，明亮代表有深度、暗色代表没有深度。它不是新的正射成品。
- `geometry-ab-02/input-bindings.json` / `dense-status.json` / `*/dense/run.log`：参数、数据绑定和退出状态。

重现必须使用新试验输出目录，不能覆盖现有数据：

```bash
ROOT=/volume/wzhang/mesh-research-workspace
JOB=$ROOT/runs/mangrove-077-20261013
cd $ROOT/mesh-research
PY=$ROOT/runtime/vehicle-env/bin/python
$PY scripts/lab/prepare_vehicle_mask_dense_ab.py \
  --workspace $ROOT --project $JOB/datasets/project \
  --masks $JOB/vehicle-source-masks-20261014/raw-mask-pilot-03 \
  --image-list $JOB/vehicle-source-masks-20261014/image-list.txt \
  --output $JOB/vehicle-source-masks-20261014/geometry-ab-new
$PY scripts/lab/run_vehicle_mask_dense_ab.py \
  --workspace $ROOT --trial $JOB/vehicle-source-masks-20261014/geometry-ab-new --threads 8
$PY scripts/lab/inspect_vehicle_mask_depth_ab.py \
  --trial $JOB/vehicle-source-masks-20261014/geometry-ab-new \
  --output $JOB/vehicle-source-masks-20261014/depth-review-new
```

解析器检查包括已知二进制头、截断、非法分辨率、负/非有限深度和边界剔除。原生数据实际读取全部 20 对通过。参考格式：[OpenMVS MvsUtils.py](https://github.com/cdcseacave/openMVS/blob/master/scripts/python/MvsUtils.py)。脚本没有复制原照片 RGB，也不修改原生深度图。

## 保留与回退

此阶段前回退：`git switch --detach checkpoint/source-vehicle-mask-pilot-20261014`。返回开发：`git switch feature/by-dev`。更早的 `checkpoint/road-real-background-20261014` 仍保留。没有 push 或主分支合并。

D 盘保留了此阶段之前的仓库完整快照，包含 Git 历史、忽略与未提交文件；它不是整个工作区环境/实验数据备份。正式全幅 TIFF 仍保持原结果。本阶段完成的是机制验收，最终消除半车、残影、标线变形和接口接缝仍未完成。
