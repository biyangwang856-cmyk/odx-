# 真实照片道路背景与亮度校正试验

本阶段继承 02c17ae。正式原生 2.5D 与 OGS2 TIFF 均保留。所有新增 TIFF 是局部候选，不能代替已通过验收的全幅。

## 方法与结果

道路内部首先选用同一张真实照片的局部平面投影，避免不同拍摄时刻的车体混合。若另一张照片在完整车身和附近阴影范围没有检测到其他对象、没有缺测，再测试复制其实际路面。重叠、覆盖不足或接缝颜色差超限时，保留原完整车体。检测器没有检测到对象本身不等于证明空道路；JSON 明确保留此限制。

`source-background-trial-01` 使用扩大坐标范围重新拟合的配准，西段和中段都未通过默认配准检查。这一轮开启明确的诊断选项后各替换一组车体，不能作为正式结果；西段该轮配准甚至使留出误差变大。已保留接受/拒绝记录，不把诊断使用改写成配准通过。

`source-background-trial-02` 修正坐标范围的处理：继续使用原配准测量坐标和固定留出单元，照片范围扩大只用于完整车体上下文，替换必须处在原验证范围内。西段 0→2 配准通过默认检查，P90 0.7277 px、最大 1.4442 px；替换 1 组，保留 7 组。它还不能证明全路段车辆消重或背景语义。

局部复制路面会出现曝光差。`ogs2_road_color.py` 仅从对象和阴影之外的路面邻域拟合 RGB 常数偏移，每通道不超过 12；固定空间留出检查全部像素，不进行几何修改或生成式补画。校正只改变来源照片颜色，来源索引和 RGB 偏移单独保存。

`source-background-colour-01` 的诊断对照：西段应用偏移 [6,8,8]，留出颜色差中位数从 7.33 降至 2.67，接缝 P90 为 3.67；中段 [-7,-8,-8]，从 7.67 降至 2.00，接缝 P90 为 3.00。这是 0–255 RGB 数值差，不能当作地理误差或全幅改善率。

`source-background-colour-02` 是仅对西段原坐标验证通过版本的颜色校正重现。输出仍是局部道路候选。

## 文件

- `app/model_engine/ogs2_road_color.py`：只拟合有界 RGB 偏移，保留独立留出与语义/几何未认证标记。
- `scripts/lab/road_source_background_trial.py`：真实照片重新采样、实例/局部暗影提议、整组替换或整组保留。默认拒绝未通过配准的区域；`--diagnostic-geometry` 才允许单独生成未接受的诊断结果。
- `scripts/lab/road_background_color_trial.py`：重放已保存的照片输入，加入路面颜色校正。输出颜色参数、像素来源索引、切边、局部 GeoTIFF 和拒绝记录。
- `tests/test_road_color.py`：已知偏移、车辆不参与拟合、留出异常不被丢弃、超限校正、无支持、整数裁剪及输入错误的反例。

研究参考：[OpenCV 曝光补偿](https://docs.opencv.org/4.x/d2/d37/classcv_1_1detail_1_1ExposureCompensator.html) 和 [LPAM ICCV 2025](https://github.com/tlliao/LPAM_seam-cutting)。本项目采用记录来源、有界校正和留出检查的专用实现；没有直接将全景自由形变输出当作测绘真值。

## 重现西段默认配准版本

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
export PYTHONPATH="$PWD/app"
export LD_LIBRARY_PATH=/volume/wzhang/mesh-research-workspace/runtime/vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
PY=/volume/wzhang/mesh-research-workspace/runtime/vehicle-env/bin/python
JOB=/volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013
$PY scripts/lab/road_source_background_trial.py \
  --workspace /volume/wzhang/mesh-research-workspace --job "$JOB" \
  --surface-dir "$JOB/road-surface-20261013" \
  --output "$JOB/road-surface-20261013/background-reproduce-new" --regions west
$PY scripts/lab/road_background_color_trial.py \
  --inputs "$JOB/road-surface-20261013/background-reproduce-new" \
  --metadata "$JOB/road-surface-20261013/expanded-patch-inputs-02" \
  --output "$JOB/road-surface-20261013/background-colour-reproduce-new" --regions west
```

输出路径必须不存在。所有依赖和计算均在服务器；本机仅接收小型预览和报告。

## 仍未完成

旧图接口处的标线跳变、全路段接缝、东段道路几何、更多车辆历史位置的真实背景覆盖仍需处理。灰色稀疏点并不自动属于路面：新导出的 6431 个候选、18978 条观测仍混有树木和车辆；其全局筛选未通过道路平面检查，没有强制写入高度场。下一轮从车道标线角点获取更明确的静态配准证据。

回退到本阶段之前：`git switch --detach checkpoint/road-coherent-patch-20261014`（02c17ae）。更早的 d413eb1 和 1cbe2a7 标签仍存在。返回当前开发使用 `git switch feature/by-dev`。没有推送或合并主分支。
