# 源照片车辆掩膜与原生稠密重建对照

继承 `ccb7179`。正式 2.5D / OGS2 TIFF 保留；此阶段不宣称全幅车辆残影已经消除。

## 已确认结果与拒绝结果

扩大窗口的配准必须同时平移固定空间留出单元原点。原窗口坐标移到扩展窗口时，测量加 `[250,200]`，配置使用 `cell_origin_xy=(250,200)`；训练、验证仍是同一物理点集，位移/Jacobian 约束覆盖完整扩展域。西段 0→2 的留出 P90 0.7277 px、最大 1.4442 px 与原结果一致。新增反例测试验证这一不变性。

整段标线保护试验不是有效改进。`source-background-markings-03` 西段保留 8 组、替换 0 组，不能报为消重收益。该选项只保留为诊断入口，默认不启用。保留前面的失败日志和拒绝记录。

单张照片道路平面投影改善了局部标线与车体，但旧图接口仍有跳变；把更多小补丁塞回旧图不能自动解决全路段几何。新对照从稠密重建之前排除动态物体，测试旧高度场是否受到车辆污染。

## 实现

`scripts/lab/mask_vehicle_source_images.py` 在现有去畸变照片的完整像素坐标中运行 GroundingDINO + SAM。全帧检测加可选重叠切片，提高小型车辆的检测细节。地面高度估计仅过滤尺寸明显异常的提议，不修改相机或路面几何。完整车体、有限附近暗影和约 0.15 米边缘扩展写入 8 位灰度掩膜：0 排除，255 可用。255 不等于静态背景证明。原 RGB、SfM 重建和轨迹均只读。

20 张试验照片的全帧版本识别 451 个跨照片实例，不能解释为 451 辆独立车辆。抽查仍有白车漏检、施工材料/小屋顶误检；因此没有将掩膜直接部署到全部 405 张照片。切片版本用于继续验证这些问题，不以检测数量增加代替准确率验证。

原生 OpenSfM 导出器把绝对掩膜路径写入 `scene.mvs`；本版本使用 `undistorted/masks/<完整照片名>.png`。ODX 的 OpenMVS 原生阶段在掩膜存在时使用 `--ignore-mask-label 0`。不修改 RGB 来模拟遮罩。

`scripts/lab/run_vehicle_mask_dense_ab.py` 对同样 20 张照片运行 control/masked 两组。相机、轨迹和照片字节绑定一致；只改变掩膜和排除标签。各组独立新建深度图目录，禁止复用旧缓存。使用固定分辨率和有上限的 CPU 线程，避免共享 GPU 显存争用。实验输出不覆盖原项目。

判定收益需要检查车辆范围的原生深度有效性、静态标线/路面完整性和最终投影；仅点数减少不能证明正射改善。深度掩膜也不能代替最终的完整车体选图：纹理端仍需保持时刻一致与可追溯来源。

## 文件与命令

- `tests/test_vehicle_mask_footprint.py`：像素/米尺度、主点与 max(width,height) 内参、平移不变性和非法几何反例。
- `tests/test_road_marking_mask.py`：白标线、灰路面、杆体、大亮区及道路外误检反例。
- `tests/test_road_registration.py`：增加扩展窗口坐标原点与物理留出集合一致性测试。

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
ROOT=/volume/wzhang/mesh-research-workspace
JOB=$ROOT/runs/mangrove-077-20261013
export PYTHONPATH=$PWD/app
export LD_LIBRARY_PATH=$ROOT/runtime/vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu
export CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
$ROOT/runtime/vehicle-env/bin/python scripts/lab/mask_vehicle_source_images.py \
  --project $JOB/datasets/project \
  --image-list $JOB/vehicle-source-masks-20261014/image-list.txt \
  --models $ROOT/runtime/vehicle-models --tile-size 1600 \
  --output $JOB/vehicle-source-masks-20261014/reproduce-mask-new
# A/B 导出输入准备完成后，在独立 trial 中执行：
$ROOT/runtime/vehicle-env/bin/python scripts/lab/run_vehicle_mask_dense_ab.py \
  --workspace $ROOT --trial $JOB/vehicle-source-masks-20261014/geometry-ab-02 --threads 8
```

输出目录必须不存在。复现输入绑定、导出器与实验状态保存在实验目录。参考 [SAHI](https://github.com/obss/sahi) 的重叠切片思想，本项目没有安装/调用 SAHI 包。参考 [OpenMVS #979](https://github.com/cdcseacave/openMVS/issues/979) 的掩膜标签排查，不能假设任意黑白格式自动正确。

回退到此阶段前：`git switch --detach checkpoint/road-real-background-20261014`。返回开发：`git switch feature/by-dev`。不推送 GitLab，不合并主分支。
