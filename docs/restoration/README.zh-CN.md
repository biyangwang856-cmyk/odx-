# 仅恢复2.5D与车辆优化（2026-10-13）

代码仓库位于 `/volume/wzhang/mesh-research-workspace/mesh-research`。这次上传的 ZIP 没有 `.git`，因此建立新的本地提交历史；没有伪造旧提交，也没有推送 GitLab。

## 回退点与范围

- `checkpoint/original-upload-20261013`：原始上传代码（根提交 `b199781`），另有 `../backups/original-upload-20261013.tar.gz`。
- `checkpoint/25d-restored-20261013`：只恢复原生 ODX 2.5D 参数、输入契约与相关测试。
- `checkpoint/25d-vehicle-restored-20261013`：在2.5D上恢复车辆检测、整车选图、保守时间关联、联合选图、接受版道路整片选图与受验证的全幅导出。

如需在 VS Code 回到仅2.5D，先保存当前工作，然后运行 `git switch --detach checkpoint/25d-restored-20261013`。该操作不删除实验输出或改写分支历史。继续恢复版可 `git switch feature/by-dev`。

没有恢复 MP-SfM、单目先验、Murre、OpenMVS改造、深度约束纹理或物理网格中间实验；没有沿用旧场景的坐标框、车辆标签、掩膜、相机与点云缓存。

## 文件与修改入口

| 文件 | 作用 |
| --- | --- |
| `app/model_engine/odx.py` | `orthophotoMode=2.5d` 时不传 `--use-3dmesh`，ODX建立2.5D俯视表面及nadir纹理。保留原完整3D输出契约与原3D模式兼容性。 |
| `app/core/contracts/photogrammetry_odx.py`及同名schema | 验证参数、资源预算、输入身份；明确2.5D选项。 |
| `app/model_engine/ogs2.py` | 由重建相机、去畸变照片、2.5D表面在同一地图网格重投影；GroundingDINO+SAM2识别车辆，整车统一来源，遮挡和完整性检查，图块间约束选图。 |
| `ogs2_visibility.py` | 与参考实现一致的编译射线可见性检查。 |
| `ogs2_temporal.py`、`ogs2_deghost.py` | 同一时钟域内的短时观测关联，选择真实整车来源，无法验证时保留基线。 |
| `ogs2_joint*.py` | 完整前景闭包、独立静态几何验证、整片来源选择及静态边缘混合。 |
| `ogs2_road_coherent.py` | 恢复已接受道路整片选图路径。需要新场景的新道路、车辆和投影证据，不能直接使用旧公交标签。 |
| `ogs2_apply_patch.py` | 在全图准确窗口导出通过质量门的补丁；网格、alpha、窗口之外像素不变；保护已有输出。 |
| `app/experiments/tools/collect_*`、`inspect_joint_static.py`、`validate_joint_lab.py` | 采集真实照片和已有正射的车辆/道路证据及独立检查。 |
| `scripts/lab/audit_aerial_metadata.py` | 读取任意新数据集的EXIF/DJI元数据，报告尺寸、GPS、镜头、畸变信息和时钟域。 |
| `scripts/lab/fix_projected_camera_poses.py`、`selfcheck_projected_camera_fix.py` | 修复本次实际发现的原生坐标导出姿态序列化问题；先另存候选、全量验证，再备份应用，不修改SfM算法或去畸变照片。 |
| `scripts/lab/check_vehicle_output.py` | 分块核对网格、alpha、来源编号与保留基线像素，重算真实改变像素数；不把像素变化当作去重成功。 |
| `scripts/lab/validate_projected_sfm.py` | 自动读真实UTM与偏移，检查完整SfM残差、修正经证明的导出姿态问题，备份并重新导出NVM；在稠密之前发布绑定文件SHA的关卡。 |
| `docker/ogs2/requirements.txt` | 车辆算法独立依赖；Torch、固定SAM2源码、权重另有版本记录。 |
| `tests/test_ogs2*`、`test_joint*`、`test_road_full_export_blocks.py` | 覆盖相机投影、单源整车、拒绝错误补丁、静态边缘、全幅导出及跨块窗口。 |

## 新数据与复现原则

本次照片为405张DJI M4E、5280×3956。必须从 SfM、稠密重建开始。原图 `DewarpFlag=0`，使用ODX默认Brown自校准与一次去畸变；不冻结旧内参。经度范围跨UTM49/50，全部照片使用同一个投影坐标系，最终以实际ODX输出核对。

DJI文件名/EXIF是本次飞行的统一墙上时间。`UTCAtExposure`字段与GPS时标需要区别，不跨时钟域计算车辆运动间隔。RTK侧文件保留为原始证据，没有冒充GCP输入。

依赖只安装在 `../runtime` 持久化目录。CPU节点下载到共享盘，GPU节点运行。公共镜像逐层校验SHA并与指定源码构建定义核对，再覆盖ODX固定主源码 `eb9fd56ac430fc3735401d88aaef9945ebd00e14`（3.8.3）。这不是原私有Harbor镜像的字节级恢复；runtime保存新的镜像和源码身份。车辆环境独立使用Torch2.6/cu124、SAM2源码 `2b90b9f5ceec907a1c18123530e92e794ad901a4`、GroundingDINO权重版本 `a2bb814dd30d776dcf7e30523b00659f4f141c71`。

OGS2、短时关联和整片道路路径的旧结果只证明旧场景已检查局部的收益，不能证明405张新照片全幅车辆去重已解决。恢复源码不等于恢复旧实验结果。运行后必须分别交付原生2.5D基线、车辆候选和实际检查报告，未通过的补丁不进入正式输出。

完整源码恢复证据见 `vehicle-source-provenance.json`；旧Git对象丢失，coherent源码来自改严格guard前的完整记录，关键OGS2、visibility和接受版export有旧运行SHA证据。当前export另保留更安全的整数窗口检查。

## 本次原生坐标导出修复

405张全部定位，原生匹配图有274/131两块，ODX按共同GPS参考拼入单个JSON；这不是视觉连接成功，跨块相对配准仍需在真实正射检查。

原生OpenSfM把ENU→UTM的非正交矩阵直接放进相机旋转。内存态可保持投影，但序列化为轴角再读入后，相机中心平均漂移约0.194米，1831810条完整观测的中位残差由1.039增至7.571像素。不能用上游排除大于4像素后的1.124像素均值放行。

本次使用实际UTM50输出与偏移191706/2494789，选择XY共形的统一尺度转换（scale=1.000774447），全部相机与点采用同一个变换。547369点的地图XY最大变化约0.00000384米，所有照片完整残差统计与原ENU逐张差异小于1e-7像素；相机中心在JSON往返后保持到浮点精度。原有内参、照片、特征、匹配、tracks和观测属性保留，不引入单目或MP-SfM。

该转换也作用于Z：相机高度最大调整约0.079765米，点高度变化绝对P99约0.026731米；60个原有极端负高程稀疏点变化超过0.25米，最大约1.138米。它们没有被隐藏或删除。此处采用“相机高度≤0.25米、统一尺度修正≤0.001、地图XY≤0.005米”的显式工程筛选，不宣称全部高程误差≤0.25米，也不宣称DEM的独立测绘精度改善。原ENU是高度参考。

候选、诊断、完整残差、验收证明及修正前root/undistorted/NVM位于JOB中；备份目录为 `checkpoints/pre-geocoords-camera-fix`。修正应用后重新导出NVM；MVS场景在修正后首次生成。6项原生几何检查全通过，无跳过。全幅车辆像素检查另用6个真实GeoTIFF有效/反例用例通过。

## 复现命令

以下路径中的新JOB名必须未运行过，输入数据须放在同一工作区datasets下。元数据审计输出不覆盖已有文件。

CPU节点准备依赖（已安装的本次运行不需重装）：

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
bash scripts/lab/setup_vehicle_cpu.sh
python3 scripts/lab/fetch_odx_runtime.py --runtime ../runtime --kind gpu
```

GPU节点验证原生引擎与CUDA驱动：

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
CUDA_VISIBLE_DEVICES=0 python3 scripts/lab/smoke_odx_runtime.py --kind gpu --native
```

当前已完成的模型真实推理证明保存为 `../runtime/vehicle-model-smoke.json`。在更换机器/驱动/模型后，应重新进行真实模型推理验证，不能用旧文件冒充新硬件验证。

对新数据生成正射：

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
PY=../runtime/vehicle-env/bin/python
DATA=../datasets/DJI_202610101454_077_1010红树林
JOB=../runs/mangrove-077-new-attempt
mkdir -p "$JOB"
"$PY" scripts/lab/audit_aerial_metadata.py --images "$DATA" --output "$JOB/metadata-audit.json"
"$PY" scripts/lab/run_orthophoto_lab.py --dataset "$DATA" --job "$JOB" \
  --gpu 0 --cpu-cores 16 --memory-gib 128 --pc-quality high --resolution-cm 3 --detach
```

参数`pc-quality high`是本次新数据集首个完整运行的显式配置；它不等于旧场景ultra稠密缓存。两版正射使用同一新场景相机、稠密表面和像素网格，车辆优化与原生2.5D比较，不能直接与不同日期的旧大疆结果作同场景数值对照。

最新离线入口默认加入`--skip-3dmodel`，在定位后自动执行投影相机验收，输出正射。内部稠密点云和2.5D地表仍必要；车辆模块直接用2.5D表面。首次完成结果、真实效果与本轮旧命令记录见`first-orthophoto-run-20261013.md`。

运行状态在JOB的 `pipeline-status.json`；SfM完成后生成 `sfm-check.json`，核对单个重建JSON容器、至少95%照片成功定位。单个JSON不代表视觉匹配图连通：ODX可能将独立分块按GPS坐标拼入同一容器。出现这种原生合并时，稠密阶段还需同一几何SHA绑定的 `sfm-merge-quality-gate.json`；完整未截尾残差、匹配图及合并限制必须先核查。各阶段日志为 `sfm.log`、`odx-25d.log`、`vehicle-ogs2.log`。输出为：

- 基线：`datasets/project/odm_orthophoto/odm_orthophoto.tif`
- 车辆整车选图：`vehicle-ogs2/ogs2_orthophoto.tif`
- 模块实际参数和限制：`vehicle-ogs2/report.json`
- Git、镜像、固定源码、输入审计及实际命令：`run-plan.json`

完成后的真实网格和有效区域检查/预览：

```bash
"$PY" scripts/lab/preview_orthophotos.py \
  --baseline "$JOB/datasets/project/odm_orthophoto/odm_orthophoto.tif" \
  --candidate "$JOB/vehicle-ogs2/ogs2_orthophoto.tif" --output "$JOB/review"
```

有界去重/车影候选需要新deghost缓存，OGS2全幅不能直接当joint缓存。`ogs2_deghost --scan-only`也不够，完整执行才有report与局部栅格。`collect_mosaic_vehicles --cache NEW_CACHE`已移除旧三个场景目录的默认值，必须明确输入新缓存，采集240像素上下文。只有通过真实来源、完整前景及接缝验收后才使用全幅导出模块；旧FT-070人工公交身份和mask不能授权新数据写入。
