# 道路整段选图与闭合接缝试验，2026-10-14

## 实际结果

局部道路平面投影明显改善原正射的车形残影、断裂车身和波浪标线。西段、中段已分别输出真实照片与旧图组合的局部 GeoTIFF；不能称为已通过的全幅正射。检查 `closed-patch-trial-02` 的实际切边发现：道路内部改善，但贴回旧图的位置仍有标线跳变、旧车残留和路旁高物体错位。因此没有替换当前正式 TIFF。

相同照片、相同相机、不同投影表面的控制对照为 `centre-same-photo-geometry-comparison.jpg`。照片是 DJI_20261010151832_0040。这比用车辆数量比较不同拍摄时刻更能解释高度表面造成的畸变。平面只适用道路；树木、灯杆、车顶不能据此宣称拥有真实三维形状或高精度坐标。

新增车道标线三角化后，西段 `multiview-fit-02` 留出高度 P90 为 0.1141 m，原照片重投影 P90 为 2.3061 px。中段 `multiview-fit-01` 留出高度 P90 为 0.1149 m，重投影 P90 为 2.3002 px。这些是预设局部试验阈值下的几何结果，语义尚为候选，而且道路中部的观测分布不足；不能等同于整个道路已验证。

有界配准的西段 0→2 照片对通过默认留出检查：P90 从 1.1883 px 降至 0.7277 px，最大留出误差 1.4442 px，最大采样位置修正 0.9384 px。其余已测试照片对未全部通过，不强行沿用。

## 代码入口

- `app/model_engine/ogs2_road_registration.py`：固定地理栅格上的源采样位置微调。平移/仿射选型使用训练内部留出，最终完整外层留出参与检查。记录 UV 修正，不可复用旧的 noWarp 证据。
- `app/model_engine/ogs2_road_patch.py`：两幅图之间的闭合硬图割。车辆/阴影禁止切边；完整输出有效覆盖、硬核心、整数容量、实际切边与哈希均复核。可行仅代表图割求解成功，不证明语义或几何正确。
- `scripts/lab/road_coherent_patch_trial.py`：读取独立试验输入，输出局部 GeoTIFF、来源索引、切缝图与 JSON。旧图破碎残影的自动检测不可靠，保存为不确定证据；当前硬保护来自源照片实例，不把旧图误识别的树带强制锁成车辆。

车辆保护改成 SAM 实例与有限局部暗影搜索，扩大照片范围避免裁剪边缘把完整车体截断。巨框、低置信度和道路外提议保留在 JSON；这仍需实图复核，不能把漏检位置标成已证实空道路。

## 试验和重现

工作区：`/volume/wzhang/mesh-research-workspace`。

试验根目录：`runs/mangrove-077-20261013/road-surface-20261013`。

- `registration-trial-01`：所有配准接受/拒绝记录。
- `closed-patch-trial-01`：小裁剪范围造成对象保护冲突的拒绝证据。
- `expanded-patch-inputs-02`：扩大照片范围、实例与阴影掩膜、输入元数据。`expanded-patch-inputs-01` 是 JSON 序列化失败保留的未完成试验。
- `closed-patch-trial-02`：西段、中段局部 GeoTIFF、切边和单源像素来源图。切缝平均 RGB 差分别约 15.77、16.78，最大值约 139.67、150.33；优化可行不等于接缝不可见。
- `road-stage2-tests.log`：62 项测试通过。数字不能当作影像优化比例。

重现局部闭合补丁（输出目录必须新建）：

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
export PYTHONPATH="$PWD/app"
export LD_LIBRARY_PATH=/volume/wzhang/mesh-research-workspace/runtime/vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
/volume/wzhang/mesh-research-workspace/runtime/vehicle-env/bin/python scripts/lab/road_coherent_patch_trial.py \
  --inputs /volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013/road-surface-20261013/expanded-patch-inputs-02 \
  --output /volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013/road-surface-20261013/closed-patch-reproduce-new
```

下一阶段从清晰原照片间建立道路背景，不再让修复内部匹配已损坏的旧道路。保留同照片的几何控制对照；任何动态车身移除都必须来自实际照片中看到的路面，并检查整车和阴影覆盖。不能用生成式补画当成测量到的背景。

## 回退

本阶段之前：`checkpoint/road-surface-seam-core-20261013`（d413eb1）。道路实验之前：`checkpoint/before-road-surface-20261013`（1cbe2a7）。整轮车辆实验之前：`checkpoint/before-corridor-source-20261013`（604aabe）。

```bash
git switch --detach checkpoint/road-surface-seam-core-20261013
# 返回开发状态
git switch feature/by-dev
```

代码只在 feature/by-dev 提交；没有推送 GitLab或合并主分支。原正式正射保留。
