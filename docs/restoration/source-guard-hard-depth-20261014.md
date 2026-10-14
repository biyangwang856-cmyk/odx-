# 源照片不确定车辆禁选与深度硬排除

继承 `e901d23`。全幅正射、原照片、SfM、原始深度文件均保留；本阶段没有宣称全局虚影已解决。

## 真实背景禁选修正

`ogs2_source_guard.py` 使用现有去畸变照片的 0/255 掩膜，以及检测已提出但 SAM 不确定的合理尺寸车辆框。后者只用于禁止背景复制，不被声明成可靠车身分割。巨大的不合理提议仍记录为未知，不整张照片误禁选；没有检测到对象也不是空路面证明。

投影使用已保存的原照片 UV，四个双线性 RGB 采样位置只要一个落入禁选区，就不能作为可用背景。原灰度掩膜、RGB、相机和地理格网不修改。`road_source_background_trial.py --source-masks ...` 为独立可复现实验入口；默认旧路径保留。

西段 `source-background-guarded-01` / `source-background-guarded-colour-01` 运行完成。原照片禁选额外保护了锚图的 5 个、另一来源的 6 个不确定提议。配准仍通过，仍只替换 1 组、保留 7 组；相较上阶段同窗口，仅 83/1,650,000 个 RGB 像素改变，没有明显新增画质收益。因此不能将更严格的保护规则报为消重进展。

## 分割重试实测

白车样例单掩膜分数约 0.715，保持 0.8 门槛。重试 0.6/1.5/3 米上下文与多掩膜，总计 12 个候选。部分分数超过 0.8，但图像复核显示旗帜、阴影也被合并；没有仅按分数将此策略全量部署。

`sam-ambiguous-body-01` 保留了诊断代码对浮点掩膜索引失败的日志；第二轮全部预测和 PNG 已生成，末尾 JSON 写入因 NumPy int64 元数据失败。随后从完整日志恢复 12 条记录，逐个核对 PNG 像素数；保留原失败日志与 source-used.py，`report.json` 明确记录恢复来源。没有伪造未完成预测。

## 深度融合前的硬约束

`enforce_vehicle_depth_cache.py` 从已完成的原生掩膜组复制 20 张 DR 深度图到独立目录。按相机/图像尺寸映射四个原照片采样位置，清零禁选位置的深度、法向与置信度；未禁选深度保持逐值一致，原始头和视图字段保留。旧文件只读。

原生 CLI 的 `fusion-mode -2` 是融合视差，不适用于此深度缓存。实际使用 mode 0，禁用几何重估与后处理插值，直接融合这组缓存。原生输出日志仅显示融合，完成后全部硬排除缓存 SHA-256 保持一致，全部原深度文件 SHA-256 也保持一致。

`strict-depth-cache-01` 实测清零 **71,512 个仍为正值的深度采样**，20 张缓存的禁选位置正深度为 0，原生融合成功生成 18,934,043 点。上一阶段为 18,949,206 点。采样包含更严格的边界支持条件；不能说 71,512 全部是填洞重引入，也不能将点数下降当作正射改善。具体残留来源尚未逐项证明。

严格深度约束保证已禁选位置不在该融合步骤重新出现；它不能发现漏检车辆、识别跨照片同车身份，也不能单独解决纹理混合或接口跳变。下一阶段需要验证更干净的几何是否改善路面高度与真实照片投影。

## 运行位置与复现

```bash
ROOT=/volume/wzhang/mesh-research-workspace
JOB=$ROOT/runs/mangrove-077-20261013
cd $ROOT/mesh-research
export PYTHONPATH=$PWD/app
export LD_LIBRARY_PATH=$ROOT/runtime/vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=$ROOT/runtime/vehicle-env/bin/python
$PY scripts/lab/road_source_background_trial.py --workspace $ROOT --job $JOB \
  --surface-dir $JOB/road-surface-20261013 --regions west \
  --source-masks $JOB/vehicle-source-masks-20261014/raw-mask-pilot-03 \
  --output $JOB/road-surface-20261013/guard-background-new
$PY scripts/lab/enforce_vehicle_depth_cache.py --workspace $ROOT \
  --trial $JOB/vehicle-source-masks-20261014/geometry-ab-02 \
  --output $JOB/vehicle-source-masks-20261014/hard-depth-new
```

输出路径必须不存在。主要文件：`ogs2_source_guard.py`，背景试验脚本，深度硬排除脚本及 DR 只读解析器；测试覆盖不确定白车、双线性边界、坐标错误、非法标签、原字节保留、空掩膜和幂等性。

回退到本轮前：`git switch --detach checkpoint/native-vehicle-mask-verified-20261014`。返回 `git switch feature/by-dev`。此前 D 盘 e901d23 仓库快照保留。本阶段不推送、不合并主分支。
