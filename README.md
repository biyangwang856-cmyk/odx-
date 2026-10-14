# ODX 2.5D 道路正射优化

本仓库保存当前已验证的航拍正射流程：固定 ODX 3.8.3 源码版本完成 SfM、稠密重建、2.5D 表面和基础正射；随后使用预训练的 Grounding DINO Tiny 与 SAM 2.1 Tiny 识别车辆，在原照片、相机投影和道路表面都有证据时进行整片单源选图、接缝保护与地理坐标保持。

当前公开快照对应实验室提交 `8b2dd85124a6eba741727146e50ee94f66123811`。它保留原始 GeoTIFF 的 CRS、仿射变换和输出像元网格。它不生成不存在的路面纹理，也不把多张照片透明叠加成车辆。无法找到可信无车背景时会保留原始正射。

## 先读结论

- 模型已经训练好，运行者不需要重新训练。
- 两个预训练权重约 846 MB，不提交到普通 Git 历史。`scripts/lab/download_vehicle_models.py` 会从固定官方版本下载并校验 SHA-256。
- ODX 运行时也不提交到 Git。`scripts/lab/fetch_odx_runtime.py` 固定下载 WebODM/ODX 源码提交 `eb9fd56ac430fc3735401d88aaef9945ebd00e14`，版本为 3.8.3，并校验容器层。
- 基础 2.5D + 车辆投影流程可以对新照片集运行；全线道路替换仍需要依据新场景生成道路范围与几何报告。旧场景坐标不能直接复用。
- 2026-10-15 的 FT-070 候选结果改善了大量半车、纹理碎片和标线断裂，但西段色调接缝、阴影暗斑和 23/34 个未通过的几何分段仍被明确保留为限制，不能宣称所有车辆残影已经消除。

## 推荐目录

以下命令假设 Linux GPU 服务器和一个持久化目录；可将 `ROOT` 换成自己的绝对路径。实验室共享服务器可直接使用 `/volume/wzhang/mesh-research-workspace`。

```bash
export ROOT=/volume/wzhang/mesh-research-workspace
git clone https://github.com/biyangwang856-cmyk/odx-.git "$ROOT/mesh-research"
cd "$ROOT/mesh-research"
git rev-parse HEAD
```

照片放入独立数据集目录，运行结果放入独立任务目录：

```text
$ROOT/datasets/<dataset-id>/photos/*.JPG
$ROOT/runs/<run-id>/
$ROOT/runtime/
```

不要把照片、模型权重或任务结果提交到 Git。仓库的 `.gitignore` 已忽略 `data/`、`outputs/`、`results/` 和日志；共享盘目录仍应放在仓库外。

## 权重：无需训练

实验室共享盘已有已验证权重时，可直接复用：

```text
$ROOT/runtime/vehicle-models/grounding-dino-tiny/model.safetensors
$ROOT/runtime/vehicle-models/sam2.1_hiera_tiny.pt
```

没有权重时，在能访问互联网的 CPU 节点运行：

```bash
cd "$ROOT/mesh-research"
python3 scripts/lab/download_vehicle_models.py \
  --root "$ROOT/runtime/vehicle-models"
sha256sum \
  "$ROOT/runtime/vehicle-models/grounding-dino-tiny/model.safetensors" \
  "$ROOT/runtime/vehicle-models/sam2.1_hiera_tiny.pt"
```

预期摘要：

```text
1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3  model.safetensors
7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69  sam2.1_hiera_tiny.pt
```

Grounding DINO 和 SAM 2 的官方实现与权重均使用 Apache-2.0；仍应保留各自上游仓库的许可证和模型说明。这里提供的是获取与校验脚本，不把 846 MB 权重复制进 Git 历史。

## ODX 3.8.3 运行时

在能访问 Docker Hub 和 GitHub 的 CPU 节点准备运行时：

```bash
cd "$ROOT/mesh-research"
python3 scripts/lab/fetch_odx_runtime.py \
  --runtime "$ROOT/runtime" \
  --kind gpu
```

按帮助信息把运行时写入 `$ROOT/runtime`。GPU 节点只需要读取同一共享目录。脚本固定并记录源码提交、镜像清单、层摘要和解包工具；不要用仅有 `3.8.3` 标签但来源不明的镜像替换。

## 新数据集基础运行

先做元数据审计：

```bash
cd "$ROOT/mesh-research"
export PYTHONPATH="$PWD/app"
python3 scripts/lab/audit_aerial_metadata.py \
  --images "$ROOT/datasets/<dataset-id>/photos" \
  --output "$ROOT/runs/<run-id>/metadata-audit.json"
```

确认没有审计错误后，在 GPU 节点运行 2.5D + 车辆优化：

```bash
python3 scripts/lab/run_orthophoto_lab.py \
  --workspace "$ROOT" \
  --dataset "$ROOT/datasets/<dataset-id>/photos" \
  --job "$ROOT/runs/<run-id>" \
  --gpu 0 \
  --cpu-cores 16 \
  --memory-gib 128 \
  --wall-hours 36 \
  --pc-quality high \
  --resolution-cm 3
```

该入口按顺序运行 SfM、相机/坐标检查、ODX 2.5D 正射和基础车辆投影，并把状态写入 `pipeline-status.json`。它要求照片注册率至少 95%，否则会停止而不是继续生成看似完整但几何不可信的正射。

基础输出：

```text
$ROOT/runs/<run-id>/datasets/project/odm_orthophoto/odm_orthophoto.tif
$ROOT/runs/<run-id>/vehicle-ogs2/ogs2_orthophoto.tif
$ROOT/runs/<run-id>/pipeline-status.json
```

## 全线道路替换

`scripts/lab/highway_full_*.py` 和 `app/model_engine/ogs2_highway_*.py` 是当前全线道路候选实现。算法核心包括：

1. 从真实原照片及已解算相机采样道路候选；
2. 用多视几何估计道路高度，并把训练点和留出点分开；
3. 使用预训练检测/分割权重识别车辆与保护边界；
4. 对一个连通车辆/前景群组整体保留或整体替换，避免在车身中间切缝；
5. 无可信无车背景时回退到原始 2.5D 正射；
6. 导出时逐块验证替换范围外像素为零变化，并保持 CRS、仿射变换、尺寸与边界。

当前脚本保存了 FT-070 的路径和人工复核范围，属于可审计实验实现，不是任意数据集的一键道路识别器。换场景必须先运行 `highway_full_source_scaffold.py`，查看生成的低分辨率范围图，再为新场景记录复核后的道路范围；不能复制 FT-070 的种子点。完整阶段、质量门和已知失败见 [全线道路流程说明](docs/lab/full-highway-orthophoto-20261015.md)。

## 测试

核心道路选择与接缝测试：

```bash
export PYTHONPATH="$PWD/app"
python -m pytest -q \
  tests/test_road_patch.py \
  tests/test_highway_background.py \
  tests/test_highway_join.py \
  tests/test_joint_closure_cpu.py \
  tests/test_joint_cohort_cpu.py \
  tests/test_joint_geometry.py \
  tests/test_joint_policy_cpu.py \
  tests/test_joint_uncertainty.py
```

仓库快照中上述相关测试已通过。完整项目测试还依赖 VTK、Typer、GDAL、CUDA 和 ODX 原生运行时；缺少这些依赖时不能把“无法收集测试”误报为算法测试通过。

## 上传与更新步骤

Windows 首次发布的逐步命令见 [上传到 GitHub 与师兄复现步骤](UPLOAD_TO_GITHUB.zh-CN.md)。

首次仓库已经由当前快照初始化。后续从实验室服务器更新时，建议先形成一个只含源码的干净提交，再推送：

```bash
cd "$ROOT/mesh-research"
git status --short
git log -1 --oneline
git remote remove github 2>/dev/null || true
git remote add github https://github.com/biyangwang856-cmyk/odx-.git
git push github HEAD:main
```

若服务器没有 GitHub 凭据，不要把个人访问令牌写入脚本或仓库。可在本机克隆 GitHub 仓库，添加经检查的源码提交后推送，或使用 GitHub App。每次上传前至少检查：

```bash
git status --short
git grep -InE 'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|ghp_|glpat-' -- . || true
git ls-files | grep -E '\.(JPG|TIF|pt|safetensors)$' && exit 1 || true
```

本公开快照排除了实验室内部 GitLab CI、发布代理配置和生成的运行数据。仓库当前没有为本项目新增许可证；公开可见不自动等于任意再许可。需要对外分发或商用前，应由项目负责人明确本项目源码许可证，并同时遵守 ODX、Grounding DINO、SAM 2 及其他依赖许可证。
