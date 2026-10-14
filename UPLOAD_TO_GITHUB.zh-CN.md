# 上传到 GitHub 与师兄复现步骤

目标仓库：<https://github.com/biyangwang856-cmyk/odx->。2026-10-15 核实时该仓库公开且为空。

## 当前已经准备好的目录

本机目录已完成筛选、检查和首次提交。你现在只需执行：

```powershell
cd "C:\Users\18906\Documents\Codex\2026-09-30\wo-b\github-publication\odx-current"
git status --short
git log -1 --oneline
git push -u origin main
```

`git status --short` 中显示的未跟踪文件正是被排除的内部 CI/发布文件，不会随 `git push` 上传。下面保留从原始快照重新构建公开提交的完整过程，供以后更新或换电脑时使用。

## 1. 初始化公开仓库

在 Windows PowerShell 中运行：

```powershell
cd "C:\Users\18906\Documents\Codex\2026-09-30\wo-b\github-publication\odx-current"

git init -b main
git config user.name "biyangwang856-cmyk"
git config user.email "biyangwang856@gmail.com"

git add -- `
  ".gitignore" `
  ".python-version" `
  "README.md" `
  "MODEL_WEIGHTS.sha256" `
  "UPLOAD_TO_GITHUB.zh-CN.md" `
  "app" `
  "browser" `
  "configs" `
  "docker" `
  "docs/lab" `
  "docs/research" `
  "docs/restoration" `
  "justfile" `
  "lefthook.yml" `
  "pixi.lock" `
  "pixi.toml" `
  "pytest.ini" `
  "ruff.toml" `
  "scripts/lab" `
  "scripts/quality" `
  "services" `
  "tests"

# 从首次提交的暂存区移除内部服务器配置和工具调用历史。
git rm --cached -- `
  "docs/restoration/vehicle-source-provenance.json" `
  "tests/test_release_contract.py" `
  "tests/test_worker_image_context.py"
git rm -r --cached -- "configs/server"
```

`git rm --cached` 只移出暂存区，不删除本机文件。

## 2. 上传前检查

```powershell
# 不应输出照片、GeoTIFF 或权重。
git diff --cached --name-only |
  Select-String -Pattern '\.(JPG|JPEG|TIF|TIFF|pt|safetensors)$'

# 不应输出密钥或令牌。
git grep --cached -n -E `
  'BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|ghp_[A-Za-z0-9]+|glpat-[A-Za-z0-9_-]+' -- .

git status --short
git diff --cached --stat
```

前两项必须没有匹配。不要把照片、模型权重、SSH 私钥、GitHub Token、运行结果或内部 CI/发布配置上传到公开仓库。

## 3. 提交并推送

```powershell
git commit -m "发布 ODX 2.5D 道路正射优化算法"
git remote add origin "https://github.com/biyangwang856-cmyk/odx-.git"
git push -u origin main
```

首次推送时，Git Credential Manager 可能要求在浏览器登录 GitHub。不要把密码或 Token 写入脚本。目标仓库当前为空，无需也不应使用 `--force`。

推送后打开目标仓库，检查 `README.md`、`app/model_engine/ogs2_highway_background.py`、`app/model_engine/ogs2_highway_join.py`、`scripts/lab/run_orthophoto_lab.py` 与测试文件均可见；确认仓库中没有 `.JPG`、`.tif`、`.pt` 或 `.safetensors`。

## 4. 师兄在服务器拉取

```bash
export ROOT=/volume/wzhang/mesh-research-workspace
git clone https://github.com/biyangwang856-cmyk/odx-.git "$ROOT/mesh-research-github"
cd "$ROOT/mesh-research-github"
git log -1 --oneline
```

在同一实验室共享盘上，直接复用已验证的运行时和权重：

```text
/volume/wzhang/mesh-research-workspace/runtime/odx-runtime-gpu
/volume/wzhang/mesh-research-workspace/runtime/vehicle-env
/volume/wzhang/mesh-research-workspace/runtime/vehicle-models
```

这不需要重新训练。

## 5. 新服务器下载预训练权重

在可访问互联网的 CPU 节点执行：

```bash
export ROOT=/自己的持久化目录
export REPO="$ROOT/mesh-research-github"
python3 "$REPO/scripts/lab/download_vehicle_models.py" \
  --root "$ROOT/runtime/vehicle-models"

cd "$ROOT/runtime/vehicle-models"
sha256sum -c "$REPO/MODEL_WEIGHTS.sha256"
```

预期两个文件均显示 `OK`。这一步下载官方预训练权重，不训练模型。

## 6. 新服务器准备 ODX 3.8.3

```bash
python3 "$REPO/scripts/lab/fetch_odx_runtime.py" \
  --runtime "$ROOT/runtime" \
  --kind gpu
```

脚本固定 ODX 源码提交 `eb9fd56ac430fc3735401d88aaef9945ebd00e14` 并校验容器层。CPU 节点负责下载到共享盘，GPU 节点从共享盘运行。

## 7. 新数据集基础正射

```bash
export DATASET="$ROOT/datasets/<dataset-id>/photos"
export JOB="$ROOT/runs/<run-id>"
cd "$REPO"
export PYTHONPATH="$PWD/app"

python3 scripts/lab/audit_aerial_metadata.py \
  --images "$DATASET" \
  --output "$JOB/metadata-audit.json" \
  --sha256

python3 scripts/lab/run_orthophoto_lab.py \
  --workspace "$ROOT" \
  --dataset "$DATASET" \
  --job "$JOB" \
  --gpu 0 \
  --cpu-cores 16 \
  --memory-gib 128 \
  --wall-hours 36 \
  --pc-quality high \
  --resolution-cm 3
```

换场景后，全线道路替换不能复用 FT-070 的道路种子点。先生成新场景道路范围图并复核，再运行 `highway_full_*.py`；具体质量门和限制见 `docs/lab/full-highway-orthophoto-20261015.md`。
