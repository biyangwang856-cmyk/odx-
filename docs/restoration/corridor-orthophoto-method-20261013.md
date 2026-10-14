# 高速车辆残影：真实照片来源约束实验

本轮目标是减少动态车辆重复、半车与残影，只输出正射影像。现有原生2.5D与OGS2完整TIFF始终保留，新候选使用独立目录。当前实验尚未完成验收，不能把代码提交或分割高分当作成图优化成功。

## 方法与修改入口

| 文件 | 作用及修改入口 |
|---|---|
| `app/model_engine/ogs2.py` | 现有OGS2基线；原始照片车辆检测、SAM车体分割、表面投影及来源索引。 |
| `app/model_engine/ogs2_corridor_projection.py` | 从全部SfM相机按区域覆盖率选照片，使用原有2.5D表面、相机和坐标，保存真实RGB、四像素有效支持、遮挡与原照片上下文。输入/相机/缓存有SHA256绑定。 |
| `app/model_engine/ogs2_corridor_observations.py` | 单点SAM道路候选、照片间曝光比较、实际近车阴影变化、独立两张照片的空路候选。输出只是候选；检测不到车不等于证明空路。当前实际照片验证发现密集负提示抑制道路候选，因此默认每次只使用一个正提示。 |
| `app/model_engine/ogs2_corridor_static.py` | GFTT角点与双向NCC测量低纹理路面位置差，固定空间留出并保留全部质量合格对应、异常点。搜索范围独立于验收门限。不得根据哪个提供者通过而自动切换；SIFT同时保留审计。 |
| `app/model_engine/ogs2_corridor_policy.py` | 一个道路冲突区采用同一真实曝光，完整对象闭包、旧位置真实空路、未知/停车保护及当前基线接缝检查。整车上下文保护不是实测阴影，也不是空路证据。 |
| `app/model_engine/ogs2_corridor_apply.py` | 接受已经审查且有哈希绑定的证书；先生成局部，再可选流式生成全幅、来源索引和逐像素QC。重叠硬区以及硬区侵入别的静态验收边界，双方都拒绝。没有接受区域时不生成伪优化全幅。 |
| `scripts/lab/run_orthophoto_lab.py` | 当前照片到原生2.5D/OGS2入口，含`--skip-3dmodel`。默认不会自动执行新的corridor证书接入器。 |
| `tests/test_corridor_*.py` | 对真实来源、无效支持、对象完整性、历史车位、静态位移、留出异常、相邻边界、输入篡改与实际GeoTIFF输出的反例。 |

## 本航次实验与证据

持久化根目录：`/volume/wzhang/mesh-research-workspace`。

作业：`runs/mangrove-077-20261013`。实验：`vehicle-corridor-20261013`。

* `*-evidence-01/manifest.json`：西/中/东各10个真实照片的完整投影输入。整个ROI仅用于候选覆盖排名，不是道路写入授权。
* `*-observations-01/`：密集负提示版本的原始拒绝记录。中段道路候选为零，不能据此断言所有道路几何不行。
* `road-sam-probe-02/`与`sam-prompt-mode-probe-01/`：相同原图、相同提示位置的输入与SAM掩膜/评分。明确保留失败模式与正常单点模式。
* `*-observations-02/`：单点版本，源到源测量、完整对应及留出/异常身份均保留。
* `*-baseline-review-01/`：当前正射、真实单照片、道路/空路/车体/阴影/未知候选的同坐标预览。未获独立空路证据的区域不能写入。
* `centre-pair-static-support-probe-01/`：静态域角色的诊断对照。主测量域提前声明为当前两张照片的车辆及可能阴影保护，加本次人工审查的高速车道测试域和中性色像素。其它域只作对照，不自动挑选通过者。

本次人工审查的局部测试域不等于实现了通用自动道路识别。SAM评分不等于道路语义正确概率。车数变少不等于改善；还要核对实际空路照片、完整车身、道路位置和接缝。

当前验收保持真实未修正位移P90≤2px、固定空间留出残差P90≤1px、足够匹配/空间分布/一致率。像元约3cm，不代表绝对地理精度为3cm。本实验不通过视觉变形修补来隐瞒错位。

## 接入与复现

在实验室服务器使用已有`runtime/vehicle-env`，依赖与大文件无需下载本机：

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
export PYTHONPATH="$PWD/app"
export LD_LIBRARY_PATH=/volume/wzhang/mesh-research-workspace/runtime/vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 GDAL_CACHEMAX=256
/volume/wzhang/mesh-research-workspace/runtime/vehicle-env/bin/python -m pytest -q \
  tests/test_corridor_policy_cpu.py tests/test_corridor_projection.py \
  tests/test_corridor_apply_cpu.py tests/test_corridor_static.py tests/test_corridor_observations.py
```

只有有效证书完成审查后，才能运行：

```bash
/volume/wzhang/mesh-research-workspace/runtime/vehicle-env/bin/python \
  -m model_engine.ogs2_corridor_apply \
  --review-evidence-certificates /absolute/path/to/reviewed-certificates.json \
  --output /volume/wzhang/mesh-research-workspace/runs/NEW_RESULT_DIRECTORY \
  --full-export
```

不加`--full-export`仅审查局部。所有输出目录必须新建，不能覆盖现有TIFF。当前三地区实验运行脚本保存在持久化`runtime/`中，对应实验计划记录代码提交及模型SHA；运行同一脚本须改成新的尝试目录，以保留旧证据。

## 回退

实验前的代码、标签和Git bundle均已保留。没有推送GitLab，没有合并主分支。

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
git switch --detach checkpoint/before-corridor-source-20261013
# 恢复当前实验分支
git switch feature/by-dev
```

回退代码不会删除`runs/`中的基线或新实验。原回退点是`604aabe`，bundle为`backups/before-corridor-source-20261013.bundle`。阶段标签另有`checkpoint/corridor-core-20261013`、`checkpoint/corridor-evidence-acceptance-20261013`与`checkpoint/corridor-road-prompts-20261013`。
