# ruff: noqa: E501
# Long lines below are literal HTML/CSS and Markdown output, not Python logic.
import argparse
import html
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()
d = a.directory
q = json.loads((d / "quality-report.json").read_text())
m = json.loads((d / "manifest.json").read_text())
locals = json.loads((d / "local-comparison-windows.json").read_text())
label = {
    "west-join": "西段接缝",
    "west-middle": "西段路面",
    "centre-west": "中段西侧",
    "centre-east": "中段东侧",
    "east-middle": "东段路面",
    "east-end": "东端",
}


def slider(title, before, after):
    return f'''<article><h2>{html.escape(title)}</h2><div class="compare"><img src="{before}" alt="改善前"><img class="after" src="{after}" alt="改善后"><span class="split"></span><b class="before-label">改善前</b><b class="after-label">改善后</b></div><input type="range" min="0" max="100" value="50" aria-label="{html.escape(title)}改善前后对比"><p class="hint">拖动滑块：左侧为改善后，右侧为改善前。两张图使用相同位置和比例。</p></article>'''


body = slider("整条主路：上下行车道", "before-highway-overview.jpg", "after-highway-overview.jpg")
for v in locals:
    body += slider(label[v["label"]], v["label"] + "-before.jpg", v["label"] + "-after.jpg")
e = q["export"]
failed = len(q["geometryFailedSegments"])
passed = 34 - failed
page = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>整条主路正射替换对比 · 2026-10-15</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#eef2f5;color:#17212b;font:16px/1.65 "Microsoft YaHei",sans-serif}}main{{max-width:1500px;margin:auto;padding:28px}}header,article,.card{{background:white;border-radius:14px;padding:22px;margin-bottom:24px}}h1{{font-size:28px;margin:0 0 12px}}h2{{font-size:21px;margin:0 0 12px}}.status{{background:#fff1d8;border-left:4px solid #c47c0d;padding:12px 16px}}.stats{{display:flex;flex-wrap:wrap;gap:14px;margin-top:16px}}.stats div{{padding:10px 18px;background:#f1f6f9;border-radius:8px}}.stats b{{display:block;font-size:23px}}.compare{{position:relative;width:100%;overflow:hidden}}.compare img{{display:block;width:100%;height:auto}}.compare .after{{position:absolute;top:0;left:0;clip-path:inset(0 50% 0 0)}}.split{{position:absolute;left:50%;top:0;bottom:0;border-left:2px solid white;box-shadow:0 0 3px black}}.before-label,.after-label{{position:absolute;top:8px;background:#17212bbf;color:white;padding:3px 10px;border-radius:5px}}.after-label{{left:8px}}.before-label{{right:8px}}input{{width:100%;margin:14px 0 0;accent-color:#207ca4}}.hint{{color:#607080;font-size:13px;margin:4px 0 0}}a{{color:#0b668e}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;padding:8px;border-bottom:1px solid #e2e7eb}}code{{word-break:break-all}}.full{{max-width:100%;height:auto}}@media(max-width:650px){{main{{padding:10px}}header,article,.card{{padding:12px}}h1{{font-size:23px}}}}
</style></head><body><main><header><h1>整条主路正射替换结果</h1><p>红树林 077 场景 · 2026-10-15 · 主路上下行车道。上方平行辅路及其他街道未纳入本轮替换。</p><p class="status">候选结果已生成。道路范围按本场景照片核对；部分几何检查和接缝色差检查尚未通过，不能把存储的坐标或 3 cm 像素间距当成测量精度证明。</p><div class="stats"><div><b>{q["roadCoreCoverage"]:.2%}</b>声明道路核心替换覆盖率</div><div><b>{e["width"] if "width" in e else e["reference"]["width"]} × {e["reference"]["height"]}</b>完整 GeoTIFF 像素</div><div><b>{e["outsideReplacementChangedPixels"]}</b>替换范围外变动像素</div><div><b>{passed} / 34</b>通过独立路面几何检查的分段</div></div><p>像素来自实际航拍照片，使用车辆保护、真实背景选取和硬接缝；不使用生成式补图。</p></header>{body}<section class="card"><h2>完整场景</h2><img class="full" src="full-scene-after.jpg" alt="完整场景正射影像"><p>网页 JPEG 仅供查看，带坐标的正式栅格文件为同目录 GeoTIFF。</p></section><section class="card"><h2>文件与复现</h2><p><a href="CANDIDATE-main-highway-orthophoto.tif">完整正射 GeoTIFF</a> · <a href="quality-report.json">质量与坐标核查</a> · <a href="README.md">流程、修改入口和回退命令</a> · <a href="manifest.json">原文件校验与结果 SHA-256</a></p><p>EPSG:32650，沿用原始仿射变换、范围和像素网格。替换没有移除坐标信息；几何误差、保留车辆及阴影仍需按局部图和报告判断。</p></section></main><script>document.querySelectorAll('article').forEach(a=>{{const i=a.querySelector('input'),im=a.querySelector('.after'),s=a.querySelector('.split');i.addEventListener('input',()=>{{im.style.clipPath=`inset(0 ${{100-i.value}}% 0 0)`;s.style.left=i.value+'%';}});}});</script></body></html>"""
(d / "index.html").write_text(page, encoding="utf8")
readme = f"""# 整条主路正射影像替换结果

生成于 2026-10-15。结果仍标为候选，不能声称所有残影和接缝已完全解决。

## 结果与核查

- 完整文件：`CANDIDATE-main-highway-orthophoto.tif`，{e["reference"]["width"]} × {e["reference"]["height"]}，EPSG:32650，约 3 cm 像素间距。
- 原坐标系、仿射变换、范围、尺寸、波段与像素间距完全一致。
- 替换范围外变动像素 {e["outsideReplacementChangedPixels"]}；替换范围内与道路候选不一致的像素 {e["insideReplacementMismatchPixels"]}。
- 声明道路核心替换覆盖率 {q["roadCoreCoverage"]:.6%}；未获得有效覆盖的道路像素 {q["uncoveredRoadPixels"]}。
- 34 个路面分段中 {passed} 个通过几何检查，{failed} 个未通过。未通过检查的模型及训练支持区外外推明确保留在报告中，混合后的连续高度场也没有独立精度认证。
- 车身保护区切割计数是遮罩上的检测结果，不等于人工确认零半车，也不等于消除比例。保留前景像素包含误检和配准差异，不能当成车辆数。
- JPEG 用于查看，不能代替带地理信息的 GeoTIFF。打开 `index.html` 查看整条道路和 6 处局部滑块。

## 方法

1. 读取既有 OpenSfM 内参、相机位置、方向、去畸变照片和地理偏移。
2. 从真实照片核对主路范围。固定的场景范围种子不是地面控制点，不能视为新场景通用自动分割。
3. 在静态路面上建立多视图匹配和三角化，按训练/独立检查划分拟合路面高度。稀疏/未通过区域作为诊断估计保留。
4. 把实际照片投到原有 3 cm 世界网格。车辆使用 GroundingDINO + SAM 遮罩；低置信度保留完整检测框。
5. 只从有效、未被车辆遮罩覆盖的真实照片选取背景。无可用背景时保留原有前景，不生成虚构路面。不跨照片做车身透明混合。
6. 整块车身优先使用同一来源；必要时以多源图割联合选取背景。接缝检查两侧车身遮罩，绕开车身。时序颜色差也可能把错位标线当成车辆：硬禁选不可行时，该类候选改作接缝代价，车身检测仍保持硬约束。
7. 在完整栅格中只替换声明区域，逐块重新读取验证像素和地理信息，原图 SHA-256 再次核对。

## 文件与修改入口

服务器根目录：`/volume/wzhang/mesh-research-workspace`。
仓库：`mesh-research`；数据与中间件：`runs/mangrove-077-20261013`。

| 文件 | 作用 / 修改入口 |
| --- | --- |
| `app/model_engine/ogs2_highway_background.py` | 时序异常候选、车身禁选、真实背景来源、单源/联合图割、色差门槛 |
| `app/model_engine/ogs2_highway_join.py` | 可横向绕行的硬接缝；检查实际切边两端的车辆/有效像素 |
| `scripts/lab/highway_full_plan.py` | 原世界网格、相机与道路运行范围规划 |
| `scripts/lab/highway_full_source_scaffold.py` | 用照片生成道路搜索底图；其临时 3 m 高度不是最终几何 |
| `scripts/lab/highway_full_reviewed_scope.py` | 本场景道路范围种子与边界；新场景必须重新核对 |
| `scripts/lab/highway_full_geometry.py` | 静态匹配、三角化、训练与独立检查；没有通过的测量不能包装为通过 |
| `scripts/lab/highway_full_render.py` | 高度场、照片投影、原始 6 cm DSM 按世界坐标采样、车身与来源缓存、分段拼接、道路边界 |
| `scripts/lab/export_full_highway_orthophoto.py` | 合成完整 GeoTIFF，逐块核验替换范围及坐标，导出局部对比和校验清单 |
| `scripts/lab/compose_full_highway_review.py` | 离线查看页面及本文档 |
| `source-photo-provenance.tif` | 道路范围内的真实照片编号；0 表示保留原图/无新来源 |
| `retained-foreground.tif` | 保留前景/异常候选；不是车辆计数或真值 |
| `road-geometry-measurements.json` | 路面三角化、训练与独立检查的全部记录 |
| `quality-report.json` | 本次拼接、覆盖、几何限制和坐标核验 |
| `manifest.json` | 原始文件不变的 SHA-256 和结果文件逐项校验 |

## 从当前保存的中间结果复现

在服务器执行；导出使用新的目录名，避免覆盖现有证据。

```bash
cd /volume/wzhang/mesh-research-workspace/mesh-research
export PYTHONPATH="$PWD/app"
export LD_LIBRARY_PATH=/volume/wzhang/mesh-research-workspace/runtime/vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
PY=/volume/wzhang/mesh-research-workspace/runtime/vehicle-env/bin/python
JOB=/volume/wzhang/mesh-research-workspace/runs/mangrove-077-20261013
"$PY" scripts/lab/export_full_highway_orthophoto.py \\
  --job "$JOB" \\
  --candidate "$JOB/highway-full-20261014-01/full-render-scale1-09" \\
  --output "$JOB/orthophoto-full-highway-replay-01"
"$PY" scripts/lab/compose_full_highway_review.py "$JOB/orthophoto-full-highway-replay-01"
"$PY" -m pytest -q tests/test_highway_background.py tests/test_highway_join.py tests/test_road_patch.py
```

从照片重新执行的完整阶段顺序、缓存规则及修复记录见仓库 `docs/lab/full-highway-orthophoto-20261015.md`。现有实验目录不能原地重跑；先修改运行目录名，在副本中运行。流程仍复用原项目 SfM 和去畸变照片，未引入中间尝试过的单目/SfM 优化。

## Git 回退

本轮前：`checkpoint/before-full-highway-20261014`。
本轮后：`checkpoint/full-highway-orthophoto-20261015`。

```bash
git switch --detach checkpoint/before-full-highway-20261014
# 查看本轮完成状态
git switch --detach checkpoint/full-highway-orthophoto-20261015
# 返回开发分支
git switch feature/by-dev
```

切换前先检查 `git status`，保留后来产生的未提交修改。未推送 GitLab，未合并主分支。原影像、中间候选和失败记录均保留。
"""
(d / "README.md").write_text(readme, encoding="utf8")
print(str(d / "index.html"))
