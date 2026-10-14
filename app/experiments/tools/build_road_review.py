"""Build an offline review of a verified road correction using actual pixels.

Run on the laboratory server. No model inference, colour normalisation,
hand alignment, image warping of vehicles, or generated image content is used.
Only the DJI reference is geographically reprojected to the common map grid.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio as rio
from affine import Affine
from PIL import Image
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def aligned_window(dataset, frame):
    if dataset.crs != frame.crs:
        raise ValueError(f"CRS differs: {dataset.name}")
    if not np.allclose(
        [dataset.transform.a, dataset.transform.b, dataset.transform.d, dataset.transform.e],
        [frame.transform.a, frame.transform.b, frame.transform.d, frame.transform.e],
        rtol=0,
        atol=1e-10,
    ):
        raise ValueError(f"Resolution or rotation differs: {dataset.name}")
    x, y = ~dataset.transform * (frame.transform.c, frame.transform.f)
    if max(abs(x - round(x)), abs(y - round(y))) > 1e-6:
        raise ValueError(f"Pixel centres differ: {dataset.name}")
    if x < 0 or y < 0 or x + frame.width > dataset.width or y + frame.height > dataset.height:
        raise ValueError(f"Review frame exceeds raster: {dataset.name}")
    return Window(round(x), round(y), frame.width, frame.height)


def rgba(dataset, window=None):
    if dataset.count < 3 or dataset.dtypes[0] != "uint8":
        raise ValueError(f"Expected actual uint8 RGB/RGBA raster: {dataset.name}")
    rgb = dataset.read([1, 2, 3], window=window).transpose(1, 2, 0)
    alpha = dataset.read(4, window=window) if dataset.count == 4 else dataset.dataset_mask(window=window)
    return np.dstack((rgb, alpha))


def bbox(mask, margin):
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("Reviewed bus evidence is empty")
    return [
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(mask.shape[1], int(xs.max()) + 1 + margin),
        min(mask.shape[0], int(ys.max()) + 1 + margin),
    ]


def save_views(output, arrays, box):
    x0, y0, x1, y1 = box
    for key, image in arrays.items():
        Image.fromarray(image[y0:y1, x0:x1]).save(output / f"bus-{key}.png")
        Image.fromarray(image).save(output / f"road-{key}.png")
        # JPEG has no transparency. Black only represents coverage outside the
        # corresponding input; it is never inserted into the exported mosaic.
        opaque = np.zeros(image.shape[:2] + (3,), np.uint8)
        visible = image[:, :, 3] > 0
        opaque[visible] = image[:, :, :3][visible]
        preview = Image.fromarray(opaque)
        preview.thumbnail((1400, 1400), Image.Resampling.LANCZOS)
        preview.save(output / f"road-{key}.jpg", quality=94, subsampling=0)


PAGE = r"""<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>公交去重与车影对照 · FT-070</title>
<style>
:root{color-scheme:dark;
font-family:system-ui,"Microsoft YaHei",sans-serif;
background:#111827;
color:#e5e7eb}
body{margin:0;
line-height:1.6}main{max-width:1500px;
margin:auto;
padding:28px 20px 70px}
h1{font-size:28px;
margin:0 0 10px}h2{font-size:20px;
margin:30px 0 10px}p{max-width:1050px}
a{color:#93c5fd}button,select{font:inherit;
background:#253248;
color:inherit;
border:1px solid #526278;
border-radius:6px;
padding:7px 13px}
button[aria-pressed="true"]{background:#1d4ed8}button,select,input{cursor:pointer}
.muted{color:#adb9cb}.note{padding:12px 16px;
border-left:3px solid #60a5fa;
background:#192337;
border-radius:3px}
.facts{display:flex;
gap:10px;
flex-wrap:wrap}.fact{background:#1b293f;
padding:9px 14px;
border-radius:7px}
.toolbar{display:flex;
align-items:center;
gap:12px;
flex-wrap:wrap;
margin:15px 0}
.compare{position:relative;
width:100%;
overflow:hidden;
background:#000;
aspect-ratio:__BUS_ASPECT__;
border:1px solid #475569}
.compare>img{position:absolute;
top:0;
left:0;
width:100%;
height:100%;
object-fit:contain;
image-rendering:auto}
.compare .new{clip-path:inset(0 50% 0 0)}.line{position:absolute;
top:0;
bottom:0;
left:50%;
width:2px;
background:white;
pointer-events:none}
.tag{position:absolute;
top:10px;
padding:4px 9px;
border-radius:5px;
background:#101724df;
font-size:14px}.old-tag{right:10px}.new-tag{left:10px}
.slider{width:100%;
margin:15px 0}.five{display:grid;
grid-template-columns:repeat(5,minmax(0,1fr));
gap:8px}
figure{margin:0;
min-width:0}figure img{display:block;
width:100%;
background:#000;
border:1px solid #475569}
figcaption{font-size:14px;
margin-bottom:5px}small{font-size:12px}.zoom{overflow:auto;
border:1px solid #475569;
background:black;
max-height:650px}
.zoom img{display:block;
width:auto;
max-width:none}.readout{font-variant-numeric:tabular-nums}
@media(max-width:850px){
.five{grid-template-columns:repeat(2,minmax(0,1fr))}
.compare{aspect-ratio:__BUS_ASPECT__}
main{padding:18px 12px}}
</style>
<main>
<h1>公交去重与车影：真实影像对照</h1>
<p>这里显示同一坐标位置的原 2.5D、旧 OGS2、道路统一拍摄时刻的新结果、
实际源照片投影及大疆正射参考。照片投影采用项目原有相机与地表，
像素未经生成、修补或调色。</p>
<div class="note">__REVIEW_CLAIM__ 本页只核验这段道路及此处公交；
没有证据表明全场车辆重复和残影都已解决。
自动分割产生的掩膜数量不能作为车辆数量。</div>
<h2>旧版与新版滑动对比</h2>
<div class="toolbar">
<button id="bus" aria-pressed="true">公交及其历史位置</button>
<button id="road" aria-pressed="false">整段道路</button>
<span class="muted">拖动滑块；左侧为新版，右侧为旧 OGS2。</span>
</div>
<div class="compare" id="compare">
<img id="old-image" src="bus-old.png" alt="旧 OGS2">
<img id="new-image" class="new" src="bus-new.png" alt="道路统一时刻新版">
<div class="line" id="line">
</div>
<span class="tag new-tag">新版</span>
<span class="tag old-tag">旧 OGS2</span>
</div>
<input class="slider" id="slider" type="range" min="0" max="100" value="50" aria-label="对比位置">
<h2>五份结果并列</h2>
<div class="five" id="five">
</div>
<p class="muted">大疆参考只按 GeoTIFF 的坐标信息重投影到本项目的 5 厘米网格，
没有人工配准、平移或旋转。它的原始分辨率和处理链不同，
细节差异不能直接归因于本次改动。</p>
<h2>原尺寸查看</h2>
<div class="toolbar">
<label for="single">选择：</label>
<select id="single">
</select>
<span class="muted">可滚动查看原始像素，点击图像在新标签页打开。</span>
</div>
<div class="zoom">
<a id="single-link" href="bus-new.png" target="_blank">
<img id="single-image" src="bus-new.png" alt="原尺寸证据">
</a>
</div>
<p class="note">源照片视域以外显示黑色或透明，是该张照片没有观察到那里；
这些像素不会写入新正射，也不代表正射新增空白。
整段道路预览会缩小，公交局部 PNG 保留原始像素。</p>
<h2>实际像素核查</h2>
<div class="facts">
<div class="fact">全幅实际改变：<b class="readout">__CHANGED__</b> 像素</div>
<div class="fact">允许写入范围外改变：<b class="readout">__OUTSIDE__</b>
</div>
<div class="fact">覆盖范围 / 透明度改变：<b class="readout">__ALPHA__</b>
</div>
</div>
<p>这些数字来自全幅输出的实际逐像素核查。数值为零的约束仅说明保存与写入边界符合记录，不能代替对车身、真实阴影、路沿和接缝的目视检查。</p>
<p>当前采用：<strong>__SOURCE__</strong>。道路内采用同一张真实照片，保留其完整车身和当时的阴影；历史重复位置采用这张照片实际观察到的道路背景。</p>
<p>车影证据来自真实影像的亮度差与人工核验区域。阴影不是独立测量真值；应查看源照片，避免把车辆当时的真实阴影误当作残影。</p>
<p>
<a href="summary.json">输入路径、SHA256、对照范围与处理说明</a> ·
<a href="full-pixel-check.json">全幅像素核查报告</a> ·
<a href="patch-pixel-check.json">局部源像素核查报告</a>
</p>
<p class="muted">下载原尺寸 PNG：<span id="downloads">
</span>。页面不连接外部服务，所有图像均是服务器上真实文件的裁图。</p>
</main>
<script>
const names=[['native','原 2.5D'],['old','旧 OGS2'],['new','道路统一时刻新版'],
['source','实际源照片投影'],['dji','大疆正射参考']];

let scope='bus';
const $=id=>document.getElementById(id);

names.forEach(([key,label])=>{let o=document.createElement('option');
o.value=key;
o.textContent=label;
$('single').appendChild(o)});
$('single').value='new';

function render(){let prefix=scope+'-';
let ext=scope==='road'?'.jpg':'.png';

$('old-image').src=prefix+'old'+ext;
$('new-image').src=prefix+'new'+ext;

$('compare').style.aspectRatio=scope==='bus'?'__BUS_ASPECT__':'__ROAD_ASPECT__';

$('bus').setAttribute('aria-pressed',scope==='bus');
$('road').setAttribute('aria-pressed',scope==='road');

$('five').replaceChildren();
$('downloads').replaceChildren();

names.forEach(([key,label])=>{
let f=document.createElement('figure'),c=document.createElement('figcaption'),
a=document.createElement('a'),im=document.createElement('img');
c.textContent=label;
a.href=prefix+key+'.png';
a.target='_blank';
im.src=prefix+key+ext;
im.alt=label;
im.loading='lazy';
a.appendChild(im);
f.append(c,a);
$('five').appendChild(f);

let d=document.createElement('a');
d.href=prefix+key+'.png';
d.textContent=label;
d.download='';
$('downloads').append(d,document.createTextNode('　'))});
single()}
function single(){let path=scope+'-'+$('single').value+'.png';
$('single-image').src=path;
$('single-link').href=path;
}
$('single').onchange=single;
$('bus').onclick=()=>{scope='bus';
render()};
$('road').onclick=()=>{scope='road';
render()};

$('slider').oninput=()=>{let n=$('slider').value;
$('new-image').style.clipPath='inset(0 '+(100-n)+'% 0 0)';
$('line').style.left=n+'%'};
render();

</script>
</html>"""


def build(args):
    project, old_path, full_dir, patch_dir, dji_path, output = [
        Path(value).resolve()
        for value in (args.project, args.old_ogs, args.full, args.patch, args.dji, args.output)
    ]
    if output.exists():
        raise FileExistsError("Review output must be fresh")
    for preserved in (project, old_path, full_dir, patch_dir, dji_path):
        if output.is_relative_to(preserved) or preserved.is_relative_to(output):
            raise ValueError("Review output must not overlap preserved inputs")
    native_path = project / "odm_orthophoto/odm_orthophoto.tif"
    full_report_path = full_dir / "report.json"
    full_report = load_json(full_report_path)
    full_path = full_dir / "ogs2_orthophoto.tif"
    patch_path = patch_dir / "ogs2_orthophoto.tif"
    patch_report_path = patch_dir / "report.json"
    load_json(patch_report_path)
    if Path(full_report["base"]).resolve() != old_path:
        raise ValueError("Full export has a different composition base")
    if Path(full_report["patch"]).resolve() != patch_dir:
        raise ValueError("Full export has a different accepted patch")
    for key in ("outsideWriteDifferentPixels", "alphaDifferentPixels"):
        if full_report[key] != 0:
            raise ValueError(f"Full export violated {key}")
    validation_path = Path(full_report["validation"]).resolve()
    validation = load_json(validation_path)
    if not validation.get("checksPassed") or not validation.get("allCachesLinkedToActualPhotoHashes"):
        raise ValueError("Accepted patch lacks successful actual-source validation")
    objects_path = patch_dir / "objects.json"
    accepted = [item for item in load_json(objects_path) if item.get("state") == "accepted"]
    if len(accepted) != 1:
        raise ValueError("This review expects exactly one accepted road component")
    chosen = accepted[0]
    source_path = Path(chosen["sourceCache"]).resolve()
    metadata_path = source_path.with_suffix(".json")
    metadata = load_json(metadata_path)
    source_photo = project / "opensfm/undistorted/images" / chosen["sourceImage"]
    evidence_path = patch_dir / "reviewed-bus-evidence.npz"
    all_paths = [
        native_path,
        old_path,
        full_path,
        patch_path,
        dji_path,
        source_path,
        metadata_path,
        source_photo,
        evidence_path,
        full_report_path,
        validation_path,
        patch_report_path,
        objects_path,
    ]
    inputs = [{"path": str(path), "sha256": digest(path)} for path in dict.fromkeys(all_paths)]
    pinned = {entry["path"]: entry["sha256"] for entry in inputs}
    for path, expected in (
        (old_path, full_report["baseSha256"]),
        (full_path, full_report["outputSha256"]),
        (patch_path, full_report["patchSha256"]),
        (validation_path, full_report["validationSha256"]),
        (source_photo, metadata["sourceImageSha256"]),
    ):
        if pinned[str(path)] != expected:
            raise ValueError(f"Preserved input hash changed: {path}")
    with (
        rio.open(patch_path) as frame,
        rio.open(native_path) as native,
        rio.open(old_path) as old,
        rio.open(full_path) as full,
    ):
        if not frame.crs.is_projected or frame.crs.linear_units_factor[1] != 1:
            raise ValueError("Review requires projected coordinates measured in metres")
        if not np.allclose(
            [frame.transform.a, frame.transform.e, frame.transform.b, frame.transform.d],
            [0.05, -0.05, 0, 0],
            rtol=5e-5,
            atol=1e-10,
        ):
            raise ValueError("This review must use the actual north-up 5 cm pixel grid")
        canonical = {
            "crs": frame.crs.to_string(),
            "transform": list(frame.transform),
            "width": frame.width,
            "height": frame.height,
        }
        source_grid = metadata["grid"]
        if (
            rio.crs.CRS.from_user_input(source_grid["crs"]) != frame.crs
            or source_grid["width"] != frame.width
            or source_grid["height"] != frame.height
        ):
            raise ValueError("Source projection grid differs from patch")
        if not np.allclose(
            list(Affine(*source_grid["transform"][:6])),
            list(frame.transform),
            rtol=0,
            atol=1e-8,
        ):
            raise ValueError("Source projection pixel centres differ from patch")
        arrays = {
            "native": rgba(native, aligned_window(native, frame)),
            "old": rgba(old, aligned_window(old, frame)),
            "new": rgba(full, aligned_window(full, frame)),
        }
        if not np.array_equal(arrays["new"], rgba(frame)):
            raise ValueError("Full output does not contain the actual accepted patch")
        if not np.array_equal(arrays["old"][:, :, 3], arrays["new"][:, :, 3]):
            raise ValueError("Full output changes patch alpha")
        with np.load(source_path, allow_pickle=False) as cache:
            image, valid = cache["image"], cache["valid"].astype(bool)
        if (
            image.shape != (frame.height, frame.width, 3)
            or image.dtype != np.uint8
            or valid.shape != frame.shape
        ):
            raise ValueError("Source projection array does not match actual frame")
        arrays["source"] = np.dstack((image, valid.astype(np.uint8) * 255))
        with rio.open(dji_path) as dji:
            if dji.crs is None:
                raise ValueError("DJI reference has no declared coordinate system")
            dji_native = {
                "crs": dji.crs.to_string(),
                "transform": list(dji.transform),
                "width": dji.width,
                "height": dji.height,
                "resampling": "bilinear",
                "alignment": "Declared GeoTIFF coordinates only; no fitted translation or rotation",
            }
            options = {
                "crs": frame.crs,
                "transform": frame.transform,
                "width": frame.width,
                "height": frame.height,
                "resampling": Resampling.bilinear,
            }
            if dji.count == 3:
                options["add_alpha"] = True
            with WarpedVRT(dji, **options) as vrt:
                arrays["dji"] = rgba(vrt)
    with np.load(evidence_path, allow_pickle=False) as masks:
        support = np.zeros(arrays["new"].shape[:2], bool)
        evidence_pixels = {}
        for key in ("retained", "historical", "selectedShadow", "historicalShadow"):
            mask = masks[key].astype(bool)
            if mask.shape != support.shape:
                raise ValueError("Bus evidence differs from actual map frame")
            evidence_pixels[key] = int(mask.sum())
            support |= mask
    box = bbox(support, 80)
    old_instances, new_instances = args.reviewed_old_instances, args.reviewed_new_instances
    if (old_instances is None) != (new_instances is None):
        raise ValueError("Manual instance counts must be supplied together")
    if old_instances is not None and (old_instances < 1 or new_instances < 1):
        raise ValueError("Manual physical-vehicle counts must be positive")
    claim = "重点核验同一辆公交的历史重复位置、完整车身与真实阴影。"
    if old_instances is not None:
        claim = (
            f"人工核验的同一辆公交：旧版出现 {old_instances} 处，新版保留 {new_instances} 处。"
            "这个数量来自真实影像人工核验，不是分割掩膜计数。"
        )
    summary = {
        "schemaVersion": "ogs2-offline-road-review/v1",
        "inputs": inputs,
        "grid": canonical,
        "busWindowOnPatch": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
        "roadWindowOnPatch": [0, 0, canonical["width"], canonical["height"]],
        "sourceImage": chosen["sourceImage"],
        "evidencePixelCountsNotVehicleCounts": evidence_pixels,
        "scope": (
            "Reviewed local road and one manually reconciled bus identity; "
            "not all-scene vehicle truth or a global solution."
        ),
        "manualReviewedInstances": {
            "old": old_instances,
            "new": new_instances,
            "basis": (
                "Optional explicit counts from human visual review of the same physical bus; "
                "not inferred from mask component counts."
            ),
        },
        "fullExportChecks": {
            key: full_report[key]
            for key in ("changedPixels", "outsideWriteDifferentPixels", "alphaDifferentPixels")
        },
        "djiReference": dji_native,
        "pixelProcessing": {
            "busPng": "Actual RGBA pixels, lossless PNG, no resize or colour adjustment",
            "roadPng": "Actual RGBA pixels, lossless PNG",
            "roadJpeg": (
                "Only downsampled to at most 1400 px for page preview; transparent coverage displayed black"
            ),
            "source": (
                "Existing calibrated source projection, not independently rendered or aligned by this script"
            ),
            "generatedContent": False,
            "handRegistration": False,
        },
        "limitations": [
            (
                "The vehicle's real shadow must remain; dark-difference evidence "
                "is not independent semantic shadow ground truth."
            ),
            (
                "Native 2.5D and DJI are references; only old OGS2 versus new "
                "uses the same frozen base composition."
            ),
            "No absolute survey accuracy claim; DJI original resolution and processing differ.",
            (
                "Source coverage outside the exposure is transparent/black only in review, "
                "never newly written into the orthophoto."
            ),
        ],
        "scriptSha256": digest(Path(__file__)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".road-review-", dir=output.parent))
    try:
        save_views(staging, arrays, box)
        write_json(staging / "summary.json", summary)
        shutil.copyfile(full_report_path, staging / "full-pixel-check.json")
        shutil.copyfile(validation_path, staging / "patch-pixel-check.json")
        replacements = {
            "__BUS_ASPECT__": f"{box[2] - box[0]} / {box[3] - box[1]}",
            "__ROAD_ASPECT__": f"{canonical['width']} / {canonical['height']}",
            "__REVIEW_CLAIM__": html.escape(claim),
            "__CHANGED__": f"{full_report['changedPixels']:,}",
            "__OUTSIDE__": str(full_report["outsideWriteDifferentPixels"]),
            "__ALPHA__": str(full_report["alphaDifferentPixels"]),
            "__SOURCE__": html.escape(chosen["sourceImage"]),
        }
        page = PAGE
        for marker, value in replacements.items():
            page = page.replace(marker, value)
        (staging / "index.html").write_text(page, encoding="utf-8")
        # Validate artifact dimensions without any screenshot manipulation.
        for key in arrays:
            with Image.open(staging / f"bus-{key}.png") as image:
                if image.size != (box[2] - box[0], box[3] - box[1]):
                    raise RuntimeError("Written crop dimensions differ")
        if output.exists():
            raise FileExistsError("Output appeared during review generation")
        staging.rename(output)
    except Exception:
        # Keep failure evidence in the private staging directory for diagnosis.
        raise
    return {
        "output": str(output),
        "index": str(output / "index.html"),
        "summary": str(output / "summary.json"),
        "busWindowOnPatch": summary["busWindowOnPatch"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "old-ogs", "full", "patch", "dji", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--reviewed-old-instances", type=int)
    parser.add_argument("--reviewed-new-instances", type=int)
    args = parser.parse_args()
    print(json.dumps(build(args), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
