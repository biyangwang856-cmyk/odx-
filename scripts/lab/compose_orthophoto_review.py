"""Compose an explicitly unaccepted whole-scene review without changing production.

Rejected seams require --include-rejected-seams-for-review. This visualization
switch cannot change a quality decision or authorize production adoption.
"""
from pathlib import Path
import argparse,hashlib,json,shutil,time
import numpy as np,rasterio as rio
from rasterio.windows import Window
from rasterio.enums import Resampling
from PIL import Image

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',required=True,type=Path);p.add_argument('--include-rejected-seams-for-review',action='store_true');a=p.parse_args()
 summary=json.loads((a.inputs/'summary.json').read_text());original=Path(summary['originalRaster']);output=a.inputs/'UNACCEPTED-current-full-orthophoto-review.tif'
 if output.exists():raise ValueError('Never overwrite an existing review raster')
 if sha(original)!=summary['originalSha256']:raise ValueError('Original binding changed')
 patches=[]
 for r in summary['regions']:
  if not r['closedCutFeasible']:continue
  if not r.get('photometricSeamGatePassed',False) and not a.include_rejected_seams_for_review:raise ValueError('Rejected seam requires explicit review switch')
  for path,digest in r['files'].items():
   if sha(path)!=digest:raise ValueError('Candidate inputs changed')
  window=Window(*r['window'])
  with np.load(a.inputs/(r['region']+'-closed-patch-provenance.npz'),allow_pickle=False) as d:take=d['take_candidate'];support=d['geometric_support'];index=d['source_index'];seam=d['seam']
  if not np.array_equal(index,take.astype(np.uint8)):raise ValueError('Source index changed')
  with rio.open(original) as src,rio.open(a.inputs/(r['region']+'-closed-patch-candidate.tif')) as local:
   before=src.read([1,2,3],window=window);rgb=local.read([1,2,3])
   if local.transform!=src.window_transform(window) or local.crs!=src.crs or tuple(rgb.shape[1:])!=take.shape:raise ValueError('Candidate grid mismatch')
   if not np.array_equal(before[:,~take],rgb[:,~take]):raise ValueError('Candidate changed fixed base pixels')
  patches.append(dict(window=window,rgb=rgb,take=take,support=support,seam=seam,report=r))
 if not patches:raise ValueError('No feasible closed patch to visualize')
 print('Copying original into separate UNACCEPTED review raster',flush=True);shutil.copy2(original,output)
 with rio.open(output,'r+') as dst:
  dst.update_tags(status='UNACCEPTED-experimental-review',accepted_for_production='false',rejected_seams_included=str(a.include_rejected_seams_for_review).lower(),regions=','.join(p['report']['region'] for p in patches),original_sha256=summary['originalSha256'],original_raster=str(original))
  for patch in patches:
   old=dst.read(window=patch['window']);old[:3]=np.where(patch['take'][None],patch['rgb'],old[:3]);dst.write(old,window=patch['window'])
  factors=dst.overviews(1) or [2,4,8,16,32];dst.build_overviews(factors,Resampling.average)
  overview=dst.read([1,2,3],out_shape=(3,2000,1998),resampling=Resampling.average).transpose(1,2,0);Image.fromarray(overview).save(a.inputs/'current-full-preview.jpg',quality=95)
  with rio.open(original) as src:
   # A road overview on a common grid, plus untouched original for comparison.
   win=Window(5000,17500,16500,3500)
   for name,reader in [('before-road-overview.jpg',src),('current-road-overview.jpg',dst)]:
    image=reader.read([1,2,3],window=win,out_shape=(3,875,4125),resampling=Resampling.average).transpose(1,2,0);Image.fromarray(image).save(a.inputs/name,quality=95)
 print('Verifying every pixel outside declared replacement masks',flush=True)
 changed=0;outside=0;bands_outside_rgb=0
 with rio.open(original) as src,rio.open(output) as dst:
  profile=dict(driver='GTiff',height=src.height,width=src.width,count=1,dtype='uint8',crs=src.crs,transform=src.transform,tiled=True,blockxsize=512,blockysize=512,compress='deflate')
  with rio.open(a.inputs/'current-full-provenance.tif','w',**profile) as prov:
   prov.update_tags(classes='0=unchanged-OGS2;1=western-unaccepted-review;2=central-unaccepted-review',accepted_for_production='false')
   for _,block in prov.block_windows(1):
    before=src.read(window=block);after=dst.read(window=block);diff=np.any(before!=after,axis=0);legal=np.zeros(diff.shape,np.uint8)
    for idx,patch in enumerate(patches,1):
     window=patch['window'];take=patch['take'];l=max(int(block.col_off),int(window.col_off));t=max(int(block.row_off),int(window.row_off));r=min(int(block.col_off+block.width),int(window.col_off+window.width));b=min(int(block.row_off+block.height),int(window.row_off+window.height))
     if l<r and t<b:
      part=legal[t-int(block.row_off):b-int(block.row_off),l-int(block.col_off):r-int(block.col_off)];mask=take[t-int(window.row_off):b-int(window.row_off),l-int(window.col_off):r-int(window.col_off)];part[mask]=idx
    changed+=int(diff.sum());outside+=int((diff&(legal==0)).sum())
    if before.shape[0]>3:bands_outside_rgb+=int(np.count_nonzero(before[3:]!=after[3:]))
    prov.write(legal,1,window=block)
 if outside or bands_outside_rgb:raise ValueError('Review violated immutable-base/alpha contract')
 if sha(original)!=summary['originalSha256']:raise ValueError('Original modified during composition')
 report=dict(state='completed',status='UNACCEPTED-experimental-review',acceptedForProduction=False,fullSceneSolved=False,path=str(output),sha256=sha(output),sizeBytes=output.stat().st_size,originalSha256=summary['originalSha256'],originalPreserved=True,changedPixels=changed,changesOutsideDeclaredPatches=outside,nonRgbChanges=bands_outside_rgb,rejectedSeamsIncluded=True,regions=[{k:p['report'][k] for k in ['region','photometricSeamGatePassed','seamRgbDifferenceP90','seamRgbDifferenceMax','holdoutHeightP90M','holdoutReprojectionP90Px']} for p in patches])
 (a.inputs/'whole-review.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 manifest=dict(files=[dict(name=f.name,sizeBytes=f.stat().st_size,sha256=sha(f)) for f in a.inputs.iterdir() if f.is_file() and f.name!='manifest.json' and f.suffix!='.log'])
 (a.inputs/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
