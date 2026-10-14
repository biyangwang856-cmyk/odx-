"""Compose reviewable road patches without replacing the production raster.

Inputs are two independently prepared photographs on one map grid plus masks.
This is an experimental renderer, not an automatic production acceptance gate.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import rasterio as rio
from rasterio.transform import Affine
from PIL import Image
from model_engine.ogs2_road_patch import select_road_patch,compose_road_patch

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--inputs',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path)
 p.add_argument('--regions',nargs='+',default=['west','centre'])
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 reports=[]
 for region in a.regions:
  if not region.replace('-','').replace('_','').isalnum():raise ValueError('Invalid region identifier')
  started=time.time();meta=json.loads((a.inputs/f'{region}-report.json').read_text())
  with np.load(a.inputs/f'{region}.npz',allow_pickle=False) as d:
   base=d['base'];candidate=d['candidate'];va=d['valid_base'];vb=d['valid_candidate'];hard=d['hard'];allowed=d['allowed'];guard=d['guard_candidate']
  # A damaged old vehicle cannot be reliably segmented from the already
  # corrupted mosaic. Old proposals are retained as uncertain in the inputs,
  # and colour/gradient differences remain in the cut cost. Only the measured
  # source-object protection is a hard no-cut constraint in this ablation.
  cut=select_road_patch(base,candidate,va,vb,hard,allowed,guard)
  report=dict(region=region,discreteCutFeasible=cut.accepted,reason=cut.reason,
              elapsedSeconds=time.time()-started,qc=cut.qc,bindings=cut.bindings,
              inputDirectory=str(a.inputs.resolve()),sourceImage=meta['sourceImage'],
              sourceMetadata=meta,oldObjectCoverageCertified=False,
              candidateSurfaceExtrapolationCertified=False,
              productionRasterReplaced=False,automaticQualityAcceptance=False)
  if cut.accepted:
   image,index,qc=compose_road_patch(base,candidate,cut);h,w=index.shape
   with rio.open(a.output/f'{region}-experimental.tif','w',driver='GTiff',width=w,height=h,count=3,dtype='uint8',crs=meta['crs'],transform=Affine(*meta['transform']),compress='deflate') as dst:dst.write(image.transpose(2,0,1))
   Image.fromarray(image).save(a.output/f'{region}-experimental.png')
   ann=image.copy();ann[cut.seam_mask]=[255,0,255];Image.fromarray(ann).save(a.output/f'{region}-seam.jpg',quality=95)
   np.savez_compressed(a.output/f'{region}-provenance.npz',source_index=index,seam=cut.seam_mask,hard=hard,allowed=allowed,forbidden=guard)
   if 'previewTargetWindow' in meta:
    x,y,w,h=meta['previewTargetWindow'];Image.fromarray(image[y:y+h,x:x+w]).save(a.output/f'{region}-target.png')
  reports.append(report);(a.output/'reports.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
  print(json.dumps({k:report[k] for k in ['region','discreteCutFeasible','reason','elapsedSeconds','qc']}),flush=True)

if __name__=='__main__':main()
