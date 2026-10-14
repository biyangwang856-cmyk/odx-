"""Export current road orthophoto candidates with explicit review limitations.

Does not promote experimental rasters or replace the original orthophoto.
Requires previously measured road-plane geometry and binds all input files.
"""
from pathlib import Path
import argparse,hashlib,json,time
import cv2,numpy as np,rasterio as rio
from rasterio.windows import Window
from rasterio.transform import Affine
from rasterio.enums import ColorInterp,Resampling
from PIL import Image
from model_engine.ogs2_road_patch import select_road_patch,compose_road_patch
from model_engine.ogs2_road_color import fit_road_color_offset
from model_engine.ogs2_source_guard import project_source_guard
from road_source_background_trial import source_guard
from road_background_color_trial import marking_mask

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def write_rgb(path,rgb,valid,transform,crs):
 with rio.open(path,'w',driver='GTiff',height=rgb.shape[0],width=rgb.shape[1],count=4,dtype='uint8',transform=transform,crs=crs,tiled=True,compress='deflate',BIGTIFF='IF_SAFER') as dst:
  dst.write(np.concatenate([rgb,np.where(valid,255,0).astype(np.uint8)[...,None]],axis=2).transpose(2,0,1));dst.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
  dst.update_tags(status='experimental-road-candidate',automatic_quality_acceptance='false',road_plane_is_not_vehicle_surface='true')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',required=True,type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 job=args.job;out=args.output;surf=job/'road-surface-20261013';inputs=surf/'expanded-patch-inputs-02';rawmasks=job/'vehicle-source-masks-20261014/raw-mask-pilot-03';reports=[];patches=[]
 original=job/'vehicle-ogs2/ogs2_orthophoto.tif'
 original_sha=sha(original)
 for region,folder in [('west','multiview-fit-02'),('centre','multiview-fit-01')]:
  started=time.time();meta_path=inputs/f'{region}-report.json';npz_path=inputs/f'{region}.npz';fit_path=surf/folder/f'{region}-lanes-fit.json';support_path=surf/folder/f'{region}-lanes-source-000.npz'
  meta=json.loads(meta_path.read_text());fit=json.loads(fit_path.read_text())['fit']
  if not fit['geometryAccepted']:raise ValueError('Never render a rejected road plane')
  with np.load(npz_path,allow_pickle=False) as d:
   base=d['base'];candidate=d['candidate'].copy();va=d['valid_base'];vb=d['valid_candidate'];road=d['road'];hard=d['hard'];allowed=d['allowed'];guard=d['guard_candidate']|d['guard_base'];uv=d['uv'];candidate_guard=d['guard_candidate'].copy()
  if region=='west':
   selected=surf/'source-background-guarded-colour-01/west-source-coherent.png';candidate=np.asarray(Image.open(selected).convert('RGB')).copy()
  else:selected=npz_path
  native_guard,guard_report=source_guard(rawmasks,meta['sourceImage'],job/'datasets/project')
  extra=project_source_guard(native_guard,uv);guard|=extra;candidate_guard|=extra
  x,y,w,h=meta['previewTargetWindow'];support=np.zeros(road.shape,bool)
  with np.load(support_path,allow_pickle=False) as d:
   if d['support'].shape!=(h,w):raise ValueError('Support window mismatch')
   support[y:y+h,x:x+w]=d['support']
  transform=Affine(*meta['transform']);window=Window(*meta['window'])
  with rio.open(original) as src:
   actual=src.read([1,2,3],window=window).transpose(1,2,0)
   if not np.array_equal(actual,base) or src.window_transform(window)!=transform or str(src.crs)!=meta['crs']:raise ValueError('Original grid/RGB binding mismatch')
  # Deliver both the source-plane context and the bounded road support. The
  # former is a diagnostic orthophoto; trees/vehicles are not a road plane.
  write_rgb(out/f'{region}-current-context.tif',candidate,vb,transform,meta['crs'])
  write_rgb(out/f'{region}-supported-road.tif',candidate,vb&road&support,transform,meta['crs'])
  Image.fromarray(base[y:y+h,x:x+w]).save(out/f'{region}-before.png');Image.fromarray(candidate[y:y+h,x:x+w]).save(out/f'{region}-current.png')
  quality=np.zeros(road.shape,np.uint8);quality[vb]=1;quality[vb&road]=2;quality[vb&road&support]=3
  with rio.open(out/f'{region}-quality-scope.tif','w',driver='GTiff',height=road.shape[0],width=road.shape[1],count=1,dtype='uint8',transform=transform,crs=meta['crs'],compress='deflate') as dst:dst.write(quality,1);dst.update_tags(classes='0=invalid;1=unvalidated-context;2=road-plane-extrapolation;3=road-and-geometric-support;none-is-full-semantic-certification')
  paint=marking_mask(base,road)|marking_mask(candidate,road)
  static_collar=road&va&vb&~guard&~paint&~hard
  colour=fit_road_color_offset(base,candidate,static_collar)
  corrected=colour.apply(candidate) if colour.accepted else candidate
  cut=select_road_patch(base,corrected,va,vb,hard,allowed,guard|paint)
  report=dict(region=region,status='local-experimental-orthophoto-exported',sourceImage=meta['sourceImage'],window=meta['window'],previewTargetWindow=meta['previewTargetWindow'],planeSource=str(fit_path),geometryAccepted=fit['geometryAccepted'],semanticAccepted=fit['accepted'],holdoutHeightP90M=fit['holdoutHeightP90M'],holdoutReprojectionP90Px=fit['holdoutReprojectionP90Px'],heightIsRelativeNotSurveyGroundTruth=True,supportedRoadPixels=int((vb&road&support).sum()),roadPixels=int(road.sum()),sourceGuard=guard_report,colourCorrection=colour.report,closedCutFeasible=cut.accepted,closedCutReason=cut.reason,cutQuality=cut.qc,cutBindings=cut.bindings,fullSceneSolved=False,automaticQualityAcceptance=False,originalRasterReplaced=False,contextOutsideRoadIsUnvalidatedPlaneProjection=True,carPixelsProjectedOnRoadPlaneNotTrueCarSurface=True,sourceTimestampIsRegionSpecificNotOneGlobalEpoch=True,files={str(f):sha(f) for f in [meta_path,npz_path,fit_path,support_path,selected]})
  if cut.accepted:
   image,index,qc=compose_road_patch(base,corrected,cut);values=np.abs(base.astype(float)-corrected).mean(2)[cut.seam_mask]
   report.update(seamRgbDifferenceP90=float(np.percentile(values,90)),seamRgbDifferenceMax=float(values.max()),candidateOutsideGeometricSupportPixels=int((cut.take_candidate&~support).sum()))
   photo_ok=report['seamRgbDifferenceP90']<=25 and report['seamRgbDifferenceMax']<=80
   report['photometricSeamGatePassed']=photo_ok
   write_rgb(out/f'{region}-closed-patch-candidate.tif',image,va|vb,transform,meta['crs']);ann=image.copy();ann[cut.seam_mask]=[255,0,255];Image.fromarray(ann).save(out/f'{region}-closed-seam.jpg',quality=95)
   np.savez_compressed(out/f'{region}-closed-patch-provenance.npz',source_index=index,seam=cut.seam_mask,take_candidate=cut.take_candidate,geometric_support=support)
   # A full review raster may include road extrapolation, which is explicitly
   # recorded. It is never declared a certified replacement for production.
   if photo_ok:patches.append((window,image,cut.take_candidate,region))
  report['seconds']=time.time()-started;reports.append(report);(out/'reports.json').write_text(json.dumps(reports,indent=2),encoding='utf-8');print(region,report['closedCutReason'],report.get('photometricSeamGatePassed'),flush=True)
 if patches:
  dstpath=out/'current-full-orthophoto-review.tif'
  with rio.open(original) as src:
   profile=src.profile.copy();profile.update(tiled=True,blockxsize=512,blockysize=512,compress='deflate',BIGTIFF='IF_SAFER')
   with rio.open(dstpath,'w',**profile) as dst:
    dst.colorinterp=src.colorinterp;dst.update_tags(**src.tags(),status='experimental-review-not-production',geometry_support='see-local-quality-scope-rasters')
    for _,block in dst.block_windows(1):dst.write(src.read(window=block),window=block)
    for window,rgb,take,region in patches:
     old=src.read(window=window);old[:3,:, :]=np.where(take[None,:,:],rgb.transpose(2,0,1),old[:3,:,:]);dst.write(old,window=window)
    dst.build_overviews([2,4,8,16,32],Resampling.average)
  # Prove every pixel outside explicit replacement masks is unchanged.
  changed=0;invalid_change=0
  with rio.open(original) as src,rio.open(dstpath) as dst:
   for _,block in dst.block_windows(1):
    before=src.read(window=block);after=dst.read(window=block);diff=np.any(before!=after,axis=0);legal=np.zeros(diff.shape,bool)
    for window,rgb,take,region in patches:
     l=max(int(block.col_off),int(window.col_off));t=max(int(block.row_off),int(window.row_off));r=min(int(block.col_off+block.width),int(window.col_off+window.width));b=min(int(block.row_off+block.height),int(window.row_off+window.height))
     if l<r and t<b:legal[t-int(block.row_off):b-int(block.row_off),l-int(block.col_off):r-int(block.col_off)]=take[t-int(window.row_off):b-int(window.row_off),l-int(window.col_off):r-int(window.col_off)]
    changed+=int(diff.sum());invalid_change+=int((diff&~legal).sum())
  if invalid_change:raise ValueError('Full raster changed pixels outside declared patch')
  full=dict(path=str(dstpath),sha256=sha(dstpath),changedPixels=changed,changesOutsideDeclaredPatch=invalid_change,status='experimental-review',acceptedForProduction=False,regions=[p[3] for p in patches])
 else:full=dict(path=None,status='no-full-raster-passed-closed-seam-gates',acceptedForProduction=False)
 if sha(original)!=original_sha:raise ValueError('Original raster changed during export')
 summary=dict(state='completed',originalRaster=str(original),originalSha256=original_sha,originalPreserved=True,fullRaster=full,regions=reports,rejectedDensePlanes=str(surf/'dense-road-fit-02/reports.json'),densePlaneFailuresAdopted=False)
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(full),flush=True)

if __name__=='__main__':main()
