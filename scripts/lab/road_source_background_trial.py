"""Exploratory source-coherent road background from real photographs.

Does not replace the production raster or certify unobserved road content.
"""
from pathlib import Path
import argparse,json,time,hashlib
import numpy as np,cv2,rasterio as rio
from rasterio.windows import Window
from rasterio.transform import Affine
from PIL import Image
from model_engine.ogs2 import Detector,Config,cameras_from_checkpoint
from model_engine.ogs2_corridor_projection import bilinear_support
from model_engine.ogs2_road_registration import fit_road_registration,RoadRegistrationConfig
from model_engine.ogs2_road_patch import select_road_patch,compose_road_patch
from model_engine.ogs2_source_guard import build_source_guard,project_source_guard

def source_guard(directory,name,project):
 manifest=json.loads((directory/'manifest.json').read_text())
 if manifest['state']!='completed':raise ValueError('Require completed source masks')
 reconstruction=project/'opensfm/undistorted/reconstruction.json'
 if hashlib.sha256(reconstruction.read_bytes()).hexdigest()!=manifest['reconstructionSha256']:raise ValueError('Source camera binding changed')
 photo=next((p for p in manifest['photos'] if p['image']==name),None)
 if photo is None or Path(name).name!=name:raise ValueError('Missing bound source photo')
 file=directory/'masks'/(name+'.png')
 if hashlib.sha256(file.read_bytes()).hexdigest()!=photo['maskSha256']:raise ValueError('Source mask changed')
 with Image.open(file) as image:
  if image.mode!='L':raise ValueError('Require original grey source mask')
  mask=np.asarray(image).copy()
 guard,report=build_source_guard(mask,photo)
 report.update(image=name,maskSha256=photo['maskSha256'],reconstructionSha256=manifest['reconstructionSha256'])
 return guard,report
def objects(rgb,road):
 h,w=road.shape;torch=detector.torch;proposals=[];disputed=[]
 for y in sorted(set([0,max(0,h-760),*range(0,max(1,h-760),600)])):
  for x in sorted(set([0,max(0,w-760),*range(0,max(1,w-760),600)])):
   tile=rgb[y:y+760,x:x+760];inputs=detector.processor(images=Image.fromarray(tile),text='car. truck. bus.',return_tensors='pt').to(detector.device)
   with torch.inference_mode():
    result=detector.model(**inputs);r=detector.processor.post_process_grounded_object_detection(result,inputs.input_ids,box_threshold=.22,text_threshold=.2,target_sizes=[tile.shape[:2]])[0]
   for bb,score in zip(r['boxes'].cpu().numpy(),r['scores'].cpu().numpy()):
    l,t,rr,b=bb+[x,y,x,y];cx,cy=np.rint([(l+rr)/2,(t+b)/2]).astype(int);area=float((rr-l)*(b-t)*.03**2);record=dict(box=[float(i) for i in [l,t,rr,b]],score=float(score),areaM2=area)
    if not road[min(h-1,max(0,cy)),min(w-1,max(0,cx))] or area>90:disputed.append(record);continue
    proposals.append(record)
 # NMS avoids duplicated tile proposals; unmatched/oversize are saved as unknown.
 order=sorted(proposals,key=lambda r:-r['score']);keep=[]
 for p in order:
  a=np.array(p['box']);duplicate=False
  for q in keep:
   b=np.array(q['box']);xy=np.maximum(a[:2],b[:2]);uv=np.minimum(a[2:],b[2:]);intersection=np.maximum(0,uv-xy).prod();union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
   if intersection/max(union,1)>.45:duplicate=True;break
  if not duplicate:keep.append(p)
 body=np.zeros((h,w),bool);guard=np.zeros((h,w),bool)
 if keep:
  with torch.inference_mode():
   detector.sam.set_image(rgb);masks,quality,_=detector.sam.predict(box=np.array([p['box'] for p in keep]),multimask_output=False)
  gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
  for p,mask,q in zip(keep,masks.reshape(-1,h,w),np.ravel(quality)):
   l,t,r,b=np.rint(p['box']).astype(int);p['samScore']=float(q);mask=mask>0
   box=np.zeros((h,w),np.uint8);box[max(0,t):min(h,b),max(0,l):min(w,r)]=1
   if q<.8 or mask.sum()==0 or (mask&~box.astype(bool)).sum()>mask.sum()*.1:
    mask=box.astype(bool);p['maskKind']='conservative-box'
   else:p['maskKind']='sam'
   body|=mask
   l0,t0,r0,b0=max(0,l-90),max(0,t-100),min(w,r+100),min(h,b+70)
   small=mask[t0:b0,l0:r0];g=gray[t0:b0,l0:r0];ring=cv2.dilate(small.astype(np.uint8),np.ones((25,25),np.uint8)).astype(bool)&~small
   typical=float(np.median(g[ring])) if ring.any() else 145.
   dark=(g<min(120.,typical*.78));join=dark|(cv2.dilate(small.astype(np.uint8),np.ones((9,9),np.uint8))>0)
   count,labels=cv2.connectedComponents(join.astype(np.uint8));touch=np.unique(labels[small]);touch=touch[touch>0]
   shadow=np.isin(labels,touch);local=cv2.dilate((shadow|small).astype(np.uint8),np.ones((13,13),np.uint8))>0
   guard[t0:b0,l0:r0]|=local;p['shadowSearchBox']=[int(v) for v in [l0,t0,r0,b0]];p['shadowPixels']=int((shadow&~small).sum())
 return body,guard,dict(proposals=keep,disputed=disputed,shadowAlgorithm='local dark connected component touching measured body; not guaranteed complete',semanticCertified=False)

def main():
 global detector
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--workspace',required=True,type=Path)
 p.add_argument('--job',required=True,type=Path)
 p.add_argument('--surface-dir',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path)
 p.add_argument('--regions',nargs='+',default=['west','centre'])
 p.add_argument('--diagnostic-geometry',action='store_true',help='Allow bounded rejected registration only for explicitly unaccepted diagnostic output')
 p.add_argument('--source-masks',type=Path,help='Use bound source-pixel exclusions, including uncertain vehicle boxes; independent of projected-image segmentation')
 args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 root=args.workspace;job=args.job;base=args.surface_dir;out=args.output
 detector=None if args.source_masks else Detector(root/'runtime/vehicle-models','cuda:0',Config())
 reports=[]
 with rio.open(job/'datasets/project/odm_orthophoto/odm_orthophoto.tif') as grid:
  factory,data=cameras_from_checkpoint(job/'datasets/project',grid.crs)
  for region in args.regions:
   start=time.time();meta=json.loads((base/'expanded-patch-inputs-02'/f'{region}-report.json').read_text());folder=meta['planeSource']
   with np.load(base/'expanded-patch-inputs-02'/f'{region}.npz') as d:
    anchor=d['candidate'];va=d['valid_candidate'];body=d['body_candidate'];guard=d['guard_candidate'];road=d['road'];anchor_uv=d['uv']
   anchor_guard_report=None
   if args.source_masks:
    native_guard,anchor_guard_report=source_guard(args.source_masks,meta['sourceImage'],job/'datasets/project')
    guard=guard|project_source_guard(native_guard,anchor_uv)
   h,w=road.shape;win=Window(*meta['window']);tr=grid.window_transform(win)
   pair=json.loads((base/folder/f'{region}-lanes-pair-0-2.json').read_text())['qualityAcceptedMeasurements']
   a=np.array([r['first'] for r in pair])+[250,200];b=np.array([r['second'] for r in pair])+[250,200]
   # Move the split origin with the coordinate origin. The exact same physical
   # points remain in each holdout cell, while the entire sampling domain gets
   # its own displacement and Jacobian check.
   _,_,original_width,original_height=meta['previewTargetWindow']
   fit=fit_road_registration(a,b,(h,w),config=RoadRegistrationConfig(cell_origin_xy=(250.,200.)))
   report=dict(region=region,registration=fit.report,registrationAccepted=fit.accepted,
               unacceptedDiagnosticCorrectionUsed=not fit.accepted,
               sourceObjectAbsenceIsNotEmptyRoadProof=True,automaticQualityAcceptance=False,
               productionRasterReplaced=False,groups=[],fullSceneSolved=False)
   report['sourceMaskGuardEnabled']=bool(args.source_masks);report['anchorSourceGuard']=anchor_guard_report
   (out/f'{region}-registration.json').write_text(json.dumps(fit.report,indent=2))
   bounded=(fit.matrix is not None and fit.report.get('maxDomainDisplacementPx',1e6)<=8 and
            min(fit.report.get('jacobianSingularValues',[0]))>=.99 and max(fit.report.get('jacobianSingularValues',[2]))<=1.01)
   if not fit.accepted and not (args.diagnostic_geometry and bounded):
    report['reason']='registration-refused';reports.append(report);(out/'reports.json').write_text(json.dumps(reports,indent=2));print(region,report['reason'],flush=True);continue
   yy,xx=np.mgrid[:h,:w];points=np.stack([xx,yy],-1).astype(float)
   registration_domain=np.ones((h,w),bool)
   xy=fit.source_coordinates(points,allow_diagnostic=args.diagnostic_geometry);X,Y=tr*(xy[...,0]+.5,xy[...,1]+.5)
   report['registrationDomainExpandedWithSamePhysicalHoldoutCells']=True
   report['extrapolatedRoadSurfaceCertified']=False
   surface=json.loads((base/folder/f'{region}-lanes-fit.json').read_text())['fit'];c=surface['coefficients'];ox,oy=surface['originXY'];Z=c[0]*(X-ox)+c[1]*(Y-oy)+c[2]
   tracks=json.loads((base/'multiview-tracks-02/tracks.json').read_text());r=next(r for r in tracks['regions'] if r['label']==region+'-lanes');name=r['sourceViews'][2]['name'];cam=factory.get(name)
   uv=cam.world_to_pixel(np.stack([X,Y,Z]).reshape(3,-1)).reshape(2,h,w).astype(np.float32)
   with rio.open(job/'datasets/project/opensfm/undistorted/images'/name) as photo:raw=photo.read([1,2,3]).transpose(1,2,0);rawvalid=photo.dataset_mask()>0
   second=cv2.remap(raw,uv[0],uv[1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT);vb=bilinear_support(rawvalid,uv)
   if args.source_masks:
    native_guard,detections=source_guard(args.source_masks,name,job/'datasets/project')
    secondguard=project_source_guard(native_guard,uv);secondbody=np.zeros(road.shape,bool)
    detections.update(projectedGuardPixels=int(secondguard.sum()),projectedBodySegmentationRecomputed=False)
   else:secondbody,secondguard,detections=objects(second,road)
   np.savez_compressed(out/f'{region}-aligned-inputs.npz',anchor=anchor,second=second,valid_anchor=va,valid_second=vb,body_anchor=body,guard_anchor=guard,body_second=secondbody,guard_second=secondguard,road=road,registration_domain=registration_domain,uv_second=uv)
   report['secondSourceImage']=name;report['anchorSourceImage']=meta['sourceImage'];report['secondDetections']=detections
   Image.fromarray(second).save(out/f'{region}-second.jpg',quality=95)
   count,labels=cv2.connectedComponents(guard.astype(np.uint8));output=anchor.copy();provenance=np.zeros((h,w),np.uint8);allseams=np.zeros((h,w),bool)
   for index in range(1,count):
    core=labels==index;record=dict(group=index,pixels=int(core.sum()),bodyPixels=int((body&core).sum()))
    if record['bodyPixels']<20:record['reason']='no-source-body';report['groups'].append(record);continue
    if (core&~road).any() or (core&~va).any() or (core&~vb).any() or (core&~registration_domain).any():record['reason']='incomplete-road-photograph-or-registration-coverage';report['groups'].append(record);continue
    if (core&secondguard).any():record['reason']='other-source-object-or-shadow-overlap-keep-whole-anchor';report['groups'].append(record);continue
    allowed=(cv2.dilate(core.astype(np.uint8),np.ones((81,81),np.uint8))>0)&road&va&vb&registration_domain
    cut=select_road_patch(output,second,va,vb,core,allowed,guard|secondguard)
    record.update(cutFeasible=cut.accepted,cutReason=cut.reason,qc=cut.qc,applied=False)
    if cut.accepted:
     difference=np.mean(np.abs(output.astype(float)-second),axis=2);values=difference[cut.seam_mask]
     record['seamRgbDifferenceP90']=float(np.percentile(values,90));record['seamRgbDifferenceMax']=float(values.max())
     if record['seamRgbDifferenceP90']<=25 and record['seamRgbDifferenceMax']<=80:
      result,local,qc=compose_road_patch(output,second,cut);output=result;provenance[cut.take_candidate]=2;allseams|=cut.seam_mask;record['applied']=True
     else:record['reason']='visible-photometric-seam-refused'
    report['groups'].append(record)
   report['replacedGroups']=sum(r.get('applied',False) for r in report['groups']);report['keptWholeGroups']=sum(r.get('bodyPixels',0)>=20 and not r.get('applied',False) for r in report['groups']);report['elapsedSeconds']=time.time()-start
   Image.fromarray(output).save(out/f'{region}-source-coherent.png');ann=output.copy();ann[allseams]=[255,0,255];Image.fromarray(ann).save(out/f'{region}-source-seams.jpg',quality=95)
   x,y,pw,ph=meta['previewTargetWindow'];Image.fromarray(output[y:y+ph,x:x+pw]).save(out/f'{region}-target.png')
   np.savez_compressed(out/f'{region}-provenance.npz',source_index=provenance,seams=allseams)
   rgba=np.concatenate([output,np.where(va,255,0).astype(np.uint8)[...,None]],axis=2)
   with rio.open(out/f'{region}-source-coherent-experimental.tif','w',driver='GTiff',height=h,width=w,count=4,dtype='uint8',transform=tr,crs=grid.crs,compress='deflate') as dst:dst.write(rgba.transpose(2,0,1))
   reports.append(report);(out/'reports.json').write_text(json.dumps(reports,indent=2))
   print(region,'registration',fit.accepted,'replaced',report['replacedGroups'],'kept',report['keptWholeGroups'],flush=True)

if __name__=='__main__':main()
