"""Replay independent background inputs with a measured road colour correction."""
from pathlib import Path
import argparse,json,time
import cv2,numpy as np,rasterio as rio
from rasterio.transform import Affine
from PIL import Image
from model_engine.ogs2_road_color import fit_road_color_offset
from model_engine.ogs2_road_patch import select_road_patch,compose_road_patch

def marking_mask(rgb, road):
 f=rgb.astype(float)
 white=(f.min(-1)>190)&(f.max(-1)-f.min(-1)<65)
 yellow=(f[...,0]>150)&(f[...,1]>135)&(f[...,2]<140)&(f[...,0]-f[...,2]>25)
 paint=cv2.morphologyEx(((white|yellow)&road).astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,15),np.uint8))
 count,labels,stats,centres=cv2.connectedComponentsWithStats(paint)
 keep=np.zeros(count,bool)
 for k in range(1,count):
  _,_,w,h,area=stats[k]
  if 25<=area<=5000 and w>=4*max(1,h):keep[k]=True
 return cv2.dilate(keep[labels].astype(np.uint8),np.ones((5,5),np.uint8))>0

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--regions',nargs='+',default=['west','centre']);p.add_argument('--protect-markings',action='store_true');args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 prior=json.loads((args.inputs/'reports.json').read_text());reports=[]
 for region in args.regions:
  report=next(r for r in prior if r['region']==region).copy();report['previousTrial']=str(args.inputs);report['groups']=[];report['rgbExposureAdjusted']=True
  meta=json.loads((args.metadata/f'{region}-report.json').read_text())
  with np.load(args.inputs/f'{region}-aligned-inputs.npz') as d:
   anchor=d['anchor'];second=d['second'];va=d['valid_anchor'];vb=d['valid_second'];body=d['body_anchor'];guard=d['guard_anchor'];sg=d['guard_second'];road=d['road'];domain=d['registration_domain'] if 'registration_domain' in d.files else np.ones(road.shape,bool)
  output=anchor.copy();provenance=np.zeros(road.shape,np.uint8);allseams=np.zeros(road.shape,bool);offset_map=np.zeros((*road.shape,3),np.float32)
  paint=(marking_mask(anchor,road)|marking_mask(second,road)) if args.protect_markings else np.zeros(road.shape,bool)
  paint_count,paint_labels=cv2.connectedComponents(paint.astype(np.uint8));report['wholeMarkingProtection']=args.protect_markings
  count,labels=cv2.connectedComponents(guard.astype(np.uint8))
  for index in range(1,count):
   core=labels==index;record=dict(group=index,pixels=int(core.sum()),bodyPixels=int((body&core).sum()),applied=False)
   if record['bodyPixels']<20:record['reason']='no-source-body'
   elif (core&(~road|~va|~vb|~domain)).any():record['reason']='incomplete-road-photograph-or-registration-coverage'
   elif (core&sg).any():record['reason']='other-source-object-or-shadow-overlap-keep-whole-anchor'
   else:
    touched=np.unique(paint_labels[core]);touched=touched[touched>0]
    whole_markings=np.isin(paint_labels,touched) if len(touched) else np.zeros(road.shape,bool)
    hard=core|whole_markings
    if (hard&(~road|~va|~vb|~domain|sg)).any():
     record['reason']='whole-marking-has-object-or-coverage-conflict';report['groups'].append(record);continue
    allowed=(cv2.dilate(hard.astype(np.uint8),np.ones((81,81),np.uint8))>0)&road&va&vb&domain
    collar=(cv2.dilate(core.astype(np.uint8),np.ones((121,121),np.uint8))>0)&road&va&vb&~guard&~sg&domain
    color=fit_road_color_offset(anchor,second,collar);record['colourCalibration']=color.report
    if not color.accepted:record['reason']='colour-calibration-refused'
    else:
     corrected=color.apply(second);cut=select_road_patch(output,corrected,va,vb,hard,allowed,guard|sg|paint);record.update(cutFeasible=cut.accepted,cutReason=cut.reason,qc=cut.qc,wholeMarkingPixels=int(whole_markings.sum()))
     if cut.accepted:
      difference=np.mean(np.abs(output.astype(float)-corrected),axis=2);values=difference[cut.seam_mask];record.update(seamRgbDifferenceP90=float(np.percentile(values,90)),seamRgbDifferenceMax=float(values.max()))
      if record['seamRgbDifferenceP90']<=25 and record['seamRgbDifferenceMax']<=80:
       output,local,qc=compose_road_patch(output,corrected,cut);provenance[cut.take_candidate]=2;allseams|=cut.seam_mask;offset_map[cut.take_candidate]=color.offset;record['applied']=True
      else:record['reason']='visible-photometric-seam-refused'
   report['groups'].append(record)
  report['replacedGroups']=sum(r['applied'] for r in report['groups']);report['keptWholeGroups']=sum(r['bodyPixels']>=20 and not r['applied'] for r in report['groups'])
  Image.fromarray(output).save(args.output/f'{region}-source-coherent.png');ann=output.copy();ann[allseams]=[255,0,255];Image.fromarray(ann).save(args.output/f'{region}-source-seams.jpg',quality=95)
  x,y,w,h=meta['previewTargetWindow'];Image.fromarray(output[y:y+h,x:x+w]).save(args.output/f'{region}-target.png')
  np.savez_compressed(args.output/f'{region}-provenance.npz',source_index=provenance,seams=allseams,offset_rgb=offset_map)
  with rio.open(args.output/f'{region}-source-coherent-experimental.tif','w',driver='GTiff',height=road.shape[0],width=road.shape[1],count=4,dtype='uint8',transform=Affine(*meta['transform']),crs=meta['crs'],compress='deflate') as dst:dst.write(np.concatenate([output,np.where(va,255,0).astype(np.uint8)[...,None]],axis=2).transpose(2,0,1))
  reports.append(report);(args.output/'reports.json').write_text(json.dumps(reports,indent=2));print(region,'replaced',report['replacedGroups'],'kept',report['keptWholeGroups'],flush=True)

if __name__=='__main__':main()
