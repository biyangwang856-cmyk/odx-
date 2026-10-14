"""Prepare vehicle exclusion masks in original undistorted-photo coordinates.

RGB files and SfM poses are read only. Masks are proposals for an independent
dense-reconstruction comparison, not proof of completely static background.
"""
from pathlib import Path
import argparse, hashlib, json, math, time
import cv2
import numpy as np
import rasterio as rio
from PIL import Image
from model_engine.ogs2 import Detector, Config, reliable_mask

def footprint_area(box, shot, camera, ground_z):
    focal=camera['focal']*max(camera['width'],camera['height'])
    box=np.asarray(box,float)
    if not np.isfinite(focal) or focal<=0 or box.shape!=(4,) or not np.isfinite(box).all() or box[2]<=box[0] or box[3]<=box[1]:return None
    cx,cy=(np.array([camera['width'],camera['height']])-1)/2
    l,t,r,b=box
    pixels=np.array([[l,t],[r,t],[r,b],[l,b]],float)
    rays=np.column_stack(((pixels[:,0]-cx)/focal,(pixels[:,1]-cy)/focal,np.ones(4)))
    rotation=cv2.Rodrigues(np.asarray(shot['rotation'],float))[0]
    centre=-rotation.T@np.asarray(shot['translation'],float)
    directions=rays@rotation
    if (directions[:,2]>=-1e-6).any():return None
    distance=(ground_z-centre[2])/directions[:,2]
    if (distance<=0).any():return None
    xy=centre[:2]+directions[:,:2]*distance[:,None]
    xy-=xy.mean(axis=0)
    area=.5*abs(np.sum(xy[:,0]*np.roll(xy[:,1],-1)-xy[:,1]*np.roll(xy[:,0],-1)))
    return float(area)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',required=True,type=Path)
    parser.add_argument('--image-list',required=True,type=Path)
    parser.add_argument('--models',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--tile-size',type=int,default=0,help='Optional overlapping high-detail detection tiles; masks remain in full-photo coordinates')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    maskdir=args.output/'masks';maskdir.mkdir();preview=args.output/'previews';preview.mkdir()
    file=args.project/'opensfm/undistorted/reconstruction.json'
    reconstruction_bytes=file.read_bytes()
    reconstruction_hash=hashlib.sha256(reconstruction_bytes).hexdigest()
    reconstruction=json.loads(reconstruction_bytes)
    if len(reconstruction)!=1:raise ValueError('Require one bound reconstruction')
    reconstruction=reconstruction[0]
    names=[n.strip() for n in args.image_list.read_text().splitlines() if n.strip()]
    if len(set(names))!=len(names):raise ValueError('Duplicate images')
    for name in names:
        if name not in reconstruction['shots'] or Path(name).name!=name:raise ValueError('Unknown or unsafe photo name')
    z=[p['coordinates'][2] for p in reconstruction['points'].values()]
    ground_z=float(np.percentile(z,10))
    detector=Detector(args.models,args.device,Config())
    source_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch=detector.torch;records=[];started=time.time()
    for i,name in enumerate(names):
        path=args.project/'opensfm/undistorted/images'/name
        shot=reconstruction['shots'][name];camera=reconstruction['cameras'][shot['camera']]
        if camera['projection_type']!='perspective':raise ValueError('Require calibrated undistorted perspective images')
        with rio.open(path) as source:
            rgb=source.read([1,2,3]).transpose(1,2,0);valid=source.dataset_mask()>0
        if rgb.dtype!=np.uint8 or rgb.shape[:2]!=(camera['height'],camera['width']):raise ValueError('Photo-camera size/dtype mismatch')
        h,w=valid.shape;body=np.zeros((h,w),bool);exclusion=np.zeros((h,w),bool)
        def detect(view,offset=(0,0),internal_tile=False):
            vh,vw=view.shape[:2]
            inputs=detector.processor(images=Image.fromarray(view),text='car. truck. bus.',return_tensors='pt').to(args.device)
            with torch.inference_mode():
                outputs=detector.model(**inputs)
                result=detector.processor.post_process_grounded_object_detection(outputs,inputs.input_ids,box_threshold=.25,text_threshold=.2,target_sizes=[(vh,vw)])[0]
            found=[]
            for box,score in zip(result['boxes'].cpu().numpy(),result['scores'].cpu().numpy()):
                l,t,r,b=box
                ox,oy=offset
                # Full-frame proposals remain available for very large objects.
                # Internal tile cuts cannot masquerade as complete detections.
                if internal_tile and ((ox>0 and l<8) or (oy>0 and t<8) or (ox+vw<w and r>vw-8) or (oy+vh<h and b>vh-8)):continue
                found.append((box+np.array([ox,oy,ox,oy]),float(score)))
            return found
        found=detect(rgb)
        if args.tile_size:
            if args.tile_size<640:raise ValueError('Detection tiles must preserve useful context')
            size=args.tile_size;stride=size//2
            xs=sorted(set([0,max(0,w-size),*range(0,max(1,w-size),stride)]))
            ys=sorted(set([0,max(0,h-size),*range(0,max(1,h-size),stride)]))
            for y0 in ys:
                for x0 in xs:found.extend(detect(rgb[y0:y0+size,x0:x0+size],(x0,y0),True))
        proposals=[];uncertain=[]
        for box,score in found:
            area=footprint_area(box,shot,camera,ground_z)
            record=dict(box=box.tolist(),score=float(score),estimatedGroundBoxAreaM2=area)
            if area is None or not .3<=area<=90:uncertain.append({**record,'reason':'implausible-or-unknown-ground-footprint'});continue
            proposals.append(record)
        kept=[]
        for p in sorted(proposals,key=lambda p:-p['score']):
            a=np.array(p['box']);duplicate=False
            for q in kept:
                b=np.array(q['box']);intersection=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod();union=np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
                if intersection/max(union,1)>.45:duplicate=True;break
            if not duplicate:kept.append(p)
        accepted=[];gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
        for p in kept:
            l,t,r,b=np.rint(p['box']).astype(int);l=max(0,l);t=max(0,t);r=min(w,r);b=min(h,b)
            if r-l<2 or b-t<2:uncertain.append({**p,'reason':'incomplete-pixel-box'});continue
            pitch=math.sqrt(p['estimatedGroundBoxAreaM2']/max(1,(r-l)*(b-t)))
            margin=max(16,min(150,int(math.ceil(.6/max(pitch,.001)))))
            x0,y0,x1,y1=max(0,l-margin),max(0,t-margin),min(w,r+margin),min(h,b+margin)
            local=rgb[y0:y1,x0:x1];localbox=np.array([l-x0,t-y0,r-x0,b-y0],float)
            with torch.inference_mode():
                detector.sam.set_image(local);masks,scores,_=detector.sam.predict(box=localbox,multimask_output=False)
            mask=masks.reshape(-1,*local.shape[:2])[0]>0;score=float(np.ravel(scores)[0]);p['samScore']=score
            if not reliable_mask(mask,localbox,score,Config()):uncertain.append({**p,'reason':'uncertain-body-mask'});continue
            full=np.zeros((h,w),bool);full[y0:y1,x0:x1]=mask;body|=full
            # A finite local shadow candidate; completeness remains unproven.
            radius=max(20,min(500,int(math.ceil(3/max(pitch,.001)))))
            sx0,sy0,sx1,sy1=max(0,l-radius),max(0,t-radius),min(w,r+radius),min(h,b+radius)
            localbody=full[sy0:sy1,sx0:sx1];g=gray[sy0:sy1,sx0:sx1]
            ring=(cv2.dilate(localbody.astype(np.uint8),np.ones((25,25),np.uint8))>0)&~localbody
            road_tone=float(np.median(g[ring])) if ring.any() else 145.
            dark=g<min(120.,road_tone*.78)
            joined=dark|(cv2.dilate(localbody.astype(np.uint8),np.ones((7,7),np.uint8))>0)
            count,labels=cv2.connectedComponents(joined.astype(np.uint8));touch=np.unique(labels[localbody]);touch=touch[touch>0]
            shadow=np.isin(labels,touch)
            pad=max(3,min(61,2*int(math.ceil(.15/max(pitch,.001)))+1))
            protected=cv2.dilate((shadow|localbody).astype(np.uint8),np.ones((pad,pad),np.uint8))>0
            exclusion[sy0:sy1,sx0:sx1]|=protected
            p.update(bodyPixels=int(mask.sum()),shadowCandidatePixels=int((shadow&~localbody).sum()),approximatePixelPitchM=pitch,sourceBorderTouches=bool(l<=2 or t<=2 or r>=w-2 or b>=h-2));accepted.append(p)
        fraction=float(exclusion.sum()/exclusion.size)
        if fraction>.30:raise RuntimeError(f'Excessive mask coverage for {name}; preserve attempt and inspect')
        mask=np.where(valid&~exclusion,255,0).astype(np.uint8)
        maskpath=maskdir/(name+'.png');Image.fromarray(mask).save(maskpath)
        ann=rgb.copy();ann[exclusion]=(ann[exclusion]*.45+np.array([255,0,0])*.55).astype(np.uint8);image=Image.fromarray(ann);image.thumbnail((1600,1200));image.save(preview/(name+'.jpg'),quality=92)
        record=dict(image=name,mask=str(maskpath),width=w,height=h,acceptedBodies=len(accepted),excludedPixelFraction=fraction,bodyPixels=int(body.sum()),exclusionPixels=int(exclusion.sum()),objects=accepted,uncertainProposals=uncertain,maskSha256=hashlib.sha256(maskpath.read_bytes()).hexdigest(),semanticsCertified=False,shadowCompletenessCertified=False,rgbChanged=False,approximateGroundZUsedOnlyForProposalScale=ground_z)
        records.append(record);(args.output/'manifest.json').write_text(json.dumps(dict(reconstructionSha256=reconstruction_hash,sourceScriptSha256=source_hash,detectionTileSize=args.tile_size,photos=records,state='running',sourcePixelFrame='existing undistorted-photo',elapsedSeconds=time.time()-started),indent=2))
        print(f'{i+1}/{len(names)} {name}: bodies={len(accepted)} excluded={fraction:.3%}',flush=True)
    if hashlib.sha256(file.read_bytes()).hexdigest()!=reconstruction_hash:raise RuntimeError('Camera reconstruction changed during preprocessing')
    manifest=json.loads((args.output/'manifest.json').read_text());manifest.update(state='completed',elapsedSeconds=time.time()-started,inputPhotos=len(names),maskLabelConvention={'0':'excluded vehicle/body-shadow candidate or invalid pixel','255':'available pixel; not semantic static proof'})
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
