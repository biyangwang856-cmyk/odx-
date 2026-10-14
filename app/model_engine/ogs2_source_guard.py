"""Conservative source-pixel guards for true-photo background selection.

An uncertain vehicle segmentation cannot prove empty road. Plausible detector
boxes stay unavailable for background copying without being certified bodies.
No RGB, camera, reconstruction or original mask is changed.
"""
import math
import numpy as np

def build_source_guard(availability, photo, padding_m=.15):
    availability=np.asarray(availability)
    if availability.dtype!=np.uint8 or availability.ndim!=2 or not np.isin(availability,[0,255]).all():raise ValueError('Require uint8 0/255 source labels')
    h,w=availability.shape
    if (photo.get('height'),photo.get('width'))!=(h,w):raise ValueError('Source-photo coordinate frame mismatch')
    if not math.isfinite(padding_m) or padding_m<0:raise ValueError('Invalid padding')
    guard=availability==0;uncertain=np.zeros_like(guard);records=[];skipped=0
    for proposal in photo.get('uncertainProposals',[]):
        if proposal.get('reason')!='uncertain-body-mask':skipped+=1;continue
        box=np.asarray(proposal.get('box'),float);area=proposal.get('estimatedGroundBoxAreaM2')
        if box.shape!=(4,) or not np.isfinite(box).all() or area is None or not math.isfinite(area) or not .3<=area<=90:skipped+=1;continue
        l,t,r,b=box
        if r<=l or b<=t:raise ValueError('Invalid uncertain vehicle box')
        pitch=math.sqrt(area/((r-l)*(b-t)));pad=int(math.ceil(padding_m/max(pitch,.001)))
        x0,y0,x1,y1=max(0,math.floor(l)-pad),max(0,math.floor(t)-pad),min(w,math.ceil(r)+pad),min(h,math.ceil(b)+pad)
        if x1<=x0 or y1<=y0:skipped+=1;continue
        uncertain[y0:y1,x0:x1]=True;records.append(dict(box=box.tolist(),paddedBox=[x0,y0,x1,y1],samScore=proposal.get('samScore'),reason=proposal['reason']))
    guard|=uncertain
    return guard,dict(uncertainBoxesGuarded=len(records),uncertainBoxPixels=int(uncertain.sum()),skippedUnknownOrImplausibleProposals=skipped,
                      uncertainBoxes=records,emptyRoadCertified=False,sourceRGBChanged=False,sourceMasksChanged=False,
                      uncertaintyUsedAsBackgroundExclusionOnly=True)

def project_source_guard(guard, uv):
    guard=np.asarray(guard);uv=np.asarray(uv)
    if guard.dtype!=np.bool_ or guard.ndim!=2 or uv.ndim!=3 or uv.shape[0]!=2:raise ValueError('Require boolean source guard and 2xHxW source coordinates')
    finite=np.isfinite(uv).all(axis=0);safe=np.where(np.isfinite(uv),uv,-2.)
    x=np.floor(np.clip(safe[0],-2,guard.shape[1])).astype(np.int64);y=np.floor(np.clip(safe[1],-2,guard.shape[0])).astype(np.int64)
    inside=finite&(x>=0)&(y>=0)&(x+1<guard.shape[1])&(y+1<guard.shape[0])
    output=np.ones(uv.shape[1:],bool);xx=x[inside];yy=y[inside]
    # All four bilinear taps must avoid vehicle/uncertain pixels. Nearest-only
    # masks could accept a mixed RGB sample at the edge of a moving vehicle.
    output[inside]=guard[yy,xx]|guard[yy+1,xx]|guard[yy,xx+1]|guard[yy+1,xx+1]
    return output
