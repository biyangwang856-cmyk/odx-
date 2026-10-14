"""Read native DR depth files and inspect exclusion behavior without editing RGB.

Format reference: OpenMVS scripts/python/MvsUtils.py (loadDMAP). This reads only
the header and depth plane. Counts verify a mechanism, not orthophoto quality.
"""
from pathlib import Path
import argparse,json,struct,hashlib
import numpy as np
from PIL import Image,ImageDraw

def read_dmap(path):
    with Path(path).open('rb') as stream:
        def read(count):
            data=stream.read(count)
            if len(data)!=count:raise ValueError('Truncated native depth file')
            return data
        magic,flags,reserved=struct.unpack('<2sBB',read(4))
        iw,ih,dw,dh=struct.unpack('<IIII',read(16))
        if magic!=b'DR' or not flags&1 or not 0<dw<=iw<=30000 or not 0<dh<=ih<=30000:raise ValueError('Unsupported native depth header')
        dmin,dmax=struct.unpack('<ff',read(8));length,=struct.unpack('<H',read(2));name=read(length).decode('utf8')
        views,=struct.unpack('<I',read(4))
        if not 1<=views<=10000:raise ValueError('Invalid view count')
        view_ids=struct.unpack('<'+'I'*views,read(4*views))
        K=np.frombuffer(read(72),dtype='<f8').reshape(3,3).copy()
        R=np.frombuffer(read(72),dtype='<f8').reshape(3,3).copy()
        C=np.frombuffer(read(24),dtype='<f8').copy()
        depth_offset=stream.tell()
        depth=np.frombuffer(read(4*dw*dh),dtype='<f4').reshape(dh,dw)
        if not np.isfinite(depth).all() or (depth<0).any():raise ValueError('Invalid depth values')
    return dict(file_name=name,image_size=(iw,ih),depth_size=(dw,dh),K=K,R=R,C=C,view_ids=view_ids,depth=depth,range=(dmin,dmax),depth_offset=depth_offset,content_flags=flags)

def interior(mask):
    out=np.zeros_like(mask,dtype=bool)
    if min(mask.shape)<3:return out
    out[1:-1,1:-1]=np.logical_and.reduce([mask[y:y+mask.shape[0]-2,x:x+mask.shape[1]-2] for y in range(3) for x in range(3)])
    return out

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trial',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();status=json.loads((args.trial/'dense-status.json').read_text())
    if len(status)!=2 or any(s['state']!='completed' or s['exitCode']!=0 for s in status):raise ValueError('Both dense cases must finish before final comparison')
    bindings=json.loads((args.trial/'input-bindings.json').read_text());names=bindings['branches'][0]['images']
    maps={}
    for mode in ['control','masked']:
        maps[mode]={}
        for path in sorted((args.trial/mode/'dense').glob('*.dmap')):
            data=read_dmap(path);name=Path(data['file_name']).name
            if name not in names:raise ValueError('Unknown depth image')
            if name in maps[mode]:raise ValueError('Multiple depth maps for one image')
            maps[mode][name]=(path,data)
        if set(maps[mode])!=set(names):raise ValueError('Incomplete per-photo depth set')
    args.output.mkdir(exist_ok=False);records=[]
    sfm=Path(bindings['branches'][0]['sfm'])
    reconstruction=json.loads((sfm/'undistorted/reconstruction.json').read_text())[0]
    for name in names:
        cpath,c=maps['control'][name];mpath,m=maps['masked'][name]
        for key in ['image_size','depth_size']:
            if c[key]!=m[key]:raise ValueError('Uncontrolled depth resolution change')
        for key in ['K','R','C']:
            if not np.allclose(c[key],m[key],rtol=0,atol=1e-9):raise ValueError('Uncontrolled camera change')
        shot=reconstruction['shots'][name];camera=reconstruction['cameras'][shot['camera']]
        if camera['projection_type']!='perspective':raise ValueError('Expected undistorted perspective camera')
        expected=np.eye(3);iw,ih=c['image_size'];dw,dh=c['depth_size']
        expected[0,0]=camera['focal']*max(camera['width'],camera['height'])*iw/camera['width']
        expected[1,1]=camera['focal']*max(camera['width'],camera['height'])*ih/camera['height']
        expected[0,2]=(iw-1)/2;expected[1,2]=(ih-1)/2
        if not np.allclose(c['K'],expected,rtol=0,atol=.01):raise ValueError('Native intrinsic scale/frame differs from calibrated photo; inspect before using masks')
        source_mask=Image.open(Path(bindings['sourceMasks'])/'masks'/(name+'.png')).convert('L')
        if source_mask.size!=(camera['width'],camera['height']):raise ValueError('Wrong mask coordinate frame')
        labels=np.asarray(source_mask)
        if not np.isin(labels,[0,255]).all():raise ValueError('Wrong mask labels')
        excluded=np.asarray(source_mask.resize((dw,dh),Image.Resampling.NEAREST))==0
        core=interior(excluded);available=interior(~excluded)
        cv=c['depth']>0;mv=m['depth']>0
        ratio=lambda x,where:float(x[where].mean()) if where.any() else None
        shared=available&cv&mv;delta=np.abs(c['depth'][shared]-m['depth'][shared])
        record=dict(image=name,depthSize=[dw,dh],exclusionIncludesVehicleProposalsAndInvalidPixels=True,
                    excludedInteriorPixels=int(core.sum()),controlDepthValidFractionInsideExclusion=ratio(cv,core),
                    maskedDepthValidFractionInsideExclusion=ratio(mv,core),
                    availablePixelControlDepthFraction=ratio(cv,available),availablePixelMaskedDepthFraction=ratio(mv,available),
                    availablePixelSemanticsCertified=False,commonAvailableDepthDeltaMedian=float(np.median(delta)) if len(delta) else None,
                    commonAvailableDepthDeltaP90=float(np.percentile(delta,90)) if len(delta) else None,
                    nativeDepthFileControlSha256=hashlib.sha256(cpath.read_bytes()).hexdigest(),
                    nativeDepthFileMaskedSha256=hashlib.sha256(mpath.read_bytes()).hexdigest(),orthophotoBenefitCertified=False)
        records.append(record)
        if any(token in name for token in ['_0016_','_0040_','_0135_','_0406_']):
            with Image.open(sfm/'undistorted/images'/name) as image:rgb=np.asarray(image.convert('RGB').resize((dw,dh),Image.Resampling.BILINEAR))
            panels=[]
            for valid in [cv,mv]:
                panel=rgb.copy();panel[~valid]=(panel[~valid].astype(float)*.25).astype(np.uint8)
                panels.append(Image.fromarray(panel))
            canvas=Image.new('RGB',(dw*2,dh+60),'white');canvas.paste(panels[0],(0,60));canvas.paste(panels[1],(dw,60))
            draw=ImageDraw.Draw(canvas);draw.text((20,18),'CONTROL: bright=valid native depth; dark=missing depth',fill='black')
            draw.text((dw+20,18),'MASKED: bright=valid native depth; dark=missing depth',fill='black')
            canvas.thumbnail((2800,1200));canvas.save(args.output/(name+'.jpg'),quality=94)
    result=dict(sourceTrial=str(args.trial),photos=records,originalRGBChanged=False,
                fullOrthophotoReplaced=False,vehicleSemanticsCertified=False,orthophotoBenefitCertified=False,
                purpose='verify mask/depth mechanism; does not measure ghost-removal success',
                stochasticPatchMatchDifferencesMayExist=True)
    (args.output/'report.json').write_text(json.dumps(result,indent=2));print('Inspected',len(records),'native depth pairs')

if __name__=='__main__':main()
