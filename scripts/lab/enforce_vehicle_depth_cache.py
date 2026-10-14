"""Hard exclusions in a new copy of completed native depth maps, then native fusion.

No source depth, original RGB, poses or final orthophoto is replaced. This
isolates the post-estimation mask mechanism; semantic correctness is separate.
"""
from pathlib import Path
import argparse,json,hashlib,os,subprocess,time
import numpy as np
from PIL import Image
from inspect_vehicle_mask_depth_ab import read_dmap
from model_engine.ogs2_source_guard import project_source_guard

def masked_depth_bytes(raw,data,excluded):
    if excluded.dtype!=np.bool_ or excluded.shape!=data['depth'].shape:raise ValueError('Invalid depth exclusion frame')
    result=bytearray(raw);count=data['depth'].size;offset=data['depth_offset']
    depth=np.frombuffer(result,dtype='<f4',count=count,offset=offset).reshape(data['depth'].shape)
    before=depth.copy();changed=int((excluded&(depth>0)).sum());depth[excluded]=0;offset+=4*count
    if data['content_flags']&2:
        normal=np.frombuffer(result,dtype='<f4',count=count*3,offset=offset).reshape((*excluded.shape,3));normal[excluded]=0;offset+=12*count
    if data['content_flags']&4:
        confidence=np.frombuffer(result,dtype='<f4',count=count,offset=offset).reshape(excluded.shape);confidence[excluded]=0
    if not np.array_equal(depth[~excluded],before[~excluded]):raise RuntimeError('Unexcluded depths changed')
    return result,changed

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','trial','output']:p.add_argument('--'+name,required=True,type=Path)
    args=p.parse_args();binding=json.loads((args.trial/'input-bindings.json').read_text());statuses=json.loads((args.trial/'dense-status.json').read_text())
    if len(statuses)!=2 or any(s['state']!='completed' for s in statuses):raise ValueError('Require completed A/B')
    args.output.mkdir(exist_ok=False);cache=args.output/'cache';cache.mkdir();rows=[]
    maskdir=Path(binding['sourceMasks']);manifest=json.loads((maskdir/'manifest.json').read_text());byname={p['image']:p for p in manifest['photos']}
    source=args.trial/'masked/dense';files=sorted(source.glob('*.dmap'))
    if len(files)!=binding['subsetPhotos']:raise ValueError('Incomplete depth set')
    for file in files:
        start=time.time();raw=file.read_bytes();sha=hashlib.sha256(raw).hexdigest();data=read_dmap(file);name=Path(data['file_name']).name
        photo=byname[name];maskfile=maskdir/'masks'/(name+'.png')
        if hashlib.sha256(maskfile.read_bytes()).hexdigest()!=photo['maskSha256']:raise ValueError('Original source mask changed')
        with Image.open(maskfile) as image:
            if image.mode!='L' or image.size!=(photo['width'],photo['height']):raise ValueError('Wrong source mask frame')
            labels=np.asarray(image).copy()
        if not np.isin(labels,[0,255]).all():raise ValueError('Wrong mask labels')
        dw,dh=data['depth_size'];yy,xx=np.mgrid[:dh,:dw]
        uv=np.stack([(xx+.5)*photo['width']/dw-.5,(yy+.5)*photo['height']/dh-.5])
        excluded=project_source_guard(labels==0,uv)
        altered,changed=masked_depth_bytes(raw,data,excluded);target=cache/file.name;target.write_bytes(altered)
        check=read_dmap(target)
        if (check['depth'][excluded]>0).any():raise RuntimeError('Hard exclusion incomplete')
        rows.append(dict(image=name,source=str(file),target=str(target),sourceSha256=sha,hardCacheSha256=hashlib.sha256(altered).hexdigest(),
                         excludedDepthPixels=int(excluded.sum()),previouslyPositiveExcludedPixels=changed,positiveExcludedPixelsAfter=0,
                         outsideExclusionDepthsUnchanged=True,semanticExclusionCertified=False,elapsedSeconds=time.time()-start))
        print(name,'removed cached depths',changed,flush=True)
    report=dict(state='cache-completed',photos=rows,sourceTrial=str(args.trial),sourceMasks=str(maskdir),sourceFilesChanged=False,orthophotoBenefitCertified=False)
    reportfile=args.output/'report.json';reportfile.write_text(json.dumps(report,indent=2))
    config=cache/'Densify.ini';config.write_text('Optimize = 0\nEstimation Geometric Iters = 0\nMin Views Filter = 1\n')
    pre=json.loads((args.workspace/'runtime/odx-runtime-gpu/native-loader-python-preflight.json').read_text());binary=args.workspace/'runtime/odx-runtime-gpu/rootfs/code/SuperBuild/install/bin/OpenMVS/DensifyPointCloud'
    scene=Path(next(b for b in binding['branches'] if b['mode']=='masked')['sfm'])/'undistorted/openmvs/scene.mvs'
    command=[pre['loader'],'--library-path',pre['libraryPath'],str(binary),'-i',str(scene),'-o',str(cache/'scene_dense.mvs'),'-w',str(cache),'--dense-config-file',str(config),'--resolution-level','1','--max-resolution','5280','--min-resolution','640','--max-threads','8','--number-views-fuse','2','--geometric-iters','0','--postprocess-dmaps','0','--fusion-mode','0','--cuda-device','-2','--ignore-mask-label','0','--archive-type','3','-v','1']
    report.update(state='fusion-running',nativeCommand=command,depthReestimationRequested=False,gapInterpolationEnabled=False);reportfile.write_text(json.dumps(report,indent=2))
    env={**os.environ,**pre['environment'],'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8'}
    with (args.output/'fusion.log').open('xb') as log:result=subprocess.run(command,cwd=cache,env=env,stdout=log,stderr=subprocess.STDOUT)
    report.update(nativeExitCode=result.returncode,state='fusion-completed' if result.returncode==0 else 'fusion-failed');reportfile.write_text(json.dumps(report,indent=2));result.check_returncode()
    for row in rows:
        for key,hashkey in [('source','sourceSha256'),('target','hardCacheSha256')]:
            with Path(row[key]).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=row[hashkey]:raise RuntimeError('Native fusion changed bound depth cache: '+key)
    report.update(state='completed',allHardDepthCachesPreservedByNativeFusion=True,allOriginalDepthsUnchanged=True)
    reportfile.write_text(json.dumps(report,indent=2));print('Hard mask cache + native fusion completed',flush=True)

if __name__=='__main__':main()
