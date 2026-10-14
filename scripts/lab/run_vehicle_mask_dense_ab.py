"""Run independent native OpenMVS dense comparisons with fixed RGB and poses.

The supplied trial contains control/masked exports and input-bindings.json.
Fresh depth directories are mandatory: an old depth cache invalidates a mask
comparison. This CPU pilot does not replace the project's official outputs.
"""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, time

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--trial',required=True,type=Path)
    parser.add_argument('--threads',type=int,default=8)
    args=parser.parse_args()
    if not 1<=args.threads<=16:raise ValueError('Use a bounded pilot thread count')
    bindings=json.loads((args.trial/'input-bindings.json').read_text())
    branches={row['mode']:row for row in bindings['branches']}
    if set(branches)!={'control','masked'}:raise ValueError('Require both bound branches')
    control,masked=branches['control'],branches['masked']
    for key in ['cameraSha256','trackSha256','images']:
        if control[key]!=masked[key]:raise ValueError(f'Uncontrolled A/B difference: {key}')
    pre=json.loads((args.workspace/'runtime/odx-runtime-gpu/native-loader-python-preflight.json').read_text())
    binary=args.workspace/'runtime/odx-runtime-gpu/rootfs/code/SuperBuild/install/bin/OpenMVS/DensifyPointCloud'
    env={**os.environ,**pre['environment'],'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':str(args.threads),'OPENBLAS_NUM_THREADS':str(args.threads)}
    states=[]
    input_hash=sha(args.trial/'input-bindings.json')
    for mode in ['control','masked']:
        row=branches[mode];ud=Path(row['sfm'])/'undistorted'
        for filename,key in [('reconstruction.json','cameraSha256'),('tracks.csv','trackSha256')]:
            if sha(ud/filename)!=row[key]:raise ValueError(f'{mode}: bound {filename} changed')
        if mode=='masked':
            manifest=json.loads((Path(bindings['sourceMasks'])/'manifest.json').read_text())
            if manifest['state']!='completed':raise ValueError('Masks incomplete')
            by_name={record['image']:record for record in manifest['photos']}
            for name in row['images']:
                if sha(ud/'masks'/(name+'.png'))!=by_name[name]['maskSha256']:raise ValueError('Mask input changed')
        work=args.trial/mode/'dense';work.mkdir(exist_ok=False)
        image_hashes={name:sha(ud/'images'/name) for name in row['images']}
        if mode=='masked' and image_hashes!=states[0]['imageSha256']:raise ValueError('RGB differs between branches')
        config=work/'Densify.ini';config.write_text('Optimize = 7\nMin Views Filter = 1\n')
        scene=ud/'openmvs/scene.mvs'
        command=[pre['loader'],'--library-path',pre['libraryPath'],str(binary),'-i',str(scene),'-o',str(work/'scene_dense.mvs'),'-w',str(work),
                 '--dense-config-file',str(config),'--resolution-level','1','--max-resolution','5280','--min-resolution','640',
                 '--max-threads',str(args.threads),'--number-views-fuse','2','--sub-resolution-levels','2','--archive-type','3',
                 '--cuda-device','-2','--ignore-mask-label',str(row['maskLabel']),'-v','1']
        state=dict(mode=mode,state='running',command=command,inputBindingsSha256=input_hash,sceneSha256=sha(scene),
                   imageSha256=image_hashes,startedUnix=time.time(),nativeGpuUsed=False,depthCacheReused=False)
        states.append(state);report=args.trial/'dense-status.json'
        report.write_text(json.dumps(states,indent=2))
        with (work/'run.log').open('xb') as stream:
            result=subprocess.run(command,cwd=work,env=env,stdout=stream,stderr=subprocess.STDOUT)
        state.update(exitCode=result.returncode,elapsedSeconds=time.time()-state['startedUnix'],state='completed' if result.returncode==0 else 'failed')
        state['depthFiles']=len(list(work.glob('*.dmap')))
        state['outputs']=[dict(path=str(f),bytes=f.stat().st_size) for f in sorted(work.glob('scene_dense.*'))]
        report.write_text(json.dumps(states,indent=2));print(mode,state['state'],state['elapsedSeconds'],flush=True)
        result.check_returncode()
        for filename,key in [('reconstruction.json','cameraSha256'),('tracks.csv','trackSha256')]:
            if sha(ud/filename)!=row[key]:raise RuntimeError('Source geometry changed during reconstruction')
    if sha(args.trial/'input-bindings.json')!=input_hash:raise RuntimeError('Input binding changed during comparison')

if __name__=='__main__':main()
