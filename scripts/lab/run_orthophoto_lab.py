"""Run a fresh photo set through pinned 2.5D and restored vehicle projection.

All outputs are separate from source photos and Git. Native runtime preflight,
source/image identities and restored weights must already be available.
"""
from __future__ import annotations
import argparse,hashlib,json,os,resource,signal,subprocess,sys,time
from pathlib import Path

def write(path,value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)

def rss(pid):
    parents,memory={},{}
    for path in Path('/proc').glob('[0-9]*/status'):
        try:
            data=dict(line.split(':',1) for line in path.read_text().splitlines())
            child=int(path.parent.name)
            parents[child]=int(data['PPid'])
            memory[child]=int(data.get('VmRSS','0 kB').split()[0])*1024
        except (OSError,ValueError,KeyError):
            continue
    owned={pid}
    while True:
        expanded=owned | {p for p,parent in parents.items() if parent in owned}
        if expanded==owned:
            return sum(memory.get(p,0) for p in owned)
        owned=expanded

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,default=Path('/volume/wzhang/mesh-research-workspace'))
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--job',type=Path,required=True)
    p.add_argument('--gpu',default='0')
    p.add_argument('--cpu-cores',type=int,default=16)
    p.add_argument('--memory-gib',type=int,default=128)
    p.add_argument('--wall-hours',type=float,default=36)
    p.add_argument('--pc-quality',choices=['high','ultra'],default='high')
    p.add_argument('--resolution-cm',type=float,default=3)
    p.add_argument('--detach',action='store_true')
    a=p.parse_args()
    workspace=a.workspace.resolve(); dataset=a.dataset.resolve(); job=a.job.resolve()
    if not dataset.is_relative_to(workspace/'datasets') or not job.is_relative_to(workspace/'runs'):
        raise ValueError('Use dedicated inputs and outputs within the persistent workspace')
    if min(a.cpu_cores,a.memory_gib,a.wall_hours,a.resolution_cm)<=0:
        raise ValueError('Positive resource and resolution settings required')
    repo=workspace/'mesh-research'; runtime=workspace/'runtime'
    audit=job/'metadata-audit.json'
    meta=json.loads(audit.read_text())
    if meta['errors'] or not meta['photos']:
        raise ValueError('Metadata audit must succeed before processing')
    pre=json.loads((runtime/'odx-runtime-gpu/native-loader-python-preflight.json').read_text())
    smoke=json.loads((runtime/'odx-runtime-gpu/smoke-native/result.json').read_text())
    if smoke['state']!='passed':
        raise ValueError('Require a successful native GPU preflight')
    model_smoke=json.loads((runtime/'vehicle-model-smoke.json').read_text())
    if model_smoke['state']!='passed':
        raise ValueError('Require real GPU detection/segmentation preflight')
    if (job/'pipeline-status.json').exists() or (a.detach and (job/'pipeline.pid').exists()):
        raise FileExistsError('Preserve existing attempts; use a new job directory')
    if a.detach:
        forwarded=[arg for arg in sys.argv[1:] if arg!='--detach']
        with (job/'pipeline-supervisor.log').open('xb') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*forwarded],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (job/'pipeline.pid').write_text(str(process.pid)+'\n')
        print(json.dumps(dict(pid=process.pid,job=str(job))))
        return
    project=job/'datasets/project'
    project.mkdir(parents=True,exist_ok=False)
    (project/'images').mkdir()
    for record in meta['photos']:
        source=dataset/record['file']
        if not source.is_file():
            raise FileNotFoundError(source)
        (project/'images'/source.name).symlink_to(source)
    names=[record['file'] for record in meta['photos']]
    if len(set(Path(n).name for n in names))!=len(names):
        raise ValueError('Duplicate photo basenames')
    code=runtime/'odx-runtime-gpu/rootfs/code'
    prefix=[pre['loader'],'--library-path',pre['libraryPath'],pre['python']]
    env={**os.environ,**pre['environment'],
         'HOME':str(job),'CUDA_VISIBLE_DEVICES':a.gpu,
         'OMP_NUM_THREADS':str(a.cpu_cores),'OPENBLAS_NUM_THREADS':'1',
         'LAB_MAX_MEMORY_BYTES':str(a.memory_gib*1024**3)}
    settings=['--project-path',str(project.parent),'--max-concurrency',str(a.cpu_cores),
              '--feature-quality','high','--pc-quality',a.pc_quality,'--mesh-size','3000000',
              '--mesh-octree-depth','12','--matcher-type','flann','--matcher-neighbors','8',
              '--sfm-algorithm','incremental','--orthophoto-resolution',str(a.resolution_cm),
              '--orthophoto-compression','DEFLATE','--cog','--skip-3dmodel']
    vehicle_env={**os.environ,'PYTHONPATH':str(repo/'app'),
                 'LD_LIBRARY_PATH':str(runtime/'vehicle-system-libs/rootfs/usr/lib/x86_64-linux-gnu'),
                 'CUDA_VISIBLE_DEVICES':a.gpu,'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8',
                 'NUMBA_NUM_THREADS':'8','MKL_NUM_THREADS':'8','HF_HUB_OFFLINE':'1',
                 'TRANSFORMERS_OFFLINE':'1','TORCH_EXTENSIONS_DIR':str(runtime/'torch-extensions'),
                 'PYTHONFAULTHANDLER':'1'}
    stages=[('sfm',[*prefix,str(repo/'scripts/lab/run_bounded_native.py'),'--code-root',str(code),'--',*settings,'--end-with','opensfm','project'],env),
            ('sfm-geometry',[str(runtime/'vehicle-env/bin/python'),str(repo/'scripts/lab/validate_projected_sfm.py'),'--job',str(job),'--runtime',str(runtime/'odx-runtime-gpu')],vehicle_env),
            ('odx-25d',[*prefix,str(repo/'scripts/lab/run_bounded_native.py'),'--code-root',str(code),'--',*settings,'--end-with','odm_orthophoto','project'],env),
            ('vehicle-ogs2',[str(runtime/'vehicle-env/bin/python'),'-m','model_engine.ogs2','--project',str(project),'--output',str(job/'vehicle-ogs2'),'--models',str(runtime/'vehicle-models'),'--device','cuda:0'],vehicle_env)]
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    plan=dict(schemaVersion='restored-25d-vehicle-run/v1',gitCommit=revision,inputDirectory=str(dataset),photoCount=len(names),
              metadataAuditSha256=hashlib.sha256(audit.read_bytes()).hexdigest(),sourceRevision='eb9fd56ac430fc3735401d88aaef9945ebd00e14',
              parameters=dict(orthophotoMode='2.5d',export3dModel=False,pcQuality=a.pc_quality,featureQuality='high',cameraLens='auto',fixedCameraParams=False,resolutionCm=a.resolution_cm),
              commands=[dict(stage=n,command=c) for n,c,e in stages],imageProvenance=json.loads((runtime/'odx-runtime-gpu/image-provenance.json').read_text()),
              sourceOverlayValidation=json.loads((runtime/'odx-runtime-gpu/source-overlay-validation.json').read_text()),
              inputUploadVerification=str(workspace/'.uploads/upload-20261013/verification.json'),noOldSceneCacheReused=True)
    write(job/'run-plan.json',plan)
    affinity=sorted(os.sched_getaffinity(0))[:a.cpu_cores]
    if len(affinity)<a.cpu_cores:
        raise ValueError('Insufficient permitted CPU cores')
    def limits():
        os.sched_setaffinity(0,affinity); os.nice(10)
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    status=dict(state='running',startedAt=time.time(),stages=[],project=str(project),cpuAffinity=affinity,peakOwnedRssBytes=0)
    started=time.monotonic(); process=None
    try:
        for name,command,environment in stages:
            status['currentStage']=name; write(job/'pipeline-status.json',status)
            with (job/(name+'.log')).open('xb') as log:
                process=subprocess.Popen(command,cwd=repo,env=environment,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=limits)
            status['ownedPid']=process.pid
            while process.poll() is None:
                use=rss(process.pid); status['peakOwnedRssBytes']=max(use,status['peakOwnedRssBytes'])
                status['elapsedSeconds']=time.monotonic()-started; write(job/'pipeline-status.json',status)
                if use>a.memory_gib*1024**3 or status['elapsedSeconds']>a.wall_hours*3600:
                    raise RuntimeError('Own pipeline exceeded resource budget')
                time.sleep(5)
            status['stages'].append(dict(stage=name,returnCode=process.returncode))
            if process.returncode:
                raise RuntimeError(f'{name} failed; inspect {name}.log')
            if name=='sfm':
                recon=json.loads((project/'opensfm/reconstruction.json').read_text())
                count=sum(len(r['shots']) for r in recon)
                write(job/'sfm-check.json',dict(components=len(recon),registeredImages=count,inputImages=len(names),cameras=[r['cameras'] for r in recon]))
                if len(recon)!=1 or count<len(names)*0.95:
                    raise RuntimeError('SfM has multiple components or less than 95% photo registration; review before dense reconstruction')
        from rasterio import open as raster_open
        result=job/'vehicle-ogs2/ogs2_orthophoto.tif'
        with raster_open(result) as image:
            if not image.crs or image.count!=4 or image.width<=0 or image.height<=0:
                raise RuntimeError('Invalid final RGB+alpha georeferenced output')
            output=dict(file=str(result),crs=str(image.crs),width=image.width,height=image.height,resolution=list(image.res),bounds=list(image.bounds))
        status.update(state='succeeded',finishedAt=time.time(),output=output,baseline=str(project/'odm_orthophoto/odm_orthophoto.tif'))
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()
        status.update(state='failed',error=type(error).__name__+': '+str(error),finishedAt=time.time())
        write(job/'pipeline-status.json',status)
        raise
    write(job/'pipeline-status.json',status)
    print(json.dumps(status,indent=2))

if __name__=='__main__':
    main()
