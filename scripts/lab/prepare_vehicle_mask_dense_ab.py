"""Bind an immutable photo subset and completed masks to two native exports."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','project','masks','image-list','output']:p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();names=[n.strip() for n in a.image_list.read_text().splitlines() if n.strip()]
    if not names or len(set(names))!=len(names) or any(Path(n).name!=n for n in names):raise ValueError('Invalid subset')
    manifest=json.loads((a.masks/'manifest.json').read_text());rows={r['image']:r for r in manifest['photos']}
    if manifest['state']!='completed' or not set(names)<=set(rows):raise ValueError('Complete masks required')
    hash_file=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    if hash_file(a.project/'opensfm/undistorted/reconstruction.json')!=manifest['reconstructionSha256']:raise ValueError('Source cameras changed')
    a.output.mkdir(parents=True,exist_ok=False);listing=a.output/'image-list.txt';shutil.copy2(a.image_list,listing)
    branches=[]
    for mode in ['control','masked']:
        sfm=a.output/mode/'sfm';ud=sfm/'undistorted';(ud/'images').mkdir(parents=True);(sfm/'images').mkdir()
        for name in ['config.yaml','camera_models.json','reference_lla.json']:
            source=a.project/'opensfm'/name
            if source.exists():shutil.copy2(source,sfm/name)
        for name in ['reconstruction.json','tracks.csv']:shutil.copy2(a.project/'opensfm/undistorted'/name,ud/name)
        if mode=='masked':(ud/'masks').mkdir()
        for name in names:
            (ud/'images'/name).symlink_to(a.project/'opensfm/undistorted/images'/name)
            raw=a.project/'images'/name.removesuffix('.tif')
            if raw.exists():(sfm/'images'/raw.name).symlink_to(raw)
            if mode=='masked':
                target=ud/'masks'/(name+'.png');shutil.copy2(a.masks/'masks'/(name+'.png'),target)
                if hash_file(target)!=rows[name]['maskSha256']:raise ValueError('Mask input changed')
        branches.append(dict(mode=mode,sfm=str(sfm),cameraSha256=hash_file(ud/'reconstruction.json'),trackSha256=hash_file(ud/'tracks.csv'),images=names,maskLabel=0 if mode=='masked' else -1))
    binding=dict(subsetPhotos=len(names),sourceProject=str(a.project),sourceMasks=str(a.masks),branches=branches,
                 onlyExperimentalDifference='vehicle exclusion masks; RGB/cameras/tracks/settings fixed',originalFullSceneComparisonIsNotUsedAsABControl=True,
                 semanticMasksCertified=False,sourceScriptSha256=hash_file(Path(__file__)))
    (a.output/'input-bindings.json').write_text(json.dumps(binding,indent=2))
    pre=json.loads((a.workspace/'runtime/odx-runtime-gpu/native-loader-python-preflight.json').read_text())
    code='from opensfm.dataset import DataSet; from opensfm.actions import export_openmvs; import sys; export_openmvs.run_dataset(DataSet(sys.argv[1]),sys.argv[2])'
    env={**os.environ,**pre['environment'],'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}
    with (a.output/'export.log').open('xb') as log:
        for row in branches:
            subprocess.run([pre['loader'],'--library-path',pre['libraryPath'],pre['python'],'-c',code,row['sfm'],str(listing)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            scene=Path(row['sfm'])/'undistorted/openmvs/scene.mvs'
            if scene.stat().st_size<=0:raise RuntimeError('Empty scene export')
            print(row['mode'],'exported',scene.stat().st_size,flush=True)

if __name__=='__main__':main()
