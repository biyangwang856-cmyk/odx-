"""Prepare explicit saved-mask evidence for a completed optional experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio as rio
from model_engine.ogs2_deghost import mask_in_window


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    args=parser.parse_args()
    result=args.result.resolve()
    report=json.loads((result/'report.json').read_text())
    cache=Path(report['cache'])
    saved=json.loads((cache/'observations.json').read_text())
    original=json.loads((cache/'dynamic-groups.json').read_text())
    with rio.open(result/'ogs2_orthophoto.tif') as new,rio.open(cache/'ogs2_orthophoto.tif') as old:
        shape=(new.height,new.width)
        x,y=~new.transform * (old.transform.c,old.transform.f)
    with np.load(cache/'observation-masks.npz',allow_pickle=False) as data:
        observations=[{**o,'origin':[o['origin'][0]+round(x),o['origin'][1]+round(y)],
                       'mask':data[f'mask_{i}'].astype(bool)} for i,o in enumerate(saved['observations'])]
    components=json.loads((result/'objects.json').read_text())
    patches=[]
    for record in components:
        if record['state']!='accepted':
            continue
        chosen=record['sourceImage']
        if report['schemaVersion']=='ogs2-road-coherent/v1':
            # Track identifiers belong to the saved native observation cache;
            # road-source vehicle IDs are separate semantic observations.
            group_ids=[group['group'] for group in original if chosen in group['sourceImages']]
        else:
            group_ids=[int(track.removeprefix('track-')) for track in record['requiredTracks']]
        retained=np.zeros(shape,bool)
        discarded=np.zeros(shape,bool)
        for group in original:
            if group['group'] not in group_ids:
                continue
            for i in group['observations']:
                if observations[i]['sourceImage']==chosen:
                    retained |= mask_in_window(observations[i],(0,0),shape)
                else:
                    discarded |= mask_in_window(observations[i],(0,0),shape)
        mask_path=result/f"component-{record['component']}-evaluation.npz"
        np.savez_compressed(mask_path,retained=retained,discarded=discarded)
        def mask(path,key):
            return {'path':str(path),'key':key}
        patches.append({'id':f"component-{record['component']}",'window':record['window'],
            'sourceCache':str(Path(record['sourceCache']).resolve()),'sourceImage':chosen,'sourceIndex':record['sourceIndex'],
            'sourcePhoto':str(Path(report['project'])/'opensfm/undistorted/images'/chosen),
            'hardMask':mask(record['trackMasks'],'hard'),'writeMask':mask(record['trackMasks'],'write'),
            'retainedTrackMask':mask(mask_path,'retained'),'discardedTrackMask':mask(mask_path,'discarded'),
            'measuredChange':mask(record['trackMasks'],'measuredChange')})
    manifest={'blendFormula':'weighted-sum','blendRounding':'nearest','patches':patches}
    path=result/'validation-absolute-manifest.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(manifest,indent=2))
    print(path)


if __name__=='__main__':
    main()
