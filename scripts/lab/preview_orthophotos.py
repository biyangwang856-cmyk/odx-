"""Export geospatially identical baseline/candidate views and validity checks."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import rasterio
from rasterio.enums import Resampling

def rgba(src,shape):
    rgb=src.read([1,2,3],out_shape=(3,*shape),resampling=Resampling.average).transpose(1,2,0)
    alpha=src.dataset_mask(out_shape=shape,resampling=Resampling.nearest)
    return np.dstack((rgb,alpha))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    checks=dict(sameGrid=True,alphaDifferentPixels=0,baselineValidPixels=0,candidateValidPixels=0,
                rgbChangedPixels=0,changedPixelFraction=None,totalPixels=0)
    with rasterio.open(a.baseline) as before,rasterio.open(a.candidate) as after:
        if before.crs!=after.crs or before.transform!=after.transform or (before.width,before.height)!=(after.width,after.height):
            raise ValueError('Cannot compare different map grids without an explicit alignment')
        checks.update(crs=str(before.crs),resolutionMeters=list(before.res),width=before.width,height=before.height,
                      bounds=list(before.bounds),baseline=str(a.baseline),candidate=str(a.candidate))
        for _,window in before.block_windows(1):
            old=before.read([1,2,3],window=window);new=after.read([1,2,3],window=window)
            old_a=before.dataset_mask(window=window);new_a=after.dataset_mask(window=window)
            checks['alphaDifferentPixels']+=int(np.count_nonzero(old_a!=new_a))
            checks['baselineValidPixels']+=int(np.count_nonzero(old_a))
            checks['candidateValidPixels']+=int(np.count_nonzero(new_a))
            checks['rgbChangedPixels']+=int(np.count_nonzero(np.any(old!=new,axis=0)&(old_a>0)&(new_a>0)))
            checks['totalPixels']+=int(window.width*window.height)
        checks['changedPixelFraction']=checks['rgbChangedPixels']/max(1,checks['baselineValidPixels'])
        scale=min(1,1800/before.width,1400/before.height)
        shape=(max(1,round(before.height*scale)),max(1,round(before.width*scale)))
        for label,src in [('native-25d',before),('vehicle-ogs2',after)]:
            Image.fromarray(rgba(src,shape),'RGBA').save(a.output/(label+'.png'))
        sheet=Image.new('RGB',(shape[1]*2,shape[0]+42),'white');draw=ImageDraw.Draw(sheet)
        for i,label in enumerate(['native-25d','vehicle-ogs2']):
            image=Image.open(a.output/(label+'.png'))
            sheet.paste(image,(i*shape[1],42),image.getchannel('A'))
            draw.text((i*shape[1]+10,10),label+' (same map extent)',fill='black')
        sheet.save(a.output/'global-comparison.jpg',quality=94)
    checks['limitations']=['RGB changes and protected masks are not counts of unique vehicles or proof of quality gain.',
                           'No DJI orthophoto from the current flight was supplied; this compares the same-flight native2.5D and vehicle projection.',
                           'No resampling, warping or coordinate fitting is used for numerical checks. Preview alone is downsampled.']
    (a.output/'grid-and-pixel-check.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    main()
