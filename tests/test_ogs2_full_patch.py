import json
from pathlib import Path

import numpy as np
import pytest

import rasterio as rio
from affine import Affine
from rasterio.enums import ColorInterp

from model_engine.ogs2 import sha256
from model_engine.ogs2_apply_patch import export_patch


def raster(path,array,transform):
    path.parent.mkdir(parents=True,exist_ok=True)
    with rio.open(path,'w',driver='GTiff',width=array.shape[2],height=array.shape[1],count=array.shape[0],
                  dtype=array.dtype,crs='EPSG:32650',transform=transform) as src:
        src.write(array)
        if array.shape[0]==4:
            src.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue,ColorInterp.alpha)
            src.update_tags(scene='retained-test-scene')


def fixture(tmp_path):
    base=tmp_path/'base.tif';patch=tmp_path/'patch';patch.mkdir()
    transform=Affine(.05,0,191000,0,-.05,2494000)
    image=np.full((4,12,14),100,np.uint8);image[3]=255
    raster(base,image,transform)
    part=image[:,3:7,4:9].copy()
    support=np.zeros((1,4,5),np.uint8);support[0,1:3,1:4]=1
    part[:3,support[0]>0]=45
    raster(patch/'ogs2_orthophoto.tif',part,transform*Affine.translation(4,3))
    raster(patch/'write-support.tif',support,transform*Affine.translation(4,3))
    validation=tmp_path/'validation.json'
    pin(base,patch,validation)
    return base,patch,validation,image,part


def pin(base,patch,validation):
    validation.write_text(json.dumps({'checksPassed':True,'allCachesLinkedToActualPhotoHashes':True,
                                     'compositionBase':str(base),'inputs':[
                                     {'path':str(p),'sha256':sha256(p)} for p in
                                     (base,patch/'ogs2_orthophoto.tif',patch/'write-support.tif')]}))


def test_exact_spatial_export_keeps_metadata_and_original_pixels(tmp_path):
    base,patch,validation,original,part=fixture(tmp_path)
    record=export_patch(base,patch,validation,tmp_path/'full')
    expected=original.copy();expected[:,3:7,4:9]=part
    with rio.open(tmp_path/'full/ogs2_orthophoto.tif') as full,rio.open(base) as old:
        np.testing.assert_array_equal(full.read(),expected)
        assert full.crs==old.crs and full.transform==old.transform
        assert full.colorinterp==old.colorinterp
        assert full.tags()['scene']=='retained-test-scene'
    assert record['changedPixels']==6
    assert record['outsideWriteDifferentPixels']==record['alphaDifferentPixels']==0
    assert record['allSceneVehicleIdentityVerified'] is False
    assert sha256(base)==json.loads(validation.read_text())['inputs'][0]['sha256']


def test_stale_pixel_hash_cannot_authorize_publication(tmp_path):
    base,patch,validation,_,part=fixture(tmp_path)
    with rio.open(patch/'ogs2_orthophoto.tif','r+') as src:
        part[0,1,1]+=1;src.write(part)
    with pytest.raises(ValueError,match='validated input has changed'):
        export_patch(base,patch,validation,tmp_path/'full')
    assert not (tmp_path/'full').exists()


@pytest.mark.parametrize('kind',['alpha','undeclared','fractional','resolution'])
def test_consistent_hashes_still_cannot_bypass_physical_invariants(tmp_path,kind):
    base,patch,validation,_,part=fixture(tmp_path)
    path=patch/'ogs2_orthophoto.tif'
    with rio.open(path) as src: transform=src.transform
    if kind=='alpha':part[3,1,1]=0
    if kind=='undeclared':part[0,0,0]=1
    if kind=='fractional':transform*=Affine.translation(.5,0)
    if kind=='resolution':transform*=Affine.scale(2,1)
    raster(path,part,transform)
    pin(base,patch,validation)
    with pytest.raises(ValueError):
        export_patch(base,patch,validation,tmp_path/'full')
    assert not (tmp_path/'full/ogs2_orthophoto.tif').exists()


def test_failed_quality_gate_does_not_create_output(tmp_path):
    base,patch,validation,_,_=fixture(tmp_path)
    record=json.loads(validation.read_text());record['checksPassed']=False
    validation.write_text(json.dumps(record))
    with pytest.raises(ValueError,match='has not passed'):
        export_patch(base,patch,validation,tmp_path/'full')
    assert not (tmp_path/'full').exists()


def test_existing_result_is_never_replaced(tmp_path):
    base,patch,validation,_,_=fixture(tmp_path)
    target=tmp_path/'full';target.mkdir();(target/'sentinel.txt').write_text('keep')
    with pytest.raises(FileExistsError):export_patch(base,patch,validation,target)
    assert (target/'sentinel.txt').read_text()=='keep'
