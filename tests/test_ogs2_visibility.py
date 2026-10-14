# Optional OGS2 environment; the base CI does not install algorithm runtimes.
# ruff: noqa: E402
import numpy as np
import pytest
pytest.importorskip('numba')
pytest.importorskip('scipy')
rio = pytest.importorskip('rasterio')
from scipy import ndimage
from model_engine.ogs2_visibility import ray_visible


def reference(points, position, surface, transform, tolerance):
    delta = np.asarray(position)[:,None,None]-points
    horizontal = np.hypot(delta[0],delta[1])
    step = max(abs(transform.a),abs(transform.e))
    valid = np.isfinite(points[2])
    for distance in np.geomspace(step,max(step*2,40),24):
        sample = points+delta*np.minimum(distance/np.maximum(horizontal,step),.95)
        j,i = (~transform)*(sample[0],sample[1])
        terrain = ndimage.map_coordinates(surface,[i-.5,j-.5],order=1,
                                         mode='constant',cval=np.nan,prefilter=False)
        valid &= np.isfinite(terrain)&(terrain<=sample[2]+tolerance)
    return valid


@pytest.mark.parametrize('unknown',[False,True])
def test_compiled_rays_match_reference_with_wall_and_missing_height(unknown):
    transform=rio.transform.from_origin(191700,2494800,.1,.1)
    surface=np.zeros((500,500),np.float32)
    surface[:,250:255]=10
    if unknown:
        surface[100:200,100:200]=np.nan
    x,y=transform*np.meshgrid(np.arange(25)+50.25,np.arange(25)+190.75)
    points=np.stack([x,y,np.zeros_like(x)])
    position=np.array([191740,2494780,25.])
    step=.1
    actual=ray_visible(points,position,surface,np.array(list(~transform)[:6]),
                       np.geomspace(step,40,24),.35,step)
    np.testing.assert_array_equal(actual,reference(points,position,surface,transform,.35))
