import importlib.util,sys,struct
from pathlib import Path
import unittest,tempfile,numpy as np
repo=Path(__file__).resolve().parents[1];sys.path.insert(0,str(repo/'app'));sys.path.insert(0,str(repo/'scripts/lab'))
import enforce_vehicle_depth_cache as module
from inspect_vehicle_mask_depth_ab import read_dmap
class DepthExclusionTests(unittest.TestCase):
 def fixture(self):
  name=b'photo.JPG.tif';header=struct.pack('<2sBBIIIIffH',b'DR',15,0,8,6,4,3,0,200,len(name))+name+struct.pack('<II',1,0)+np.eye(3,dtype='<f8').tobytes()*2+np.zeros(3,dtype='<f8').tobytes()
  depth=np.arange(1,13,dtype='<f4').reshape(3,4);normal=np.arange(36,dtype='<f4').reshape(3,4,3);confidence=np.arange(20,32,dtype='<f4').reshape(3,4);views=bytes(range(48))
  raw=header+depth.tobytes()+normal.tobytes()+confidence.tobytes()+views
  with tempfile.NamedTemporaryFile(dir=Path.cwd(),prefix='depth-exclusion-test-',delete=False) as stream:stream.write(raw);path=Path(stream.name)
  try:data=read_dmap(path)
  finally:path.unlink()
  return raw,data
 def test_only_excluded_depth_normal_confidence_changed(self):
  raw,data=self.fixture();mask=np.zeros((3,4),bool);mask[1,2]=True;altered,count=module.masked_depth_bytes(raw,data,mask);off=data['depth_offset']
  self.assertEqual(count,1);self.assertEqual(altered[:off],raw[:off]);self.assertEqual(altered[-48:],raw[-48:])
  depth=np.frombuffer(altered,dtype='<f4',count=12,offset=off).reshape(3,4);self.assertEqual(depth[1,2],0);np.testing.assert_array_equal(depth[~mask],data['depth'][~mask])
  normal=np.frombuffer(altered,dtype='<f4',count=36,offset=off+48).reshape(3,4,3);self.assertTrue((normal[1,2]==0).all())
  confidence=np.frombuffer(altered,dtype='<f4',count=12,offset=off+48+144).reshape(3,4);self.assertEqual(confidence[1,2],0)
  self.assertEqual(raw,self.fixture()[0])
 def test_empty_exclusion_preserves_every_byte(self):
  raw,data=self.fixture();altered,count=module.masked_depth_bytes(raw,data,np.zeros((3,4),bool));self.assertEqual(raw,altered);self.assertEqual(count,0)
 def test_hard_exclusion_idempotent(self):
  raw,data=self.fixture();mask=np.ones((3,4),bool);first,_=module.masked_depth_bytes(raw,data,mask);second,count=module.masked_depth_bytes(first,data,mask);self.assertEqual(first,second);self.assertEqual(count,0)
 def test_wrong_shape_or_nonboolean_labels_rejected(self):
  raw,data=self.fixture()
  for mask in [np.zeros((3,4),np.uint8),np.zeros((4,3),bool)]:
   with self.assertRaises(ValueError):module.masked_depth_bytes(raw,data,mask)
if __name__=='__main__':unittest.main()
