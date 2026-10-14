import importlib.util
from pathlib import Path
import struct,tempfile,unittest
import numpy as np

source=Path(__file__).resolve().parents[1]/'scripts/lab/inspect_vehicle_mask_depth_ab.py'
spec=importlib.util.spec_from_file_location('native_depth_reader',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class NativeDepthReaderTests(unittest.TestCase):
    def payload(self,magic=b'DR',width=4,depth=None):
        name=b'photo.JPG.tif';depth=np.arange(12,dtype='<f4').reshape(3,4) if depth is None else depth
        return (struct.pack('<2sBBIIIIffH',magic,1,0,8,6,width,3,0,200,len(name))+name+struct.pack('<III',2,0,1)+
                np.eye(3,dtype='<f8').tobytes()+np.eye(3,dtype='<f8').tobytes()+np.zeros(3,dtype='<f8').tobytes()+depth.tobytes())
    def read(self,payload):
        with tempfile.NamedTemporaryFile(dir=Path.cwd(),prefix='native-depth-test-',suffix='.dmap',delete=False) as stream:
            path=Path(stream.name);stream.write(payload)
        try:return module.read_dmap(path)
        finally:path.unlink()
    def test_known_external_header_with_reference_and_neighbor(self):
        data=self.read(self.payload());self.assertEqual(data['file_name'],'photo.JPG.tif')
        self.assertEqual(data['view_ids'],(0,1));self.assertEqual(data['image_size'],(8,6));self.assertEqual(data['depth_size'],(4,3))
        np.testing.assert_array_equal(data['depth'],np.arange(12,dtype=np.float32).reshape(3,4))
    def test_partial_file_rejected(self):
        with self.assertRaisesRegex(ValueError,'Truncated'):self.read(self.payload()[:-4])
    def test_wrong_magic_or_impossible_resolution_rejected(self):
        for payload in [self.payload(magic=b'XX'),self.payload(width=9)]:
            with self.assertRaises(ValueError):self.read(payload)
    def test_negative_or_nonfinite_depth_rejected(self):
        for value in [-1,np.nan,np.inf]:
            depth=np.ones((3,4),dtype='<f4');depth[1,1]=value
            with self.assertRaises(ValueError):self.read(self.payload(depth=depth))
    def test_mask_boundary_not_counted_as_certified_interior(self):
        mask=np.zeros((7,8),bool);mask[1:6,1:7]=True;got=module.interior(mask)
        self.assertEqual(int(got.sum()),12);self.assertTrue(got[2:5,2:6].all());self.assertFalse(got[1,:].any())

if __name__=='__main__':unittest.main()
