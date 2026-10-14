import unittest
import numpy as np
from model_engine.ogs2_road_color import fit_road_color_offset

class RoadColorTests(unittest.TestCase):
    def setUp(self):
        self.a=np.full((128,160,3),140,np.uint8)
        self.b=np.full_like(self.a,146)
        self.mask=np.ones(self.a.shape[:2],bool)
    def test_known_offset_and_actual_uint8_values(self):
        r=fit_road_color_offset(self.a,self.b,self.mask)
        self.assertTrue(r.accepted,r.report)
        np.testing.assert_array_equal(r.apply(self.b),self.a)
        self.assertEqual(r.report['afterP90'],0)
        self.assertFalse(r.report['semanticsCertified'])
    def test_object_pixels_do_not_fit_offset(self):
        self.a[20:90,30:110]=20;self.mask[20:90,30:110]=False
        r=fit_road_color_offset(self.a,self.b,self.mask)
        self.assertTrue(r.accepted)
        np.testing.assert_array_equal(r.offset,[-6,-6,-6])
    def test_all_outer_outliers_remain_in_validation(self):
        y,x=np.mgrid[:128,:160];outer=((x//16+2*(y//16))%4==0)
        self.a[outer]=30
        r=fit_road_color_offset(self.a,self.b,self.mask)
        self.assertFalse(r.accepted)
        self.assertEqual(r.report['reason'],'independent-colour-holdout-failed')
        self.assertGreater(r.report['afterP90'],100)
    def test_large_offset_refused(self):
        self.b[:]=200;r=fit_road_color_offset(self.a,self.b,self.mask)
        self.assertFalse(r.accepted)
        with self.assertRaises(ValueError):r.apply(self.b)
    def test_no_support_refused(self):
        self.mask[:]=False;self.assertFalse(fit_road_color_offset(self.a,self.b,self.mask).accepted)
    def test_clipping_is_explicit_and_no_uint8_underflow(self):
        r=fit_road_color_offset(self.a,self.b,self.mask)
        dark=np.zeros_like(self.b);np.testing.assert_array_equal(r.apply(dark),dark)
    def test_shape_type_and_unbounded_config_refused(self):
        for params in [dict(max_offset=20),dict(cell_size=True),dict(sample_step=0)]:
            with self.assertRaises(ValueError):fit_road_color_offset(self.a,self.b,self.mask,**params)
        with self.assertRaises(ValueError):fit_road_color_offset(self.a,self.b,self.mask.astype(np.uint8))

if __name__=='__main__':unittest.main()
