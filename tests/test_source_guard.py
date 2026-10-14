import importlib.util
from pathlib import Path
import unittest,numpy as np
p=Path(__file__).resolve().parents[1]/'app/model_engine/ogs2_source_guard.py';s=importlib.util.spec_from_file_location('source_guard',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class SourceGuardTests(unittest.TestCase):
 def setUp(self):
  self.labels=np.full((100,200),255,np.uint8);self.photo=dict(width=200,height=100,uncertainProposals=[])
 def test_low_sam_quality_is_not_empty_background(self):
  self.photo['uncertainProposals']=[dict(reason='uncertain-body-mask',box=[30,20,60,50],estimatedGroundBoxAreaM2=9,samScore=.715)]
  g,r=m.build_source_guard(self.labels,self.photo);self.assertTrue(g[20:50,30:60].all());self.assertEqual(r['uncertainBoxesGuarded'],1);self.assertFalse(r['emptyRoadCertified']);self.assertTrue((self.labels==255).all())
 def test_known_exclusion_retained_but_giant_proposals_not_whole_photo(self):
  self.labels[10:20,10:20]=0;self.photo['uncertainProposals']=[dict(reason='implausible-or-unknown-ground-footprint',box=[0,0,200,100],estimatedGroundBoxAreaM2=500)]
  g,r=m.build_source_guard(self.labels,self.photo);self.assertEqual(g.sum(),100);self.assertEqual(r['skippedUnknownOrImplausibleProposals'],1)
 def test_invalid_mask_labels_or_coordinate_frame_rejected(self):
  with self.assertRaises(ValueError):m.build_source_guard(np.ones((100,200),np.uint8),self.photo)
  with self.assertRaises(ValueError):m.build_source_guard(self.labels,{**self.photo,'width':201})
 def test_partial_outside_uncertain_box_safely_clipped(self):
  self.photo['uncertainProposals']=[dict(reason='uncertain-body-mask',box=[-2,-5,12,10],estimatedGroundBoxAreaM2=2)]
  g,r=m.build_source_guard(self.labels,self.photo);self.assertTrue(g[:10,:12].all())
 def test_bilinear_vehicle_tap_not_hidden_by_nearest_pixel(self):
  g=np.zeros((5,5),bool);g[2,2]=True;uv=np.array([[[1.1,3.1]],[[1.1,3.1]]])
  projected=m.project_source_guard(g,uv);self.assertTrue(projected[0,0]);self.assertFalse(projected[0,1])
 def test_missing_source_coverage_fails_closed(self):
  uv=np.array([[[np.nan,-1,4]],[[2,2,2]]],float);self.assertTrue(m.project_source_guard(np.zeros((5,5),bool),uv).all())
 def test_known_integer_source_pixel_uses_same_xy_convention(self):
  g=np.zeros((9,9),bool);g[2,5]=True;uv=np.array([[[5,2]],[[2,5]]],float);v=m.project_source_guard(g,uv);self.assertTrue(v[0,0]);self.assertFalse(v[0,1])
if __name__=='__main__':unittest.main()
