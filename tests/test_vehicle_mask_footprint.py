import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/lab/mask_vehicle_source_images.py'
spec=importlib.util.spec_from_file_location('vehicle_mask_footprint_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class FootprintTests(unittest.TestCase):
 def setUp(self):
  self.camera=dict(width=2000,height=1000,focal=.5)
  self.shot=dict(rotation=[np.pi,0,0],translation=[0,0,100])
  self.box=[950,450,1050,500]
 def test_known_downward_perspective_area(self):
  self.assertAlmostEqual(module.footprint_area(self.box,self.shot,self.camera,0),50,places=6)
 def test_focal_normalisation_uses_max_dimension_for_portrait(self):
  camera=dict(width=1000,height=2000,focal=.5)
  self.assertAlmostEqual(module.footprint_area([450,950,550,1000],self.shot,camera,0),50,places=6)
 def test_world_translation_does_not_change_area(self):
  shot=dict(rotation=[np.pi,0,0],translation=[-191706,2494789,100])
  self.assertAlmostEqual(module.footprint_area(self.box,shot,self.camera,0),50,places=5)
 def test_upward_and_below_ground_cameras_refused(self):
  shot=dict(rotation=[0,0,0],translation=[0,0,-100])
  self.assertIsNone(module.footprint_area(self.box,shot,self.camera,0))
  self.assertIsNone(module.footprint_area(self.box,self.shot,self.camera,200))
 def test_invalid_focal_and_box_refused(self):
  self.assertIsNone(module.footprint_area(self.box,self.shot,{**self.camera,'focal':0},0))
  self.assertIsNone(module.footprint_area([1,2,1,4],self.shot,self.camera,0))
 def test_proposal_scale_tracks_known_camera_height(self):
  self.assertAlmostEqual(module.footprint_area(self.box,self.shot,self.camera,10),40.5,places=6)

if __name__=='__main__':unittest.main()
