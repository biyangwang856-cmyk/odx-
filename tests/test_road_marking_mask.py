import importlib.util
from pathlib import Path
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/lab/road_background_color_trial.py'
spec=importlib.util.spec_from_file_location('road_background_colour_trial_test',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class MarkingMaskTests(unittest.TestCase):
 def setUp(self):
  self.rgb=np.full((160,250,3),160,np.uint8);self.road=np.ones((160,250),bool)
 def test_bright_grey_asphalt_not_a_whole_marking(self):
  self.assertFalse(module.marking_mask(self.rgb,self.road).any())
 def test_horizontal_white_marking_is_protected(self):
  self.rgb[60:67,50:160]=235
  self.assertTrue(module.marking_mask(self.rgb,self.road)[60:67,50:160].all())
 def test_vertical_pole_not_a_ground_dash(self):
  self.rgb[20:140,80:88]=235
  self.assertFalse(module.marking_mask(self.rgb,self.road).any())
 def test_large_bright_surface_and_outside_road_refused(self):
  self.rgb[60:90,20:230]=235
  self.assertFalse(module.marking_mask(self.rgb,self.road).any())
  self.rgb[:]=160;self.rgb[60:67,50:160]=235;self.road[:]=False
  self.assertFalse(module.marking_mask(self.rgb,self.road).any())

if __name__=='__main__':unittest.main()
