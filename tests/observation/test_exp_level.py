import unittest

import numpy as np
from PIL import Image

from src.agent.observation.observe_experience_level import observe_exp_level

class TestObservation(unittest.TestCase):

    def test_exp_level_0_grass(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_level/0_grass.png"))
        self.assertEqual(0, observe_exp_level(screenshot))

    def test_exp_level_2_grass(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_level/2_grass.png"))
        self.assertEqual(2, observe_exp_level(screenshot))

    def test_exp_level_3_water(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_level/3_water.png"))
        self.assertEqual(3, observe_exp_level(screenshot))

    def test_exp_level_5_grass(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_level/5_grass.png"))
        self.assertEqual(5, observe_exp_level(screenshot))
        
    def test_exp_level_0_inventory(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_level/0_inventory.png"))
        self.assertEqual(0, observe_exp_level(screenshot))

    def test_exp_level_100040_inventory(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_level/100040_inventory.png"))
        self.assertEqual(100040, observe_exp_level(screenshot))


