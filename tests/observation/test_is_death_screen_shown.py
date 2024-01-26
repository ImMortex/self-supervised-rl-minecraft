import unittest

import numpy as np
from PIL import Image

from src.agent.observation.observe_death_screen import is_death_screen_shown

class TestObservation(unittest.TestCase):

    def test_is_death_screen_shown__not_found(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/test_count_items.png"))
        target_found = is_death_screen_shown(screenshot)
        self.assertFalse(target_found)

    def test_is_death_screen_shown__cave(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/test_death_cave.png"))
        target_found = is_death_screen_shown(screenshot)
        self.assertTrue(target_found)

    def test_is_death_screen_shown__drowned(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/test_death_drowned.png"))
        target_found = is_death_screen_shown(screenshot)
        self.assertTrue(target_found)

    def test_is_death_screen_shown__forest1(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/test_death_forest.png"))
        target_found = is_death_screen_shown(screenshot)
        self.assertTrue(target_found)

    def test_death_screen_found__forest2(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/test_death_forest2.png"))
        target_found = is_death_screen_shown(screenshot)
        self.assertTrue(target_found)
