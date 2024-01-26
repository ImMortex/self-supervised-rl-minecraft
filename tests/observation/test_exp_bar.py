import unittest

import numpy as np
from PIL import Image

from src.agent.observation.observe_experience_bar import observe_exp_bar


class TestObservation(unittest.TestCase):

    def test_exp_bar_0(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_bar/0%.png"))
        value: float = observe_exp_bar(screenshot)
        self.assertAlmostEqual(0, value, delta=0.1)

    def test_exp_bar_0_inventory(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_bar/0%_inventory.png"))
        value: float = observe_exp_bar(screenshot)
        self.assertAlmostEqual(0, value, delta=0.1)

    def test_exp_bar_29_inventory(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_bar/29%_inventory.png"))
        value: float = observe_exp_bar(screenshot)
        self.assertAlmostEqual(0.29, value, delta=0.1)

    def test_exp_bar_37(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_bar/37%.png"))
        value: float = observe_exp_bar(screenshot)
        self.assertAlmostEqual(0.37, value, delta=0.1)

    def test_exp_bar_83(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_bar/83%.png"))
        value: float = observe_exp_bar(screenshot)
        self.assertAlmostEqual(0.83, value, delta=0.1)

    def test_exp_bar_95(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/exp_bar/95%.png"))
        value: float = observe_exp_bar(screenshot)
        self.assertAlmostEqual(0.96, value, delta=0.1)

