import unittest

import numpy as np
from PIL import Image

from src.agent.observation.observe_health_bar import observe_health_bar


class TestObservation(unittest.TestCase):

    def test_health_0(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/health/0.png"))
        health: int = observe_health_bar(screenshot)
        self.assertEqual(0, health)

    def test_health_1(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/health/1.png"))
        health: int = observe_health_bar(screenshot)
        self.assertEqual(1, health)

    def test_health_2(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/health/2.png"))
        health: int = observe_health_bar(screenshot)
        self.assertEqual(2, health)

    def ttest_health_15(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/health/15.png"))
        health: int = observe_health_bar(screenshot)
        self.assertEqual(15, health)

    def test_health_16(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/health/16.png"))
        health: int = observe_health_bar(screenshot)
        self.assertEqual(16, health)

    def test_health_17(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/health/17.png"))
        health: int = observe_health_bar(screenshot)
        self.assertEqual(17, health)

    def test_health_18(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/health/18.png"))
        health: int = observe_health_bar(screenshot)
        self.assertEqual(18, health)

    def test_health_19(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/health/19.png"))
        health: int = observe_health_bar(screenshot)
        self.assertEqual(19, health)

    def test_health_20(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/health/20.png"))
        health: int = observe_health_bar(screenshot)
        self.assertEqual(20, health)
