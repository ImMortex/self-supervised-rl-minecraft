import unittest

import numpy as np
from PIL import Image

from src.agent.observation.observe_hunger_bar import observe_hunger_bar


class TestObservation(unittest.TestCase):

    def test_saturation_20(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/20.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertEqual(20, saturation)

    def test_saturation_0(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/0.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertEqual(0, saturation)

    def test_saturation_1_hunger_effect(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/1_hunger_effect.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertTrue(saturation == 1 or saturation == 2)  # todo: optimize tested function

    def test_saturation_2_hunger_effect(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/2_hunger_effect.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertEqual(2, saturation)

    def test_saturation_4(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/4.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertEqual(4, saturation)

    def test_saturation_6(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/6.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertEqual(6, saturation)

    def test_saturation_10_hunger_effect(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/10_hunger_effect.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertEqual(10, saturation)

    def test_saturation_10_inventory(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/10_inventory.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertEqual(10, saturation)

    def test_saturation_17_inventory(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/17_inventory.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertEqual(17, saturation)

    def test_saturation_18(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/18.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertEqual(18, saturation)

    def test_saturation_19(self):
        screenshot: np.ndarray = np.array(Image.open("tests/observation/img/hunger/19.png"))
        saturation: int = observe_hunger_bar(screenshot)
        self.assertEqual(19, saturation)
