import unittest

import numpy as np

from src.dataloader.transform_functions import class_index_to_one_hot

class TestTransformFunctions(unittest.TestCase):


    def test_class_index_to_one_hot(self):
        index = 2
        one_hot = class_index_to_one_hot(index, 10)
        self.assertTrue(np.alltrue([0, 0, 1.0, 0, 0, 0, 0, 0, 0, 0] == one_hot))
