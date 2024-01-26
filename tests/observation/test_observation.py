import os
import unittest

import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv

from src.agent.observation.observe_inventory.libs.observe_inventory_classify import classify_item_in_slot, \
    load_image_classifier_model
from src.agent.observation.observe_inventory.libs.observe_inventory_lib import adjust_img_color_format, \
    get_cropped_slot_img
from src.agent.observation.observe_inventory.libs.observe_inventory_recipe_book import is_inventory_open
from src.agent.observation.observe_inventory.libs.observe_inventory_slots import get_slot_positions
from src.agent.observation.observe_inventory.observe_inventory import is_inventory_recipe_book_open, \
    read_number_of_items_in_slot, \
    read_inventory
from src.agent.observation.trainers.simple_icon_classifier import get_acc_test_data_1, get_acc_test_data_2, \
    inventory_accuracy, get_classified_inventory_dict
from src.common.observation_keys import inventory_key
from src.common.observation_space import get_initial_state

load_dotenv()
PRETRAINED_ICON_CLASSIFIER = os.getenv("PRETRAINED_ICON_CLASSIFIER")
icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label = \
    load_image_classifier_model(
        model_dir=PRETRAINED_ICON_CLASSIFIER)  # makes one prediction to avoid lag on first use with GPU


class TestObservation(unittest.TestCase):

    def test_observation_inventory_keys(self):
        state = get_initial_state()

        self.assertTrue(len(state[inventory_key]) == 0)

    def test_is_inventory_open__true(self):
        # 1920x1980 img contains inventory with opened recipe book
        screenshot_path = "tests/inventory_img_original/no_full_screen_cropped.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        self.assertTrue(is_inventory_open(screenshot))

    def test_is_inventory_open__true_crafting_recipes(self):
        # 1920x1980 img contains inventory with opened recipe book
        screenshot_path = "tests/inventory_img_original/no_full_screen_cropped crafting recipes.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        self.assertTrue(is_inventory_open(screenshot))

    def test_is_inventory_open__false__night(self):
        # 1920x1980 img contains inventory with opened recipe book
        screenshot_path = "tests/observation/img/test_night.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        self.assertFalse(is_inventory_open(screenshot))

    def test_is_inventory_open__false__jungle(self):
        # 1920x1980 img contains inventory with opened recipe book
        screenshot_path = "tests/observation/img/test_jungle.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        cv2.imwrite("./tmp/tis_inventory_open.png", screenshot)
        self.assertFalse(is_inventory_open(screenshot))

    def test_is_inventory_open__false__horizon(self):
        # 1920x1980 img contains inventory with opened recipe book
        screenshot_path = "tests/observation/img/test_horizon.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        self.assertFalse(is_inventory_open(screenshot))

    def test_is_inventory_recipe_book_open_true(self):
        # 1920x1980 img contains inventory with opened recipe book
        screenshot_path = "tests/inventory_img_original/F2 Fullscreen crafting receipes.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        recipe_book_open: bool = is_inventory_recipe_book_open(screenshot)

        self.assertTrue(recipe_book_open)

    def test_is_inventory_recipe_book_open_false(self):
        # 1920x1980 img contains inventory with closed recipe book
        screenshot_path = "tests/inventory_img_original/F2 Fullscreen.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        recipe_book_open: bool = is_inventory_recipe_book_open(screenshot)

        self.assertFalse(recipe_book_open)

    def test_classify_item_in_slot_oak(self):
        screenshot_path = "tests/observation/img/test_classify_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        sp = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        screenshot = adjust_img_color_format(screenshot)

        # first row
        self.assertEqual("oak_log", classify_item_in_slot(get_cropped_slot_img(screenshot, 0, sp),
                                                          icon_classifier_model, icon_classifier_train_config,
                                                          icon_classifier_int_2_label), "0")
        self.assertEqual("oak_log", classify_item_in_slot(get_cropped_slot_img(screenshot, 1, sp),
                                                          icon_classifier_model, icon_classifier_train_config,
                                                          icon_classifier_int_2_label), "1")

    def test_classify_item_in_slot_birch(self):
        screenshot_path = "tests/observation/img/test_classify_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        sp = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        screenshot = adjust_img_color_format(screenshot)

        # second row
        self.assertEqual("birch_log",
                         classify_item_in_slot(get_cropped_slot_img(screenshot, 9, sp), icon_classifier_model,
                                               icon_classifier_train_config, icon_classifier_int_2_label), "9")
        self.assertEqual("birch_log",
                         classify_item_in_slot(get_cropped_slot_img(screenshot, 10, sp), icon_classifier_model,
                                               icon_classifier_train_config, icon_classifier_int_2_label), "10")

    def test_classify_item_in_slot_dirt(self):
        screenshot_path = "tests/observation/img/test_classify_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        sp = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        screenshot = adjust_img_color_format(screenshot)
        # third row
        self.assertEqual("dirt", classify_item_in_slot(get_cropped_slot_img(screenshot, 18, sp), icon_classifier_model,
                                                       icon_classifier_train_config, icon_classifier_int_2_label), "18")
        self.assertEqual("dirt", classify_item_in_slot(get_cropped_slot_img(screenshot, 19, sp), icon_classifier_model,
                                                       icon_classifier_train_config, icon_classifier_int_2_label), "19")

    def test_classify_item_in_slot_none(self):
        screenshot_path = "tests/observation/img/test_classify_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        sp = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        screenshot = adjust_img_color_format(screenshot)

        # third row
        self.assertEqual("None", classify_item_in_slot(get_cropped_slot_img(screenshot, 20, sp), icon_classifier_model,
                                                       icon_classifier_train_config, icon_classifier_int_2_label), "20")

    def test_read_number_of_items_in_slot_toolbar_digit_1(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # toolbar
        self.assertEqual(1, read_number_of_items_in_slot(screenshot, 27, slot_positions))

    def test_read_number_of_items_in_slot_toolbar_digit_2(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # toolbar
        self.assertEqual(2, read_number_of_items_in_slot(screenshot, 28, slot_positions))

    def test_read_number_of_items_in_slot_toolbar_digit_3(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # toolbar
        self.assertEqual(3, read_number_of_items_in_slot(screenshot, 29, slot_positions))

    def test_read_number_of_items_in_slot_toolbar_digit_4(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # toolbar
        self.assertEqual(4, read_number_of_items_in_slot(screenshot, 30, slot_positions))

    def test_read_number_of_items_in_slot_toolbar_digit_5(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # toolbar
        self.assertEqual(5, read_number_of_items_in_slot(screenshot, 31, slot_positions))

    def test_read_number_of_items_in_slot_toolbar_digit_6(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # toolbar
        self.assertEqual(6, read_number_of_items_in_slot(screenshot, 32, slot_positions))

    def test_read_number_of_items_in_slot_toolbar_digit_7(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # toolbar
        self.assertEqual(7, read_number_of_items_in_slot(screenshot, 33, slot_positions))

    def test_read_number_of_items_in_slot_toolbar_digit_8(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # toolbar
        self.assertEqual(8, read_number_of_items_in_slot(screenshot, 34, slot_positions))

    def test_read_number_of_items_in_slot_toolbar_digit_9(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # toolbar
        self.assertEqual(9, read_number_of_items_in_slot(screenshot, 35, slot_positions))

    def test_read_number_of_items_in_slot_double_digits(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))

        # first row
        self.assertEqual(64, read_number_of_items_in_slot(screenshot, 0, slot_positions))
        self.assertEqual(63, read_number_of_items_in_slot(screenshot, 1, slot_positions))
        self.assertEqual(62, read_number_of_items_in_slot(screenshot, 2, slot_positions))
        self.assertEqual(61, read_number_of_items_in_slot(screenshot, 3, slot_positions))
        self.assertEqual(60, read_number_of_items_in_slot(screenshot, 4, slot_positions))
        self.assertEqual(59, read_number_of_items_in_slot(screenshot, 5, slot_positions))
        self.assertEqual(58, read_number_of_items_in_slot(screenshot, 6, slot_positions))
        self.assertEqual(57, read_number_of_items_in_slot(screenshot, 7, slot_positions))
        self.assertEqual(56, read_number_of_items_in_slot(screenshot, 8, slot_positions))

        # second row
        self.assertEqual(55, read_number_of_items_in_slot(screenshot, 9, slot_positions))
        self.assertEqual(54, read_number_of_items_in_slot(screenshot, 10, slot_positions))
        self.assertEqual(53, read_number_of_items_in_slot(screenshot, 11, slot_positions))
        self.assertEqual(52, read_number_of_items_in_slot(screenshot, 12, slot_positions))
        self.assertEqual(51, read_number_of_items_in_slot(screenshot, 13, slot_positions))
        self.assertEqual(50, read_number_of_items_in_slot(screenshot, 14, slot_positions))
        self.assertEqual(49, read_number_of_items_in_slot(screenshot, 15, slot_positions))
        self.assertEqual(48, read_number_of_items_in_slot(screenshot, 16, slot_positions))
        self.assertEqual(47, read_number_of_items_in_slot(screenshot, 17, slot_positions))

        # third row
        self.assertEqual(16, read_number_of_items_in_slot(screenshot, 18, slot_positions))
        self.assertEqual(32, read_number_of_items_in_slot(screenshot, 19, slot_positions))
        self.assertEqual(10, read_number_of_items_in_slot(screenshot, 20, slot_positions), "20")
        self.assertEqual(11, read_number_of_items_in_slot(screenshot, 21, slot_positions), "21")

    def test_read_number_of_items_in_slot_empty_slot(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # third row
        self.assertEqual(1, read_number_of_items_in_slot(screenshot, 22, slot_positions), "22")  # 1x None

    def test_read_number_of_items_in_slot_white_items(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(screenshot))
        # third row
        self.assertEqual(64, read_number_of_items_in_slot(screenshot, 23, slot_positions), "23")
        self.assertEqual(64, read_number_of_items_in_slot(screenshot, 24, slot_positions), "24")
        self.assertEqual(64, read_number_of_items_in_slot(screenshot, 25, slot_positions), "25")
        self.assertEqual(64, read_number_of_items_in_slot(screenshot, 26, slot_positions), "26")

    def test_is_inventory_recipe_book_open_false(self):
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))

        self.assertFalse(is_inventory_recipe_book_open(screenshot))

    def test_is_inventory_recipe_book_open_true(self):
        screenshot_path = "tests/observation/img/test_recipe_book_open.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))

        self.assertTrue(is_inventory_recipe_book_open(screenshot))

    def test_read_inventory(self):
        """
        Tests multiple single tested features together
        """
        screenshot_path = "tests/observation/img/test_count_items.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        inventory: dict = read_inventory(screenshot, icon_classifier_model, icon_classifier_train_config,
                                         icon_classifier_int_2_label)
        print("\n")
        print(inventory)
        self.assertTrue("oak_log" in inventory)
        self.assertEqual(1113, inventory["oak_log"]["amount"], "oak_log")

    def test_read_toolbar(self):
        """
        Tests multiple single tested features together
        """
        screenshot_path = "tests/observation/img/test_read_toolbar.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        inventory: dict = read_inventory(screenshot, icon_classifier_model, icon_classifier_train_config,
                                         icon_classifier_int_2_label, inventory_open=False)
        print("\n")
        print(inventory)
        self.assertTrue("oak_log" in inventory)
        self.assertEqual(45, inventory["oak_log"]["amount"], "oak_log")

    def test_icon_classifier(self):
        """
        Test validation data
        """
        acc, classified_dict, true_slots = self.get_acc_val_data()
        self.assertTrue(acc >= 0.99, "accuracy " + str(acc) + " is not good enough")

    def get_acc_val_data(self):
        screenshot_path = "agent_assets/icon_classifier_data/validation_icon_classifier.png"
        screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        screenshot = adjust_img_color_format(screenshot)
        true_slots: dict = {'0': 'dirt', '1': 'coarse_dirt', '2': 'podzol', '3': 'rooted_dirt', '4': 'wooden_axe',
                            '5': 'stone_axe', '6': 'wooden_pickaxe', '7': 'stone_pickaxe', '8': 'wooden_shovel',
                            '9': 'oak_fence', '10': 'spruce_fence', '11': 'birch_fence', '12': 'jungle_fence',
                            '13': 'acacia_fence', '14': 'dark_oak_fence', '15': 'mangrove_fence',
                            '16': 'crimson_fence', '17': 'warped_fence', '18': 'oak_log', '19': 'spruce_log',
                            '20': 'birch_log', '21': 'jungle_log', '22': 'acacia_log', '23': 'dark_oak_log',
                            '24': 'mangrove_log', '25': 'stripped_oak_log', '26': 'stripped_spruce_log',
                            '27': 'stone_shovel', '28': 'iron_pickaxe', '29': 'cobblestone', '30': 'mossy_cobblestone',
                            '31': 'stick', '32': 'coal', '33': 'raw_iron',
                            '34': 'iron_ingot', '35': 'diamond'}
        classified_dict = get_classified_inventory_dict(screenshot, icon_classifier_model,
                                                             icon_classifier_train_config,
                                                             icon_classifier_int_2_label)
        acc = inventory_accuracy(classified_dict, true_slots)
        print("test_icon_classifier val data acc: " + str(acc))
        return acc, classified_dict, true_slots

    def test_icon_classifier2(self):
        """
        Test unseen data
        """
        acc = get_acc_test_data_1(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label)

        self.assertTrue(acc >= 0.8, "accuracy " + str(acc) + " is not good enough")


    def test_icon_classifier3(self):
        """
        Test unseen data
        """
        acc = get_acc_test_data_2(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label)
        self.assertTrue(acc >= 0.8, "accuracy " + str(acc) + " is not good enough")

