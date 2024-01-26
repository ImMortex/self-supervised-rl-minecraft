import logging
import time
import traceback

import cv2
import minecraft_data
import numpy as np

from src.agent.observation.compare_images import mse_imgs
from src.agent.observation.digit import classify_single_digit_mse, get_bank_of_digits, get_digit_contours, \
    get_digit_mc_digit_image_tresh
from src.agent.observation.observe_inventory.libs.observe_inventory_classify import classify_item_in_slot
from src.agent.observation.observe_inventory.libs.observe_inventory_lib import adjust_img_color_format, \
    get_cropped_slot_img
from src.agent.observation.observe_inventory.libs.observe_inventory_recipe_book import is_inventory_recipe_book_open
from src.agent.observation.observe_inventory.libs.observe_inventory_slots import slot_number_indicator_offset_x, \
    slot_number_indicator_offset_y, slot_number_indicator_width, slot_number_indicator_height, get_slot_positions, \
    get_toolbar_slot_positions
from src.common.env_utils.environment_info import get_minecraft_version_short

inventory_slot_cache: dict = {}


def recognize_number_of_items(number_indicator_img: np.ndarray, save_indicator_to_file: bool = False) -> int:
    """
    Determines the number of items from a Minecraft inventory slot
    @param number_indicator_img: image that contains the digit
    @param save_indicator_to_file: if the number_indicator_img should be saved as black white to file (for classifier)
    @return: int from 1 to 64
    """
    # original code: https://pyimagesearch.com/2017/02/13/recognizing-digits-with-opencv-and-python/ (19.06.2023)

    digit = 1  # default (number of items in minecraft)
    thresh = get_digit_mc_digit_image_tresh(number_indicator_img)  # white font, black background
    # cv2.imshow("Output", thresh)
    # cv2.waitKey(-1)

    if save_indicator_to_file:
        cv2.imwrite("./tmp/digits/" + str(time.time()) + ".png", thresh)

    digit_contours = get_digit_contours(thresh)
    digits = []

    # loop over each of the digits
    for c in digit_contours:
        # extract the digit ROI
        (x, y, w, h) = cv2.boundingRect(c)

        # debug
        output = number_indicator_img.copy()
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 1)
        # cv2.imshow("Output", output)
        # cv2.waitKey(-1)

        digit_img = thresh[y:y + h, x:x + w]
        # cv2.imshow("Output", digit_img)
        # cv2.waitKey(-1)
        classified_digit = classify_single_digit_mse(digit_img, get_bank_of_digits())
        digits.append(classified_digit)

    # minecraft has only 1-64 items in a stack
    if len(digits) == 1:
        digit = digits[0]
    elif len(digits) == 2:
        digit = int(str(digits[0]) + str(digits[1]))  # concat 2 digits

    return min(max(1, digit), 64)


def read_number_of_items_in_slot(inventory_screenshot: np.ndarray, slot_id: int, slot_positions: []):
    inventory_screenshot = adjust_img_color_format(inventory_screenshot)
    sp = slot_positions[slot_id]
    # read number of the stack in the slot (shown from 2 to 64)
    ni_x = sp[0] + slot_number_indicator_offset_x
    ni_y = sp[1] + slot_number_indicator_offset_y
    # crop image (keep only the number indicator), keep original unchanged
    number_indicator_img: np.ndarray = inventory_screenshot[ni_y:ni_y + slot_number_indicator_height,
                                       ni_x:ni_x + slot_number_indicator_width]
    return recognize_number_of_items(number_indicator_img)


def add_item_to_inventory(inventory: dict, name: str, amount: int):
    if name is not None and str(name) != "":
        if amount < 0:
            amount = 0

        if name not in inventory:
            inventory[name] = {
                'amount': amount,
                # 'damage': -1,
                # 'max_damage': -1
            }
        else:
            inventory[name]['amount'] += amount


def count_item(inventory: dict, item_name):
    count: int = 0
    try:
        if item_name is None:
            for name in inventory:
                if "amount" in inventory[name]:
                    count += inventory[name]['amount']
        else:
            if item_name in inventory and "amount" in inventory[item_name]:
                count += inventory[item_name]['amount']

    except Exception as e:
        logging.error(e)
        traceback.print_exc()

    return count


def add_placeholders_to_inventory(inventory: dict):
    """
    Fills sparse inventory dict with missing keys and value placeholders
    """
    short_version = get_minecraft_version_short()
    mc_data: minecraft_data.mod = minecraft_data(short_version)
    items_list: [] = mc_data.items_list
    for item in items_list:
        name: str = item["name"]
        add_item_to_inventory(inventory, name, 0)


def read_inventory(inventory_screenshot: np.ndarray, icon_classifier_model, icon_classifier_train_config,
                   icon_classifier_int_2_label, inventory_open: bool = True) -> dict:
    """
    Returns read inventory
    """
    global inventory_slot_cache
    inventory_screenshot = adjust_img_color_format(inventory_screenshot)
    inventory: dict = {}
    if inventory_open:
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(inventory_screenshot))
    else:
        slot_positions = get_toolbar_slot_positions()

    for slot_id in range(len(slot_positions)):
        try:
            slot_img_key = str(slot_id) + "img"
            label_key = str(slot_id) + "label"

            inventory_screenshot = adjust_img_color_format(inventory_screenshot)
            slot_img: np.ndarray = get_cropped_slot_img(inventory_screenshot, slot_id, slot_positions)

            get_label_from_cache: bool = False
            if slot_img_key not in inventory_slot_cache or inventory_slot_cache[slot_img_key] is None:
                inventory_slot_cache[slot_img_key] = slot_img

            elif mse_imgs(slot_img, inventory_slot_cache[slot_img_key]) < 1.0:
                # if different img, delete img, delete label, save new img
                inventory_slot_cache[slot_img_key] = slot_img
                inventory_slot_cache[label_key] = None

            else:
                # if same img, keep img, get label from cache
                get_label_from_cache = True

            if not get_label_from_cache:
                classified_item = classify_item_in_slot(slot_img, icon_classifier_model, icon_classifier_train_config,
                   icon_classifier_int_2_label)
                inventory_slot_cache[label_key] = None
                inventory_slot_cache[slot_img_key] = None
            else:
                classified_item = inventory_slot_cache[label_key]

            if (label_key not in inventory_slot_cache or inventory_slot_cache[label_key] is None):
                inventory_slot_cache[label_key] = classified_item

            logging.debug("slot " + str(slot_id) + " " + classified_item)
            if classified_item != "None":
                number = read_number_of_items_in_slot(inventory_screenshot, slot_id, slot_positions)
                add_item_to_inventory(inventory=inventory, name=classified_item, amount=number)
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

    return inventory
