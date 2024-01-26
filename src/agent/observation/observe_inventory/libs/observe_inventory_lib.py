import os

import cv2
import numpy as np

from src.common.helpers.helpers import load_from_json_file

inv_conf: dict = load_from_json_file("./config/inventoryScreenshotNoFullscreenConf.json")
custom_path = "./config/inventoryScreenshotNoFullscreenConfCustom.json"
if os.path.isfile(custom_path):
    inv_conf: dict = load_from_json_file(custom_path)
slot_width = inv_conf["slot_width"]
slot_height = inv_conf["slot_height"]

def adjust_img_color_format(inventory_screenshot):
    inventory_screenshot = cv2.cvtColor(inventory_screenshot, cv2.COLOR_BGRA2BGR)  # using BGR instead of RGBA
    return inventory_screenshot


def to_gray(color_img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)


def get_cropped_slot_img(inventory_screenshot, slot_id, slot_positions) -> np.ndarray:
    sp = slot_positions[slot_id]
    # crop image (keep only the slot), keep original unchanged
    slot_img = inventory_screenshot[sp[1]:sp[1] + slot_height, sp[0]:sp[0] + slot_width]
    return slot_img
