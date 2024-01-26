import time

import cv2
import numpy as np
import pyautogui
from PIL import Image

from src.common.helpers.helpers import load_from_json_file
from src.common.screen.crop_img import crop_img
from src.common.screen.imgUtils import resize, process_img_gray

mc_screen: dict = load_from_json_file("./config/minecraft_screen_conf.json")


def get_screenshot(dest_w=mc_screen["width"], dest_h=mc_screen["height"],
                   debug=False, grey=False,
                   crop_left: int = mc_screen["margin_left"],
                   crop_top: int = mc_screen["margin_top"],
                   crop_right: int = mc_screen["margin_right"],
                   crop_bottom: int = mc_screen["margin_bottom"],
                   ) -> np.ndarray:
    """
    Makes a screenshot. Crops the img by the given values. Returns resized img as np.ndarray
    :param dest_w: resize width
    :param dest_h: resize height
    :param crop_top: how many pixels should be cut
    :param crop_bottom: how many pixels should be cut
    :param crop_left: how many pixels should be cut
    :param crop_right: how many pixels should be cut
    :param debug: debug mode on/off
    :param grey: grey color image on/off
    :return: np.ndarray
    """

    screenshot: Image = pyautogui.screenshot()

    screenshot = np.array(screenshot)
    screenshot = crop_img(screenshot, crop_top=crop_top, crop_bottom=crop_bottom, crop_left=crop_left,
                          crop_right=crop_right)
    if dest_w > 0 and dest_h > 0:
        screenshot = resize(screenshot, dest_w, dest_h)
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)  # using BGR instead of RGBA

    if grey:
        screenshot = process_img_gray(screenshot)

    if debug:
        # cv2.imshow('window', screenshot)
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #    cv2.destroyAllWindows()
        filename = "./tmp/screenshot " + str(time.time()) + ".png"
        cv2.imwrite(filename, screenshot)

    return screenshot


