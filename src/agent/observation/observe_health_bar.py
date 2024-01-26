import math

import cv2
import numpy as np

from src.common.helpers.helpers import load_from_json_file

MAX_HEALTH = 20
health_bar: dict = load_from_json_file("config/health_bar_conf.json")


def observe_health_bar(screenshot: np.ndarray) -> int:
    """
    Feature extraction of the health value of a minecraft player from the given screenshot.
    @param screenshot: full minecraft screen BGR or RGB
    @return: health value from 0 to 20
    """
    # crop image (keep only important pixels of the health bar), keep original unchanged
    cropped_img: np.ndarray = screenshot[health_bar["top_left"][1]:health_bar["bottom_right"][1],
                             health_bar["top_left"][0]:health_bar["bottom_right"][0]]

    # hsv used to better delineate colors
    img_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    # red hsv mask
    mask0 = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([10, 255, 255]))
    mask1 = cv2.inRange(img_hsv, np.array([170, 0, 0]), np.array([180, 255, 255]))
    mask = mask0 + mask1  # join red masks

    thresh = mask
    first_red_pixel_x = thresh.shape[1] - 1
    pixel_found: bool = False

    # observe health bar from right to left
    for i in reversed(thresh):
        if pixel_found:
            break
        for j in reversed(i):
            if j == 0:  # black
                pixel_found = True
                break
            first_red_pixel_x -= 1

    bar_width = thresh.shape[1]
    percentage = max(0, min(first_red_pixel_x / bar_width, 1))

    # filename = "./tmp/healthThreshold " + str(a) + ".png"
    # cv2.imwrite(filename, thresh)

    return int(math.ceil(MAX_HEALTH * percentage))
