import math

import cv2
import numpy as np

from src.common.helpers.helpers import load_from_json_file

MAX_SATURATION = 20
hunger_bar: dict = load_from_json_file("config/hunger_bar_conf.json")


def observe_hunger_bar(screenshot: np.ndarray) -> int:
    """
    Feature extraction of the saturation value of a minecraft player from the given screenshot.
    @param screenshot: full minecraft screen BGR or RGB
    @return: saturation value from 0 to 20
    """
    # todo: this solution is only a workaround. Cropping each single hunger symbol could be better
    # crop image (keep only important pixels of the hunger bar), keep original unchanged
    cropped_img: np.ndarray = screenshot[hunger_bar["top_left"][1]:hunger_bar["bottom_right"][1],
                             hunger_bar["top_left"][0]:hunger_bar["bottom_right"][0]]

    # hsv used to better delineate colors
    img_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    # red hsv mask
    mask0 = cv2.inRange(img_hsv, np.array([0, 0, 0]), np.array([10, 255, 255]))
    mask1 = cv2.inRange(img_hsv,  np.array([170, 0, 0]), np.array([180, 255, 255]))
    mask = mask0 + mask1 # join red masks

    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)  # using RGB

    one_color_channel: np.ndarray = cropped_img[:, :, 2]  # extract red channel
    threshold = 200  # min red value

    # lighten up dark hunger bar while inventory view is open (gray similar tones)
    maxv = one_color_channel.max()
    for x in range(0, one_color_channel.shape[0]):
        for y in range(0, one_color_channel.shape[1]):
            if one_color_channel[x, y] >= maxv - 2:
                one_color_channel[x, y] = max(threshold + 1, maxv)
                one_color_channel[x, y] = min(one_color_channel[x, y], 255)

                # match with hsv mask
                if mask[x, y] != 0:
                    one_color_channel[x, y] = 0  # is not the color, set it to black

    # invert black and white
    thresh: np.ndarray = cv2.threshold(one_color_channel, threshold, 255, cv2.THRESH_BINARY_INV)[1]  # (red is black)

    first_red_pixel_x = 0
    pixel_found: bool = False
    # observe health bar from left to right
    for i in thresh:
        if pixel_found:
            break
        for j in i:
            if j == 0:  # black
                pixel_found = True
                break
            first_red_pixel_x += 1

    bar_width = thresh.shape[1]
    value = bar_width - first_red_pixel_x
    percentage = max(0, min(value / bar_width, 1))

    # filename = "./tmp/hungerThreshold " + str(a) + ".png"
    # cv2.imwrite(filename, thresh)

    return int(math.ceil(MAX_SATURATION * percentage))
