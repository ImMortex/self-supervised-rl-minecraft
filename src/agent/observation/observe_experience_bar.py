import cv2
import numpy as np

from src.common.helpers.helpers import load_from_json_file

exp_bar: dict = load_from_json_file("config/experience_bar_conf.json")


def observe_exp_bar(screenshot: np.ndarray) -> float:
    """
    Feature extraction of the experience bar value of a minecraft player from the given screenshot.
    @param screenshot: full minecraft screen BGR or RGB
    @return: experience bar value value from 0 to 100
    """
    # crop image (keep only important pixels of the exp bar), keep original unchanged
    cropped_img: np.ndarray = screenshot[exp_bar["top_left"][1]:exp_bar["bottom_right"][1],
                              exp_bar["top_left"][0]:exp_bar["bottom_right"][0]]
    cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)  # using RGB

    # hsv used to better delineate colors
    img_hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
    # green hsv mask
    mask = cv2.inRange(img_hsv, np.array([40, 0, 0]), np.array([70, 255, 255]))

    first_green_pixel_x = mask.shape[1]
    pixel_found: bool = False

    # observe exp bar from right to left
    for i in reversed(mask):
        if pixel_found:
            break
        for j in reversed(i):
            if j == 255:  # white
                pixel_found = True
                break
            first_green_pixel_x -= 1

    bar_width = mask.shape[1]
    percentage = max(0, min(first_green_pixel_x / bar_width, 1))

    # filename = "./tmp/expMask" + str(time.time()) + ".png"
    # cv2.imwrite(filename, mask)

    return percentage
