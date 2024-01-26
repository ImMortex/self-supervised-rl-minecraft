import numpy as np


def crop_img(screenshot: np.ndarray, crop_top: int = 0, crop_bottom: int = 0, crop_left: int = 0,
             crop_right: int = 0) -> np.ndarray:
    """
    Makes a copy of an image and returns it cropped. The input image remains unchanged.
    Given are the pixels that are to be cut
    """
    origin_x = 0
    origin_y = 0
    height = screenshot.shape[0]
    width = screenshot.shape[1]
    cropped_img = screenshot[
                  origin_y + crop_top:origin_y + height - crop_bottom,
                  origin_x + crop_left:origin_x + width - crop_right
                  ]
    return cropped_img
