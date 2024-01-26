import numpy as np


def adjust_inv_item_background(img: np.ndarray):
    # Adjust background color of slot_img
    background_color = img[0, 0]
    replacement_color = (0, 0, 0)
    img[(img == background_color).all(axis=-1)] = replacement_color
    return img
