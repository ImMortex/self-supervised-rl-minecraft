import os

import cv2
import numpy as np
from PIL import Image

from src.agent.observation.compare_images import mse_imgs
from src.agent.observation.find_image_in_image import find_image_in_image
from src.agent.observation.observe_inventory.libs.observe_inventory_lib import adjust_img_color_format, to_gray
from src.common.helpers.helpers import load_from_json_file

inv_conf: dict = load_from_json_file("./config/inventoryScreenshotNoFullscreenConf.json")
custom_path = "./config/inventoryScreenshotNoFullscreenConfCustom.json"
if os.path.isfile(custom_path):
    inv_conf: dict = load_from_json_file(custom_path)
recipe_book_img: Image = np.array(Image.open("agent_assets/minecraft_items/recipeBook/inventoryRecipeBookButtonIcon.png"))
# Insert 1920x1080 inventory screenshot: Recipe book button is located in the place, in case the recipe book is closed
recipe_book_img = adjust_img_color_format(recipe_book_img)



def gen_sift_features(gray_img):
    """
    returns kp, desc (keypoints and descriptors)
    """
    sift = cv2.SIFT_create(nfeatures=0, sigma=0.8)
    return sift.detectAndCompute(gray_img, None)

sift_recipe_book_kp, sift_recipe_book_desc = gen_sift_features(to_gray(recipe_book_img))


def get_matches_count(query_img_desc, target_img_desc) -> int:
    if target_img_desc is not None and query_img_desc is not None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(target_img_desc, query_img_desc)
        return len(matches)
    return 0


def is_inventory_recipe_book_open(inventory_screenshot: np.ndarray) -> bool:
    """
    Function checks if minecraft inventory recipe book is open using sift algorithm.

    :param inventory_screenshot: 1920x1080 inventory screenshot: Recipe book button is located in the place, in case the recipe book is closed
    :returns: True if minecraft inventory recipe book is open
    """
    inventory_screenshot = adjust_img_color_format(inventory_screenshot)
    slot_img = cut_out_recipe_book_image(inventory_screenshot)

    slot_img = cv2.resize(slot_img, (recipe_book_img.shape[1], recipe_book_img.shape[0]),
                          interpolation=cv2.INTER_AREA)

    # generate SIFT keypoints and descriptors
    query_img_gray = to_gray(slot_img)
    query_img_kp, query_img_desc = gen_sift_features(query_img_gray)

    # cv2.imshow('image', cv2.cvtColor(cv2.hconcat([slot_img, target_img]), cv2.COLOR_BGR2RGB))
    # cv2.waitKey(-1)

    # count matches
    matches: int = get_matches_count(query_img_desc, sift_recipe_book_desc)
    mse: float = mse_imgs(slot_img, recipe_book_img)
    if matches >= 10 and mse < 40000:
        return False  # Recipe book button is located in the place, in case the recipe book is closed
    else:
        return True  # recipe book is open


def cut_out_recipe_book_image(inventory_screenshot: np.ndarray) -> np.ndarray:
    """
    @param inventory_screenshot: screenshot with opened inventory
    @return: cropped img that contains the recipe books default location
    """
    inventory_screenshot = adjust_img_color_format(inventory_screenshot)
    recipe_book_top_left = inv_conf["recipe_book_icon_top_left"]
    recipe_book_bottom_right = inv_conf["recipe_book_icon_bottom_right"]

    slot_img = inventory_screenshot[recipe_book_top_left[1]:recipe_book_bottom_right[1],
               recipe_book_top_left[0]:recipe_book_bottom_right[0]]
    return slot_img


def is_recipe_book_visible(inventory_screenshot: np.ndarray) -> bool:
    """
    @param inventory_screenshot: screenshot with opened inventory
    @return: boolean
    """
    inventory_screenshot = adjust_img_color_format(inventory_screenshot)
    recipe_book_top_left = inv_conf["recipe_book_icon_top_left"]
    recipe_book_bottom_right = inv_conf["recipe_book_icon_bottom_right"]

    cropped_img = inventory_screenshot[recipe_book_top_left[1]:recipe_book_bottom_right[1],
                  recipe_book_top_left[0]:inventory_screenshot.shape[1]]  # search book at both possible locations
    target_found, top_left, bottom_right = find_image_in_image(cropped_img, recipe_book_img, threshold=0.35)
    return target_found


def is_inventory_open(inventory_screenshot: np.ndarray) -> bool:
    return is_recipe_book_visible(inventory_screenshot)
