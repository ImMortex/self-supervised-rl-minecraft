import os

import PIL
import numpy as np
from PIL import ImageDraw

from src.agent.observation.observe_inventory.libs.observe_inventory_lib import adjust_img_color_format
from src.agent.observation.observe_inventory.libs.observe_inventory_recipe_book import is_inventory_recipe_book_open
from src.common.helpers.helpers import load_from_json_file

inv_conf: dict = load_from_json_file("./config/inventoryScreenshotNoFullscreenConf.json")
custom_path = "./config/inventoryScreenshotNoFullscreenConfCustom.json"
if os.path.isfile(custom_path):
    inv_conf: dict = load_from_json_file(custom_path)

slot_width = inv_conf["slot_width"]
slot_height = inv_conf["slot_height"]
slot_distance_x = inv_conf["slot_distance_x"]
slot_distance_y = inv_conf["slot_distance_y"]
toolbar_slot_distance_x = inv_conf["toolbar_slot_distance_x"]
toolbar_slot_width = inv_conf["toolbar_slot_width"]
toolbar_slot_height = inv_conf["toolbar_slot_height"]
toolbar_first_slot_x = inv_conf["toolbar_first_slot_x"]
toolbar_first_slot_y = inv_conf["toolbar_first_slot_y"]
row_1_offset = inv_conf["row_1_offset"]
row_2_offset = inv_conf["row_2_offset"]
row_3_offset = inv_conf["row_3_offset"]
toolbar_offset = inv_conf["toolbar_offset"]
slot_number_indicator_offset_x = inv_conf["slot_number_indicator_offset_x"]
slot_number_indicator_offset_y = inv_conf["slot_number_indicator_offset_y"]
slot_number_indicator_width = inv_conf["slot_number_indicator_width"]
slot_number_indicator_height = inv_conf["slot_number_indicator_height"]

def get_inventory_slots_test_mapping_image(inventory_screenshot: PIL.Image):
    """
    Returns screenshot with marked positions of inventory slots for testing purpose
    """
    screenshot_array = np.array(inventory_screenshot)
    screenshot_array = adjust_img_color_format(screenshot_array)
    recipe_book_open: bool = is_inventory_recipe_book_open(screenshot_array)
    slot_positions: [] = get_slot_positions(recipe_book_open)
    toolbar_slot_positions: [] = get_toolbar_slot_positions()

    img_draw = ImageDraw.Draw(inventory_screenshot)

    corner_inventory_w_recipes: (int, int) = inv_conf["top_left_corner_inventory_with_recipes_static"]
    inventory_corner: (int, int) = inv_conf["top_left_corner_inventory_static"]
    recipe_book_top_left = inv_conf["recipe_book_icon_top_left"]
    recipe_book_bottom_right = inv_conf["recipe_book_icon_bottom_right"]

    for sp in slot_positions:
        shape_slot = [(sp[0], sp[1]), (sp[0] + slot_width, sp[1] + slot_height)]
        img_draw.rectangle(shape_slot, outline="red")

        ni_x = sp[0] + slot_number_indicator_offset_x
        ni_y = sp[1] + slot_number_indicator_offset_y
        shape_ni = [(ni_x, ni_y), (ni_x + slot_number_indicator_width, ni_y + slot_number_indicator_height)]
        img_draw.rectangle(shape_ni, outline="magenta")

        shape_recipe_book_button = [(recipe_book_top_left[0], recipe_book_top_left[1]),
                                    (recipe_book_bottom_right[0], recipe_book_bottom_right[1])]
        img_draw.rectangle(shape_recipe_book_button, outline="magenta")

        if not recipe_book_open:
            shape_inv_corner = [(inventory_corner[0], inventory_corner[1]),
                                (inventory_corner[0] + 1, inventory_corner[1] + 1)]

        else:
            shape_inv_corner = [(corner_inventory_w_recipes[0], corner_inventory_w_recipes[1]),
                                (corner_inventory_w_recipes[0] + 1, corner_inventory_w_recipes[1] + 1)]

        img_draw.rectangle(shape_inv_corner, outline="magenta")

    for sp in toolbar_slot_positions:
        shape_slot = [(sp[0], sp[1]), (sp[0] + toolbar_slot_width, sp[1] + toolbar_slot_height)]
        img_draw.rectangle(shape_slot, outline="red")

        ni_x = sp[0] + slot_number_indicator_offset_x
        ni_y = sp[1] + slot_number_indicator_offset_y
        shape_ni = [(ni_x, ni_y), (ni_x + slot_number_indicator_width, ni_y + slot_number_indicator_height)]
        img_draw.rectangle(shape_ni, outline="magenta")

    return inventory_screenshot







def get_slot_positions_of_row(first: (int, int), slot_dist_x) -> [(int, int)]:
    """
    Returns positions (top left corner) of all inventory slots in a row
    :param first: position on image of top left corner of the first inventory slot in the row

    :return: array of tuple (int,int)
    """
    slot_positions = []

    for n in range(9):
        x = first[0] + n * (slot_dist_x)
        slot_positions.append((x, first[1]))

    return slot_positions


def get_inventory_slots_positions(crafting_recipes_open: bool = False) -> [(int, int)]:
    print("get_inventory_slots_positions")
    slot_positions = []

    if crafting_recipes_open:
        inventory_corner: (int, int) = inv_conf["top_left_corner_inventory_with_recipes_static"]
    else:
        inventory_corner: (int, int) = inv_conf["top_left_corner_inventory_static"]

    row1_offset = inv_conf["row1_first_slot_offset_from_inventory_corner"]

    # 3 inventory rows
    row1_first_slot: (int, int) = (inventory_corner[0] + row1_offset[0], inventory_corner[1] + row_1_offset)
    row2_first_slot = (row1_first_slot[0], inventory_corner[1] + row_2_offset)
    row3_first_slot = (row1_first_slot[0], inventory_corner[1] + row_3_offset)

    slot_positions += get_slot_positions_of_row(row1_first_slot, slot_distance_x)
    slot_positions += get_slot_positions_of_row(row2_first_slot, slot_distance_x)
    slot_positions += get_slot_positions_of_row(row3_first_slot, slot_distance_x)

    # toolbar
    first_slot_toolbar: (int, int) = (row1_first_slot[0], inventory_corner[1] + toolbar_offset)

    slot_positions += get_slot_positions_of_row(first_slot_toolbar, slot_distance_x)
    return slot_positions

# positions saved for frequently use
slot_positions_if_recipe_book_closed: [(int, int)] = get_inventory_slots_positions(crafting_recipes_open=False)
slot_positions_if_recipe_book_opened: [(int, int)] = get_inventory_slots_positions(crafting_recipes_open=True)
toolbar_slot_positions: [(int, int)] = get_inventory_slots_positions(crafting_recipes_open=False)

def get_toolbar_slots_positions() -> [(int, int)]:
    print("get_inventory_slots_positions")
    slot_positions = []

    # toolbar
    first_slot_toolbar: (int, int) = (toolbar_first_slot_x, toolbar_first_slot_y)

    slot_positions += get_slot_positions_of_row(first_slot_toolbar, toolbar_slot_distance_x)
    return slot_positions

# positions saved for frequently use
slot_positions_if_recipe_book_closed: [(int, int)] = get_inventory_slots_positions(crafting_recipes_open=False)
slot_positions_if_recipe_book_opened: [(int, int)] = get_inventory_slots_positions(crafting_recipes_open=True)
toolbar_slot_positions: [(int, int)] = get_toolbar_slots_positions()

def get_slot_positions(crafting_recipes_open: bool = False) -> [(int, int)]:
    if crafting_recipes_open:
        return slot_positions_if_recipe_book_opened
    else:
        return slot_positions_if_recipe_book_closed

def get_toolbar_slot_positions() -> [(int, int)]:
    return toolbar_slot_positions
