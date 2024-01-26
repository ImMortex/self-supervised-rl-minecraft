import cv2
import numpy as np

from src.agent.observation.find_image_in_image import find_image_in_image
from src.common.helpers.helpers import load_from_json_file
from src.common.screen.crop_img import crop_img

death_screen_indicator: np.ndarray = cv2.imread('agent_assets/minecraft_death_screen/death_screen_indicator.png',
                                                cv2.IMREAD_UNCHANGED)
respawn_button: dict = load_from_json_file("./config/death_screen_respawn_button_no_fullscreen_conf.json")
respawn_top_left = respawn_button["top_left"]
respawn_bottom_right = respawn_button["bottom_right"]
mc_screen: dict = load_from_json_file("./config/minecraft_screen_conf.json")


def is_death_screen_shown(screenshot: np.ndarray) -> bool:
    # cut out important image part to reduce calculation time
    screenshot = crop_img(screenshot, crop_left=respawn_top_left[0], crop_top=respawn_top_left[1],
                          crop_right=mc_screen["width"] - respawn_bottom_right[0],
                          crop_bottom=mc_screen["height"] - respawn_bottom_right[1])
    target_found, top_left, bottom_right = find_image_in_image(screenshot, death_screen_indicator)
    return target_found
