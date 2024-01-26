import cv2
import numpy as np

from src.agent.observation.observe_experience_bar import observe_exp_bar
from src.agent.observation.observe_experience_level import observe_exp_level
from src.agent.observation.observe_health_bar import observe_health_bar
from src.agent.observation.observe_inventory.observe_inventory import read_inventory
from src.common.observation_keys import health_key, experience_key, level_key, \
    view_key, POV_WIDTH, POV_HEIGHT, inventory_key, inventory_open_key
from src.common.observation_space import get_initial_state
from src.common.screen.imgUtils import resize


class Observation:

    def __init__(self, icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label):
        self.icon_classifier_model = icon_classifier_model
        self.icon_classifier_train_config = icon_classifier_train_config
        self.icon_classifier_int_2_label = icon_classifier_int_2_label

        self.state = get_initial_state()

    def get_actual_state(self) -> dict:
        return self.state

    def set_inventory_status(self, is_inventory_open: bool):
        self.state[inventory_open_key] = is_inventory_open

    def process_inventory_screenshot(self, inventory_screenshot: np.ndarray, inventory_open: bool):
        """
        Feature extraction from screenshot about the inventory
        @param iteration:
        """

        self.state[inventory_key] = read_inventory(inventory_screenshot, self.icon_classifier_model,
                                                   self.icon_classifier_train_config, self.icon_classifier_int_2_label,
                                                   inventory_open)

    def process_screenshot(self, screenshot: np.ndarray):
        """
        Saving screenshot itself + feature extraction from screenshot e.g. about the inventory
        @param iteration:
        """
        view_bgr: np.ndarray = resize(screenshot, POV_WIDTH, POV_HEIGHT)
        view_rgb = cv2.cvtColor(view_bgr, cv2.COLOR_BGR2RGB)
        self.state[view_key] = view_rgb
        self.state[health_key] = observe_health_bar(screenshot)
        # self.state[hunger_key] = observe_hunger_bar(screenshot) # in peaceful constant
        self.state[experience_key] = observe_exp_bar(screenshot)
        self.state[level_key] = observe_exp_level(screenshot)

