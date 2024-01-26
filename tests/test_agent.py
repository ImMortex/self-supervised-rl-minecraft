import os
import unittest

from dotenv import load_dotenv

from src.agent.observation.observation import Observation
from src.agent.observation.observe_inventory.libs.observe_inventory_classify import load_image_classifier_model
from src.agent.reward_function import get_inventory_reward, is_task_done
from src.common.observation_keys import inventory_key

death_negative_reward = 64


load_dotenv()
PRETRAINED_ICON_CLASSIFIER = os.getenv("PRETRAINED_ICON_CLASSIFIER")
icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label = \
    load_image_classifier_model(
        model_dir=PRETRAINED_ICON_CLASSIFIER)  # makes one prediction to avoid lag on first use with GPU

class TestAgent(unittest.TestCase):

    def test_get_inventory_reward__equal(self):
        observation: Observation = Observation(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label)
        state: dict = observation.state

        state[inventory_key] = {
            "oak_log": {
                'amount': 64,
                'damage': -1,
                'max_damage': -1,
                'reward_value': 1
            }
        }

        reward_value = get_inventory_reward(state=state, task_item_key="oak_log")

        self.assertEqual(64, reward_value)

    def test_get_inventory_reward__too_much(self):
        observation: Observation = Observation(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label)
        state: dict = observation.state

        state[inventory_key] = {
            "oak_log": {
                'amount': 65,
                'damage': -1,
                'max_damage': -1,
                'reward_value': 1
            },
            "birch_log": {
                'amount': 65,
                'damage': -1,
                'max_damage': -1,
                'reward_value': 1
            }
        }

        reward_value = get_inventory_reward(state=state, task_item_key="oak_log")

        self.assertEqual(65, reward_value)

    def test_get_inventory_reward__not_enough(self):
        observation: Observation = Observation(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label)
        state: dict = observation.state

        state[inventory_key] = {
            "oak_log": {
                'amount': 32,
                'damage': -1,
                'max_damage': -1,
                'reward_value': 1
            }
        }

        reward_value = get_inventory_reward(state=state, task_item_key="oak_log")

        self.assertEqual(32, reward_value)

    def test_is_task_done__equal(self):
        observation: Observation = Observation(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label)
        state: dict = observation.state

        state[inventory_key] = {
            "oak_log": {
                'amount': 64,
                'damage': -1,
                'max_damage': -1,
                'reward_value': 1
            }
        }

        done = is_task_done(state=state, task_item_key="oak_log", target_amount=64)

        self.assertTrue(done)

    def test_is_task_done__too_much(self):
        observation: Observation = Observation(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label)
        state: dict = observation.state

        state[inventory_key] = {
            "oak_log": {
                'amount': 65,
                'damage': -1,
                'max_damage': -1,
                'reward_value': 1
            },
            "birch_log": {
                'amount': 65,
                'damage': -1,
                'max_damage': -1,
                'reward_value': 1
            }
        }

        done = is_task_done(state=state, task_item_key="oak_log", target_amount=64)

        self.assertTrue(done)

    def test_is_task_done__not_enough(self):
        observation: Observation = Observation(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label)
        state: dict = observation.state

        state[inventory_key] = {
            "oak_log": {
                'amount': 32,
                'damage': -1,
                'max_damage': -1,
                'reward_value': 1
            }
        }

        done = is_task_done(state=state, task_item_key="oak_log", target_amount=64)

        self.assertFalse(done)
