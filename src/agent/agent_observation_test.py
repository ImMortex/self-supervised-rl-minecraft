import copy
import logging
import os

import coloredlogs
import cv2
import numpy as np
from PIL import Image
from dotenv import load_dotenv


from config.train_config import get_train_config
from src.agent.agent import McRlAgent
from src.agent.observation.observe_inventory.libs.observe_inventory_slots import get_inventory_slots_test_mapping_image
from src.agent.reward_function import is_task_done
from src.common.observation_keys import inventory_key

coloredlogs.install(level='INFO')
load_dotenv()
PRETRAINED_ICON_CLASSIFIER = os.getenv("PRETRAINED_ICON_CLASSIFIER")
TARGET_SCORE = os.getenv("TARGET_SCORE")
TASK_ITEM_KEY = os.getenv("TASK_ITEM_KEY")

def test_inventory_content(agent, img, expected_count):
    state, terminal_state, dead = agent.observe_env(force_stop=False, screenshot=img)

    done = is_task_done(state=state, task_item_key=TASK_ITEM_KEY, target_amount=expected_count)
    return done, state


def agent_observation_test(show_mapping_image: bool = False) -> bool:
    logging.info("Running observation test.")
    logging.info("Initialize agent in debug mode.")
    path = "agentObservationTestInput/screenshot.png"
    #path2 = "agentObservationTestInput/screenshot2.png"
    screenshot: np.ndarray = None
    screenshot2: np.ndarray = None

    """
    try:
        screenshot = cv2.imread(path)
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)  # because read from img file

    except Exception as e:
        logging.error(
            "agent_observation_test: " + path + " is missing. Please read agentObservationTestInput/README.md")
        return False
    """

    try:
        screenshot2 = cv2.imread(path)
        screenshot2 = cv2.cvtColor(screenshot2, cv2.COLOR_BGR2RGB)  # because read from img file
    except Exception as e:
        logging.error(
            "agent_observation_test: " + path + " is missing. Please read agentObservationTestInput/README.md")
        return False

    train_config: dict = get_train_config()
    dry_run = True
    agent_id = "Test"
    generation_id = "Test"
    session_id = "Test"

    local_filesystem_store_root_dir = "tmp/agentTestOutput"
    mode = "train_a3c"
    agent: McRlAgent = McRlAgent(dry_run=dry_run, agent_id=agent_id, generation_id=generation_id,
                                 session_id=session_id,
                                 local_filesystem_store_root_dir=local_filesystem_store_root_dir,
                                 mode=mode, t_per_second=train_config["t_per_second"],
                                 logging_with_wandb=False)
    agent.initialize_agent_epoch(generation_id=generation_id)
    #done, state = test_inventory_content(agent, screenshot, expected_count=72)
    done2, state2 = test_inventory_content(agent, screenshot2, expected_count=45)

    #print(state)
    #print(state[inventory_key])
    print(state2[inventory_key])

    if show_mapping_image:
        img: Image = Image.open(path)
        img = get_inventory_slots_test_mapping_image(img)
        img.show()

    if done2: #done and done2:
        logging.info("Observation tested successfully")
        return True
    else:
        logging.info("Observation test failed")

        """
        if not done:
            logging.info("Inventory mapping not good enough")
        else:
            logging.info("Inventory mapping OK")
        """

        if not done2:
            logging.info("Toolbar mapping not good enough")
        else:
            logging.info("Toolbar mapping OK")

        return False
