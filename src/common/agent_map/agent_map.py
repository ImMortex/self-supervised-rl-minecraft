import copy
import json
import logging
import math
import os
import time
import traceback
import warnings

import coloredlogs
import cv2
import numpy as np
import urllib3
from dotenv import load_dotenv

from src.common.minio_fncts.minio_api_calls import download_json, download_image
from src.common.minio_fncts.minio_helpers import get_s3_transition_rel_paths_grouped_by_agent, minio_check_bucket, \
    get_minio_client_secure_no_cert
from src.common.persisted_memory import PersistedMemory
from src.common.transition import Transition

warnings.simplefilter(action='ignore', category=FutureWarning)
coloredlogs.install(level='INFO')

load_dotenv()
MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")
AGENT_STEPS_PER_SECOND = os.getenv("AGENT_STEPS_PER_SECOND")
agent_step_length = 1 / float(AGENT_STEPS_PER_SECOND)


def get_transitions_for_agent(agent_paths, minio_client, bucket_name):
    transitions: [] = []
    for transition_path in agent_paths:
        transition_file, view_file = PersistedMemory.get_transition_file_paths(transition_path)
        transition: Transition = None
        needed_retries: int = 0
        # retry if connection to server is lost
        while True:
            error = False
            try:
                start_download_time = time.time()
                transition_dict: dict = json.loads(
                    download_json(minio_client, bucket_name, transition_file))
                img = download_image(minio_client, bucket_name, view_file)
                transition = PersistedMemory.deserialize_transition(transition_dict=transition_dict, image=img)
                logging.info("Transition downloaded from minio in " + str(time.time() - start_download_time))

            except Exception as e:
                logging.error("Transition download from minio failed")
                error = True
                traceback.print_exc()

            if transition is not None and img is not None and not error:
                break  # everything went good
            needed_retries += 1
            time.sleep(5)
            logging.error("Transition download from minio error. Retries: " + str(needed_retries))

        transitions.append(transition)

    return transitions


def draw_map_for_each_agent():
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.warning("Disabled minio warnings")
    minio_client = get_minio_client_secure_no_cert()
    bucket_name = MINIO_BUCKET_NAME
    minio_check_bucket(minio_client, bucket_name)
    s3_transition_rel_paths_grouped_by_agent = get_s3_transition_rel_paths_grouped_by_agent(minio_client,
                                                                                            bucket_name)
    logging.info("transitions from " + str(len(s3_transition_rel_paths_grouped_by_agent)) + " agents")
    id = 0
    for agent_paths in s3_transition_rel_paths_grouped_by_agent:
        transitions: [] = get_transitions_for_agent(agent_paths, minio_client, bucket_name)

        img = draw_map_for_agent(transitions)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite("tmp/map" + str(id) + ".png", img)
        id += 1


def rotate_vector(vector, angle):
    x = vector[0] * math.cos(angle) - vector[1] * math.sin(angle)
    y = vector[0] * math.sin(angle) + vector[1] * math.cos(angle)
    return [x, y]


def draw_map_for_agent(transitions=None, step_length=agent_step_length) -> np.ndarray:
    """
    A map is created based on the saved transitions in order to reconstruct the agent's movements from a bird's
    eye view.

    Legend:
    Action 0 (move forward):            black line
    Action 1 (jump forward):            blue line
    Action 2 (attack):                  red half circle (one circle per repetition)
    Action 3 (turn 10 degrees right):   next line in the appropriate direction
    Action 4 (turn 10 degrees left):    next line in the appropriate direction
    Reward:                             green half circle (one circle per repetition)

    @param transitions: list of transitions
    @param step_length: length of a time step in seconds
    @return: np.ndarray
    """
    if transitions is None:
        transitions = []
    width = 8000
    height = 8000
    scale = 5.0  # n pixels are one block (one block is one meter)
    max_d = int(width / scale)
    walk_speed = 4.317  # (blocks per second) walk speed according to https://minecraft.fandom.com/wiki/Walking
    move_length = walk_speed * scale * step_length
    pos = np.array([round(width / 2), round(height / 2)]).astype('float64')
    start_pos = pos
    direction_vector = np.array([0.0, -1.0]).astype('float64')  # direction to north
    movement_vector = np.array([move_length, move_length]).astype('float64')

    img = np.zeros([height, width, 3], dtype=np.uint8)
    img.fill(255)

    # create raster (one grey dot for each block)
    for x in range(width):
        for y in range(height):
            if x % scale == 0 and y % scale == 0:
                img = cv2.circle(img, (x, y), radius=0, color=(150, 150, 150), thickness=0)

    # circles to show distance from start point
    radius = 100
    radius_px = int(radius * scale)
    num_circles = int(max_d / radius)
    for i in range(num_circles):
        img = cv2.circle(img, (int(start_pos[0]), int(start_pos[1])), radius=radius_px * (i + 1), color=(255, 150, 255),
                         thickness=0)

    img = cv2.circle(img, tuple(pos.astype('int64')), radius=2, color=(255, 0, 255), thickness=1)  # start
    attack_actions_this_position = 0
    try:
        for transition in transitions:

            transition: Transition = transition
            prev_pos = copy.deepcopy(pos)

            if transition.action_id == 0 or transition.action_id == 1:
                pos += direction_vector * movement_vector  # calculate next position
                img = cv2.circle(img, tuple(pos.astype('int64')), radius=1, color=(0, 0, 0), thickness=1)  # footprint
                attack_actions_this_position = 0
                if transition.action_id == 0:
                    img = cv2.line(img, tuple(prev_pos.astype('int64')), tuple(pos.astype('int64')), (0, 0, 0), 1)
                else:
                    img = cv2.line(img, tuple(prev_pos.astype('int64')), tuple(pos.astype('int64')), (0, 0, 255), 1)

            elif transition.action_id == 2:
                attack_actions_this_position += 1
                radius = 1 + 2 * attack_actions_this_position
                red = min(255, 155 + 10 * attack_actions_this_position)
                img = cv2.ellipse(img, tuple(prev_pos.astype('int64')), (radius, radius), 0, 0, 180, color=(red, 0, 0))

            elif transition.action_id == 3:
                direction_vector = rotate_vector(direction_vector, math.radians(10))  # turn right

            elif transition.action_id == 4:
                direction_vector = rotate_vector(direction_vector, math.radians(-10))  # turn left

            if transition.reward > 0:
                radius: int = 1 + 2 * int(transition.reward)
                green = min(255, 155 + 10 * int(transition.reward))
                img = cv2.ellipse(img, tuple(prev_pos.astype('int64')), (radius, radius), 180, 0, 180,
                                  color=(0, green, 0))
    except Exception as e:
        logging.error(e)
        traceback.print_exc()

    return img


def draw():
    draw_map_for_each_agent()
    # draw_map_for_agent()
