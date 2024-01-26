import logging

import numpy as np

from src.common.helpers.helpers import load_from_json_file


def get_zero_action_state():
    return {
        # left hand/ keyboard actions #
        "forward": False,  # w release/press
        "back": False,  # s release/press
        "left": False,  # a release/press
        "right": False,  # d release/press
        "sprint": False,  # ctr_l + w release/press (1 sets "sneak" and "back" to 0)
        "sneak": False,  # l_shift release/press
        "jump": False,  # space bar release/press
        "inventory": False,  # do nothing or tap e one time (1 sets movement actions to 0)

        # right hand/ mouse actions #
        "toolbar": 0,  # do nothing or press 1-9
        "attack": False,  # left mouse button release/press
        "use": False,  # right mouse button release/press
        "camera": (0, 0),
        # move mouse over x-axis and y-axis in degrees between -180 and 180
    }


def validate_actions(sparse_action_dicts: dict):
    found_keys: dict = {}
    for key in sparse_action_dicts:
        action_dict = sparse_action_dicts[key]
        for act_key in action_dict:
            found_keys[act_key] = act_key
    missing_keys = []
    """
    for key in get_zero_action_state():
        if key not in found_keys:
            missing_keys.append(key)
    if len(missing_keys) > 0:
        raise Exception("Actions missing subactions: " + str(missing_keys))
    """


all_actions: dict = load_from_json_file("./config/actions.json")
validate_actions(all_actions)
logging.info("Action space " + str(len(all_actions)) + " Minecraft actions loaded from config")
all_action_ids = []
attack_action_ids = []
no_attack_action_ids = []
for a in all_actions:
    a_dict: dict = all_actions[a]
    if "attack" in a_dict and a_dict["attack"]:
        attack_action_ids.append(int(a_dict["_id"]))
    else:
        no_attack_action_ids.append(int(a_dict["_id"]))
    all_action_ids.append(int(a_dict["_id"]))

BREAK_TIME_WOOD_BLOCK = 3  # https://minecraft.fandom.com/wiki/Wood (18.09.2023)
attack_action_sequence_counter = 0


def get_all_action_ids() -> []:
    return all_action_ids


def get_all_attack_action_ids() -> []:
    return attack_action_ids


def get_action_dim() -> int:
    """
    @return: total count of existing actions
    """
    return len(all_actions)


def get_random_action(step_length):
    # counter enforces attack action sequences
    if step_length >= BREAK_TIME_WOOD_BLOCK:
        attack_action_sequence_len = 0  # enforce attack action sequences deactivated
    else:
        attack_action_sequence_len = round(BREAK_TIME_WOOD_BLOCK / step_length) + 1
    global attack_action_sequence_counter
    if attack_action_sequence_counter >= attack_action_sequence_len:
        attack_action_sequence_counter = 0

    # equal distribution to dice an random action
    action = int(np.random.choice(a=all_action_ids))
    # decision tree
    if action in attack_action_ids or attack_action_sequence_counter > 0:
        attack_action_sequence_counter += 1
        # distribution only including attack actions
        return int(np.random.choice(a=attack_action_ids))
    else:
        attack_action_sequence_counter = 0
        # distribution only including none attack actions
        return int(np.random.choice(a=no_attack_action_ids))

def get_random_action_equal_distributed(action_dim):
    return int(np.random.randint(low=0, high=action_dim))


def get_last_action_id() -> int:
    return len(all_actions) - 1


def get_action_dict_for_action_id(action_id: int) -> dict:
    action: dict = get_default_action()
    try:

        if action_id >= 0 and action_id <= get_last_action_id():
            action.update(all_actions[str(action_id)])
            return action
    except Exception as e:
        logging.error(e)

    # all other cases:
    return action  # default action


def get_default_action():
    default_action: dict = get_zero_action_state()
    return default_action

def get_random_action_dist(all_action_ids, n):
    abs_frequency: dict = {}
    rel_frequency: dict = {}
    for i in all_action_ids:
        abs_frequency[str(i)] = 0
    actions = []
    for i in range(n):
        action = get_random_action(2)
        actions.append(action)
        abs_frequency[str(action)] += 1
    for k in abs_frequency:
        rel_frequency[k] = abs_frequency[k] / n
    return rel_frequency, abs_frequency, actions
