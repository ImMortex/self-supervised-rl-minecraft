import numpy as np

from src.common.observation_keys import health_key, hunger_key, experience_key, level_key, \
    view_key, POV_WIDTH, POV_HEIGHT, inventory_key, inventory_open_key


def get_blank_view(width: int = POV_WIDTH, height: int = POV_HEIGHT) -> np.ndarray:
    return np.zeros((height, width, 3), np.uint8)


def get_initial_state() -> dict:
    state: dict = {}
    """
    state[armor_key] = {
        'head': {
            'amount': 0,
            'type': 'air',
            'damage': -1,
            'max_damage': -1
        },
        'torso': {
            'amount': 0,
            'type': 'air',
            'damage': -1,
            'max_damage': -1
        },
        'legs': {
            'amount': 0,
            'type': 'air',
            'damage': -1,
            'max_damage': -1
        },
        'feets': {
            'amount': 0,
            'type': 'air',
            'damage': -1,
            'max_damage': -1
        }
    }
    state[main_hand_key] = {
        'amount': 0,
        'type': 'air',
        'damage': -1,
        'max_damage': -1
    }
    state[second_hand_key] = {
        'amount': 0,
        'type': 'air',
        'damage': -1,
        'max_damage': -1
    }
    """
    state[inventory_key]: dict = {}  # sparse dictionary includes only keys of existing items
    state[health_key]: int = 20
    state[hunger_key]: int = 20
    state[experience_key] = 0
    state[level_key] = 0
    state[view_key]: np.ndarray = get_blank_view(POV_WIDTH, POV_HEIGHT)
    state[inventory_open_key]: bool = False
    # state[mining_progress_key] = 0
    # state[window_key] = []  # additional inventory of the opened window (e.g. chest, villager, etc.)

    # add_placeholders_to_inventory(state[inventory_key])

    return state


