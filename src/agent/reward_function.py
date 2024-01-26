from src.common.observation_keys import inventory_key

def get_inventory_reward(state: dict, task_item_key) -> float:
    reward = 0
    inventory = state[inventory_key]

    if task_item_key in inventory:
        amount = inventory[task_item_key]["amount"]
        reward += amount

    return reward


def is_task_done(state: dict, task_item_key, target_amount) -> bool:
    """
    Returns True if the full task was completed
    """

    if target_amount < 0:
        return False

    done = False
    inventory = state[inventory_key]

    if task_item_key in inventory:
        amount = inventory[task_item_key]["amount"]
        if amount >= target_amount:
            done = True
        else:
            done = False

    return done
