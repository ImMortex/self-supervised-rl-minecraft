from copy import deepcopy, copy

import pandas as pd

from src.common.action_space import get_zero_action_state, validate_actions
from src.common.helpers.helpers import save_dict_as_json


def is_valid(action_dict: dict):
    # all not possible actions
    if action_dict["back"] and action_dict["forward"]:
        return False
    if action_dict["right"] and action_dict["left"]:
        return False
    if action_dict["sprint"] and not action_dict["forward"]:
        return False
    if action_dict["sneak"] and (action_dict["sprint"] or action_dict["jump"]):
        return False
    if action_dict["use"] and action_dict["attack"]:
        return False

    # reduced actions for better performance, the following combinations are forbidden:
    if action_dict["forward"] and action_dict["right"]:
        return False
    if action_dict["forward"] and action_dict["left"]:
        return False
    if action_dict["back"] and action_dict["right"]:
        return False
    if action_dict["back"] and action_dict["left"]:
        return False

    if action_dict["use"] and action_dict["right"]:
        return False
    if action_dict["use"] and action_dict["left"]:
        return False
    if action_dict["use"] and action_dict["forward"]:
        return False
    if action_dict["use"] and action_dict["back"]:
        return False

    if action_dict["attack"] and action_dict["right"]:
        return False
    if action_dict["attack"] and action_dict["left"]:
        return False
    if action_dict["attack"] and action_dict["forward"]:
        return False
    if action_dict["attack"] and action_dict["back"]:
        return False

    if action_dict["jump"] and action_dict["attack"]:
        return False
    if action_dict["jump"] and action_dict["use"]:
        return False

    if action_dict["jump"] and action_dict["left"]:
        return False
    if action_dict["jump"] and action_dict["right"]:
        return False
    if action_dict["jump"] and action_dict["back"]:
        return False

    left_hand_used = 0
    right_hand_used = 0
    for key in action_dict:
        if is_no_default_value(action_dict, key):
            if (key == "camera" or key == "attack" or "key" == "use" or "key" == "toolbar"):
                right_hand_used += 1
            else:
                if action_dict[key] != False or action_dict[key] != 0:
                    left_hand_used += 1

    if left_hand_used > 5:
        return False

    # "inventory" and "do nothing" are added separately
    if action_dict["inventory"]:
        return False

    # do nothing
    if left_hand_used == 0 and right_hand_used == 0:
        return False

    return True


def is_no_default_value(action_dict, act_key):
    return act_key != "_id" and action_dict[act_key] != 0 and action_dict[act_key] != False and action_dict[
        act_key] != (0, 0)


def loop_step(loop_dict, bool_keys):
    global combinations
    global combination_count
    if len(bool_keys) == 0:
        loop_dict["camera"] = (0, 0)
        if not is_valid(loop_dict):
            return
        combinations[combination_count] = deepcopy(loop_dict)
        combination_count += 1
        return
    loop_name = bool_keys.pop(0)

    for state in [False, True]:
        loop_dict["toolbar"] = 0
        loop_dict[loop_name] = state
        loop_step(loop_dict, copy(bool_keys))


def add_camera_action(combinations, combination_count, vertical_degree, horizontal_degree):
    new_dict = get_zero_action_state()
    new_dict["camera"] = (vertical_degree, horizontal_degree)
    combinations[combination_count] = new_dict
    combination_count += 1
    return combinations, combination_count


combination_count = 0
combinations: dict = {}

# add default "do nothing" action
combinations[combination_count] = get_zero_action_state()
combination_count += 1

# add inventory action
new_dict = get_zero_action_state()
new_dict["inventory"] = True
combinations[combination_count] = new_dict
combination_count += 1

bool_state_names = ["attack", "use", "sneak", "sprint", "forward", "jump", "left", "right", "back", "inventory"]
forward_dict = {}
loop_step(forward_dict, bool_state_names)

# rotation without movement to reduce action dim for better performance
vertical_degree = 0
horizontal_degree = 0
degrees = [0, 1, 5, 25, -1, -5, -25]

enable_vertical_and_horizontal_simultaneously: bool = False

if enable_vertical_and_horizontal_simultaneously:
    for vertical_degree in degrees:
        for horizontal_degree in degrees:
            combinations, combination_count = add_camera_action(combinations, combination_count, vertical_degree,
                                                                horizontal_degree)
else:
    for vertical_degree in degrees:
        combinations, combination_count = add_camera_action(combinations, combination_count, vertical_degree,
                                                            horizontal_degree)

    for horizontal_degree in degrees:
        combinations, combination_count = add_camera_action(combinations, combination_count, vertical_degree,
                                                            horizontal_degree)

# toolbar use without movement to reduce action dim for better performance
for toolbar_value in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    new_dict = get_zero_action_state()
    new_dict["toolbar"] = toolbar_value
    combinations[combination_count] = new_dict
    combination_count += 1

combinations_no_duplicates = []

for i, c in combinations.items():
    is_duplicate = False
    for n in combinations_no_duplicates:
        if c == n:
            is_duplicate = True
            break
    if not is_duplicate:
        combinations_no_duplicates.append(c)

combinations: dict = {}
for i, n in enumerate(combinations_no_duplicates):
    combinations[str(i)] = n

for key in combinations:
    combinations[key]["_id"] = int(key)

# save action as sparse dict (delete default values)
for key in combinations:
    action_dict = deepcopy(combinations[key])
    for act_key in action_dict:
        if not is_no_default_value(action_dict, act_key) and act_key != "_id":
            del combinations[key][act_key]

validate_actions(combinations)

print("new action dim: " + str(len(combinations)))
print("finished")

save_dict_as_json(combinations, "./config", "actions.json", sort_keys=False)

# save actions as array human readable

action_string_array: [] = []
for key in combinations:
    string_value = ""
    action_dict = deepcopy(combinations[key])
    del action_dict["_id"]

    string_value += str(action_dict)

    action_string_array.append(string_value)

save_dict_as_json({"actions": action_string_array}, "./tmp", "actions_string.json", sort_keys=False)

df = pd.DataFrame(action_string_array, columns=['subactions'])
df.to_csv("./tmp/actions.csv", index_label="id", sep=",")
