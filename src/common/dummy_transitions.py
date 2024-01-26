import time

from src.common.action_space import get_action_dict_for_action_id, get_random_action
from src.common.observation_keys import view_key
from src.common.observation_space import get_initial_state
from src.common.terminal_state import TerminalState
from src.common.transition import Transition


def get_dummy_transition_seq(x_depth, step_length, screenshot):
    transition_seq = []
    for i in range(x_depth):
        state: dict = get_initial_state()  # includes pov screenshot
        state[view_key] = screenshot

        action_id = get_random_action(step_length)
        action_dict: dict = get_action_dict_for_action_id(action_id)

        transition: Transition = Transition(t=i, state=state, action=action_dict,
                                            reward=0,
                                            terminal_state=str(TerminalState.NONE), timestamp=time.time(),
                                            action_id=action_id)
        transition_seq.append(transition)
    return transition_seq
