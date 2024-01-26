import os
import time
import unittest

import cv2
from dotenv import load_dotenv

from src.agent.observation.observation import Observation
from src.agent.observation.observe_inventory.libs.observe_inventory_classify import load_image_classifier_model
from src.common.action_space import get_action_dict_for_action_id
from src.common.observation_keys import view_key, POV_WIDTH, POV_HEIGHT
from src.common.persisted_memory import PersistedMemory
from src.common.screen.screenshotUtils import get_screenshot
from src.common.terminal_state import TerminalState
from src.common.transition import Transition

load_dotenv()
PRETRAINED_ICON_CLASSIFIER = os.getenv("PRETRAINED_ICON_CLASSIFIER")
icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label = \
    load_image_classifier_model(
        model_dir=PRETRAINED_ICON_CLASSIFIER)  # makes one prediction to avoid lag on first use with GPU

class TestPersistedMemory(unittest.TestCase):

    def test_save_transition(self):
        session_id: str = PersistedMemory.get_session_id_today()
        agent_id: str = "1"
        generation_id: str = "1"
        local_filesystem_store_root_dir: str = "tmp/testStorage"
        local_filesystem: bool = True
        persisted_memory: PersistedMemory = PersistedMemory(img_shape=(POV_HEIGHT, POV_WIDTH, 3),
                                                            session_id=session_id,
                                                            agent_id=agent_id,
                                                            generation_id=generation_id,
                                                            local_filesystem_store_root_dir=local_filesystem_store_root_dir,
                                                            persist_to_local_filesystem=local_filesystem)
        self.assertEqual(session_id, persisted_memory.session_id)
        self.assertEqual(agent_id, persisted_memory.agent_id)
        self.assertEqual(generation_id, persisted_memory.generation_id)
        self.assertEqual(
            PersistedMemory.get_agent_rel_path(session_id=session_id, generation_id=generation_id, agent_id=agent_id),
            persisted_memory.get_rel_agent_path())
        observation0: Observation = Observation(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label)
        state0: dict = observation0.get_actual_state()  # includes pov screenshot
        timestep0 = 0
        action_id0 = 0
        action_dict0: dict = get_action_dict_for_action_id(action_id0)
        timestamp0 = time.time()
        transition0: Transition = Transition(t=timestep0, state=state0, action=action_dict0,
                                             reward=0,
                                             terminal_state=str(TerminalState.NONE), timestamp=timestamp0,
                                             action_id=action_id0)

        persisted_memory.save_timestep_in_ram(transition0)

        observation1: Observation = Observation(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label)
        state1: dict = observation1.get_actual_state()  # includes pov screenshot
        screenshot = get_screenshot()
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)  # not RGB
        observation1.process_screenshot(screenshot)
        timestep1 = 1
        action_id1 = 200
        action_dict1: dict = get_action_dict_for_action_id(action_id1)
        timestamp1 = time.time()
        transition1: Transition = Transition(t=timestep1, state=state1, action=action_dict1,
                                             reward=0,
                                             terminal_state=str(TerminalState.NONE), timestamp=timestamp1,
                                             action_id=action_id1)
        persisted_memory.save_timestep_in_ram(transition1)
        persisted_memory.save_from_ram_to_persisted()

        loaded_transition0: Transition = persisted_memory.load_timestep(session_id=session_id,
                                                                        generation_id=generation_id, agent_id=agent_id,
                                                                        timestep_id=0)
        loaded_transition1: Transition = persisted_memory.load_timestep(session_id=session_id,
                                                                        generation_id=generation_id, agent_id=agent_id,
                                                                        timestep_id=1)

        self.assertEqual(transition0.t, loaded_transition0.t, "0 timestep")
        self.assertEqual(transition0.action_id, loaded_transition0.action_id, "0 action_id")
        self.assertEqual(transition0.state[view_key].tolist(), loaded_transition0.state[view_key].tolist(),
                         "0 state.view")
        self.assertEqual(transition0.terminal_state, loaded_transition0.terminal_state, "0 terminal_state")

        self.assertNotEqual(loaded_transition0.state[view_key].tolist(), loaded_transition1.state[view_key].tolist(),
                            "comparing state.view of both transitions. They should not be equal")

        transition0.state[view_key] = None
        loaded_transition0.state[view_key] = None
        self.assertDictEqual(transition0.state, loaded_transition0.state, "0 state")

        self.assertEqual(transition1.t, loaded_transition1.t, "1 timestep")
        self.assertEqual(transition1.action_id, loaded_transition1.action_id, "1 action_id")
        self.assertEqual(transition1.state[view_key].tolist(), loaded_transition1.state[view_key].tolist(),
                         "1 state.view")
        self.assertEqual(transition1.terminal_state, loaded_transition1.terminal_state, " 1terminal_state")

        transition1.state[view_key] = None
        loaded_transition1.state[view_key] = None
        self.assertDictEqual(transition1.state, loaded_transition1.state, "1 state")

    def test_get_transition_id_from_path(self):
        path = "D:/McRlAgentTestSmall/sessions/session_2023.07.24.00.00.00/generation_2023_07_24__15_20/agent_INS-202-PC12/timestep_280"

        t_id: int = PersistedMemory.get_transition_id_from_path(path)
        self.assertEqual(280, t_id)

    def test_get_transition_id_from_path(self):
        path = "D:/McRlAgentTestSmall/sessions/session_2023.07.24.00.00.00/generation_2023_07_24__15_20/agent_INS-202-PC12/timestep_280"
        n_path = "D:/McRlAgentTestSmall/sessions/session_2023.07.24.00.00.00/generation_2023_07_24__15_20/agent_INS-202-PC12/timestep_281"

        result = PersistedMemory.get_transition_neighbour_path(path, 281)
        self.assertEqual(n_path, result)
