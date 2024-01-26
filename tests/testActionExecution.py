import unittest

from src.agent.actionExecution import ActionExecution


class TestActionExecution(unittest.TestCase):

    def test_mouse_move_9_steps(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        timestep = 0.2
        needed_time, rest_x, rest_y = action_execution.move_mouse(10, 10, timestep, 9)
        self.assertEqual(0, rest_x, "rest_x")
        self.assertEqual(0, rest_y, "rest_y")
        self.assertLess(needed_time, timestep)

    def test_mouse_move_10_steps(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        timestep = 0.2
        needed_time, rest_x, rest_y = action_execution.move_mouse(10, 10, timestep, 10)
        self.assertEqual(0, rest_x, "rest_x")
        self.assertEqual(0, rest_y, "rest_y")
        self.assertLess(needed_time, timestep)

    def test_mouse_move_1_step(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        timestep = 0.2
        needed_time, rest_x, rest_y = action_execution.move_mouse(10, 10, timestep, 10)
        self.assertEqual(0, rest_x, "rest_x")
        self.assertEqual(0, rest_y, "rest_y")
        self.assertLess(needed_time, timestep)

    def test_get_mouse_down_px_max(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        px = action_execution.camera_m_degree_in_mouse_m_px(180)
        margin = 5
        result = action_execution.get_mouse_down_px(px=px, window_height=1080, mouse_pos=(0, 1080 - margin),
                                                    window_margin=margin)
        self.assertEqual(0, result)

    def test_get_mouse_down_px_max_2(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        px = action_execution.camera_m_degree_in_mouse_m_px(180)
        margin = 5
        result = action_execution.get_mouse_down_px(px=px, window_height=1080, mouse_pos=(0, 1000),
                                                    window_margin=margin)
        self.assertEqual(80 - margin, result)

    def test_get_mouse_up_px_max(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        px = action_execution.camera_m_degree_in_mouse_m_px(180)
        margin = 5
        result = action_execution.get_mouse_up_px(px=px, window_origin_y=0, mouse_pos=(0, 0), window_margin=margin)
        self.assertEqual(0, result)

    def test_get_mouse_up_px_max_2(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        px = action_execution.camera_m_degree_in_mouse_m_px(180)
        margin = 5

        result = action_execution.get_mouse_up_px(px=px, window_origin_y=0, mouse_pos=(0, 30), window_margin=margin)
        self.assertEqual(30 - margin, result)

    def test_get_mouse_right_px_max(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        px = action_execution.camera_m_degree_in_mouse_m_px(180)
        margin = 5

        result = action_execution.get_mouse_right_px(px=px, window_width=1920, mouse_pos=(1920 - margin, 0),
                                                     window_margin=margin)
        self.assertEqual(0, result)

    def test_get_mouse_right_px_max_2(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        px = action_execution.camera_m_degree_in_mouse_m_px(180)
        margin = 5

        result = action_execution.get_mouse_right_px(px=px, window_width=1920, mouse_pos=(1800, 0),
                                                     window_margin=margin)
        self.assertEqual(120 - margin, result)

    def test_get_mouse_left_px_max(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        px = action_execution.camera_m_degree_in_mouse_m_px(180)
        margin = 5

        result = action_execution.get_mouse_left_px(px=px, window_origin_x=0, mouse_pos=(margin, 0),
                                                    window_margin=margin)
        self.assertEqual(0, result)

    def test_get_mouse_left_px_max_2(self):
        action_execution: ActionExecution = ActionExecution()
        action_execution.stay_on_main_screen = False
        px = action_execution.camera_m_degree_in_mouse_m_px(180)
        margin = 5

        result = action_execution.get_mouse_left_px(px=px, window_origin_x=0, mouse_pos=(133, 0), window_margin=margin)
        self.assertEqual(133 - margin, result)
