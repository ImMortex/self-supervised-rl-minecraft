import logging
import math
import time
from threading import Thread

import mouse
import pynput
from pynput.keyboard import Key

from src.agent.observation.agent_make_screenshot import agent_make_screenshot
from src.agent.observation.observe_inventory.libs.observe_inventory_recipe_book import is_inventory_open
from src.common.action_space import get_zero_action_state, get_default_action, get_action_dict_for_action_id
from src.common.env_utils.environment_info import get_application_name
from src.common.force_stop_if_app_not_found import force_stop_if_app_not_found
from src.common.helpers.helpers import load_from_json_file
from src.common.screen.screenUtils import get_main_screen_height, get_main_screen_width, get_main_screen_origin_x, \
    get_main_screen_origin_y

respawn_button: dict = load_from_json_file("./config/death_screen_respawn_button_no_fullscreen_conf.json")
respawn_top_left = respawn_button["top_left"]
respawn_bottom_right = respawn_button["bottom_right"]


class ActionExecution:

    def __init__(self):
        self.stay_on_main_screen = True
        """
        If True executes virtual keyboard and mouse interactions only when cursor is on main screen 
        to avoid unexpected behavior.
        """

        self.pynput_keyboard = pynput.keyboard.Controller()
        self.pynput_mouse_controller = pynput.mouse.Controller()
        self.action_state: dict = get_zero_action_state()
        """
        Encoding of an action. This dict contains all subaction keys to represent the current execution state.
        """

        self.current_action: dict = {}
        """
        Encoding of an action. Contains only the subaction keys whose values are to change.
        """

        self.used_keys: set = set()
        self.sprint_is_on: bool = False  # if the agent sprints instead of walking
        self.inventory_is_open_by_agent: bool = False  # if the agent opened the inventory by action (not forced)

        # Minecraft Mouse settings
        self.mc_mouse_sensitivity = 0.5  # mouse sensitivity default: 0.5, min: 0.0, max: 1.0
        self.mc_invert_mouse: bool = False  # inverted mouse default: False

        # action execution only allowed on main screen
        self.window_origin_x = get_main_screen_origin_x()
        self.window_origin_y = get_main_screen_origin_y()
        self.window_height = get_main_screen_height()
        self.window_width = get_main_screen_width()
        self.window_margin = 45  # virtual mouse cursor should not enter this margin
        self.force_stop = False  # stops
        self.execution_disabled = False  # deactivates execution of actions

    def open_inventory_for_screenshot(self):
        if self.inventory_is_open_by_agent:
            return
        else:
            self.press_inventory_key()

    def close_inventory_for_screenshot(self):
        if self.inventory_is_open_by_agent:
            return
        else:
            self.press_inventory_key()

    def move_mouse(self, xm, ym, timestep, steps=10):
        """
        Funktion divides mouse move in equal long parts (steps)
        param: xm: mouse move x-axis in px
        param: ym: mouse move y-axis in px
        param: timestep: max available time in s
        param: movement steps with breaks
        """
        duration = timestep * 0.5
        start = time.time()
        rest_x = xm
        rest_y = ym
        x = xm * (1 / steps)
        y = ym * (1 / steps)

        for i in range(steps):
            if i == steps - 1:
                mouse.move(rest_x, rest_y, absolute=False)
                rest_x = 0
                rest_y = 0
            else:
                mouse.move(x, y, absolute=False)
                rest_x -= x
                rest_y -= y

            time.sleep(duration / steps)
            if time.time() >= start + timestep:  # Avoidance of not keeping the timestep
                logging.warning("move_mouse takes longer than the given timestep!")
                break

        needed_time = time.time() - start
        logging.debug("move time " + str(needed_time) + " steps " + str(steps) + " rest " + str(rest_x))
        return needed_time, rest_x, rest_y

    def execute_timestep(self, action_id: int = None, action_dict=None, timestep_length_sec: float = 0.2):
        """
        Executes actions using virtual keyboard and mouse inputs only when cursor is on main screen
        """
        try:
            if action_dict is None:
                action_dict = get_default_action()

            if action_id is None:
                self.current_action = action_dict
            else:
                self.current_action = get_action_dict_for_action_id(action_id)

            if (self.stay_on_main_screen and not self.is_cursor_on_main_monitor() and
                not self.execution_disabled) or self.force_stop:
                self.release_all_keys()  # Safety measure
                self.force_stop = True
                return

            if self.execution_disabled:
                return

            self.action_state.update(self.current_action)

            # left hand/ keyboard actions #
            self.handle_inventory_action("inventory")
            self.handle_keyboard_action("sneak", pynput.keyboard.Key.shift_l)
            self.handle_toolbar_action("toolbar")

            self.handle_sprint_action("sprint")
            self.handle_keyboard_action("forward", "w")
            self.handle_keyboard_action("back", "s")
            self.handle_keyboard_action("left", "a")
            self.handle_keyboard_action("right", "d")
            self.handle_keyboard_action("jump", pynput.keyboard.Key.space)

            # right hand/ mouse actions #
            self.handle_mouse_button_action("attack", pynput.mouse.Button.left)
            self.handle_mouse_button_action("use", pynput.mouse.Button.right)

            self.handle_mouse_move_action(timestep_length_sec)

            logging.debug(self.current_action)
        except KeyboardInterrupt:
            self.force_stop = True
            logging.warning("ActionExecution: KeyboardInterrupt. Kindly stopping agent...")


    def handle_mouse_move_action(self, timestep_length_sec):
        """
        Minecraft Mouse movement
        """
        rot_yaw_degree: float = self.action_state["camera"][0]
        rot_pitch_degree: float = self.action_state["camera"][1]
        d_x_px: int = 0  # mouse move x-axis
        d_y_px: int = 0  # mouse move y-axis

        if rot_yaw_degree != 0:
            d_x_px = self.camera_m_degree_in_mouse_m_px(rot_yaw_degree)
        if rot_pitch_degree != 0:
            d_y_px = self.camera_m_degree_in_mouse_m_px(rot_pitch_degree)
        if self.mc_invert_mouse:
            d_y_px = -d_y_px
        logging.debug({"d_x_px": d_x_px, "d_y_px": d_y_px})
        steps = 1  # mouse move is divided in steps
        threads = []  # Threads for parallel execution are closed after the end of the timestep
        if d_x_px != 0:
            if d_x_px < 0:
                tx = Thread(target=self.try_mouse_move_left, args=(abs(d_x_px), timestep_length_sec, steps,))
            else:
                tx = Thread(target=self.try_mouse_move_right, args=(abs(d_x_px), timestep_length_sec, steps,))
            tx.start()
            threads.append(tx)
        if d_y_px != 0:
            if d_y_px < 0:
                ty = Thread(target=self.try_mouse_move_up, args=(abs(d_y_px), timestep_length_sec, steps,))
            else:
                ty = Thread(target=self.try_mouse_move_down, args=(abs(d_y_px), timestep_length_sec, steps,))
            ty.start()
            threads.append(ty)
        for thread in threads:
            # if thread.is_alive():
            thread.join()

    def camera_m_degree_in_mouse_m_px(self, degree):
        """
        Given is the angle by which the camera should rotate.
        This angle is converted to px by which the computer mouse must be moved to move the camera
        Equation used from https://www.mcpk.wiki/wiki/Mouse_Movement (08.06.2023)
        """
        if degree == 0:
            return 0
        f = math.pow((0.6 * self.mc_mouse_sensitivity + 0.2), 3) * 1.2
        return int(degree / f)

    def handle_keyboard_action(self, action_key, key):
        if action_key in self.action_state:
            if self.action_state[action_key] == 0:
                self.handle_release_key(key)
            elif self.action_state[action_key] == 1:
                self.handle_press_key(key)

    def handle_sprint_action(self, action_key):
        if self.action_state[action_key] == 0:
            self.handle_end_sprint()
        elif self.action_state[action_key] == 1:
            self.handle_begin_sprint()

    def handle_inventory_action(self, action_key):
        if self.action_state[action_key] == 0:
            return
        elif self.action_state[action_key] == 1:
            if self.inventory_is_open_by_agent:
                self.handle_open_inventory()
            else:
                self.handle_close_inventory()

    def handle_toolbar_action(self, action_key):
        if self.action_state[action_key] == 0:
            return
        elif 1 <= self.action_state[action_key] <= 9:
            self.pynput_keyboard.tap(str(self.action_state[action_key]))

    def handle_mouse_button_action(self, action_key, button):
        if action_key in self.action_state:

            if self.action_state[action_key] == 1:
                self.pynput_mouse_controller.press(button)
                return

        # all other cases
        try:
            self.pynput_mouse_controller.release(button)
        except Exception as e:
            logging.error(e)

    def handle_press_key(self, key: str):
        self.pynput_keyboard.press(key)
        self.used_keys.add(key)

    def handle_release_key(self, key: str):
        if key in self.used_keys:
            self.pynput_keyboard.release(key)
            self.used_keys.remove(key)

    def handle_begin_sprint(self):
        self.action_state["sneak"] = 0
        self.action_state["back"] = 0
        self.action_state["forward"] = 1

        self.pynput_keyboard.press("w")
        self.pynput_keyboard.press(pynput.keyboard.Key.ctrl)
        self.used_keys.add("w")
        self.sprint_is_on = True

    def handle_end_sprint(self):
        if self.sprint_is_on:
            self.pynput_keyboard.release("w")
            self.used_keys.remove("w")
        self.sprint_is_on = False

    def handle_open_inventory(self):
        self.action_state["left"] = 0
        self.action_state["right"] = 0
        self.action_state["sneak"] = 0
        self.action_state["back"] = 0
        self.action_state["forward"] = 0
        self.action_state["jump"] = 0
        self.action_state["sprint"] = 0
        self.press_inventory_key()

    def press_inventory_key(self):
        self.pynput_keyboard.tap("e")
        self.used_keys.add("e")

    def handle_close_inventory(self):
        self.press_inventory_key()

    def try_mouse_move_down(self, px, timestep, steps: int):
        px = self.get_mouse_down_px(px, self.window_height, self.pynput_mouse_controller.position, self.window_margin)
        if px <= 0:
            return
        self.move_mouse(0, px, timestep, min(steps, px))
        return 0

    def try_mouse_move_up(self, px, timestep, steps: int):
        px = self.get_mouse_up_px(px, self.window_origin_y, self.pynput_mouse_controller.position, self.window_margin)
        if px <= 0:
            return
        self.move_mouse(0, -px, timestep, min(steps, px))

    def try_mouse_move_right(self, px, timestep, steps: int):
        px = self.get_mouse_right_px(px, self.window_width, self.pynput_mouse_controller.position, self.window_margin)
        if px <= 0:
            return

        self.move_mouse(px, 0, timestep, min(steps, px))

    def try_mouse_move_left(self, px, timestep, steps: int):
        px = self.get_mouse_left_px(px, self.window_origin_x, self.pynput_mouse_controller.position, self.window_margin)
        if px <= 0:
            return
        self.move_mouse(-px, 0, timestep, min(steps, px))

    @staticmethod
    def get_mouse_down_px(px, window_height, mouse_pos, window_margin):
        diff = (window_height - window_margin) - mouse_pos[1]
        if diff <= 0:
            px = 0

        elif diff < px:
            px = diff
        return px

    @staticmethod
    def get_mouse_up_px(px, window_origin_y, mouse_pos, window_margin):
        diff = mouse_pos[1] - (window_origin_y + window_margin)
        if diff <= 0:
            px = 0

        elif diff < px:
            px = diff
        return px

    @staticmethod
    def get_mouse_right_px(px, window_width, mouse_pos, window_margin):
        diff = (window_width - window_margin) - mouse_pos[0]
        if diff <= 0:
            px = 0

        elif diff < px:
            px = diff
        return px

    @staticmethod
    def get_mouse_left_px(px, window_origin_x, mouse_pos, window_margin):
        diff = mouse_pos[0] - (window_origin_x + window_margin)
        if diff <= 0:
            px = 0

        elif diff < px:
            px = diff
        return px

    def is_cursor_on_main_monitor(self):
        return self.is_cursor_in_area(self.pynput_mouse_controller.position, self.window_origin_x,
                                      self.window_origin_y, self.window_width, self.window_height)

    @staticmethod
    def is_cursor_in_area(position, window_origin_x, window_origin_y, window_width, window_height):
        return (window_origin_x <= position[0] <= window_width) and (
                window_origin_y <= position[1] <= window_height)

    def click_on_respawn_button(self):
        button_mid = (
            (respawn_top_left[0] + respawn_bottom_right[0]) / 2, (respawn_top_left[1] + respawn_bottom_right[1]) / 2)
        if self.is_cursor_in_area(button_mid, self.window_origin_x,
                                  self.window_origin_y, self.window_width, self.window_height):
            time.sleep(1)
            mouse.move(button_mid[0], button_mid[1], absolute=True, duration=0)
            time.sleep(3)
            mouse.click()
            time.sleep(1)

    def release_all_keys(self):
        """
        This function must be called after using virtual mouse and keyboard presses to release all.
        :return:
        """

        for _ in range(3):
            try:
                mouse.release(pynput.mouse.Button.right)
            except Exception as e:
                pass
            try:
                mouse.release(pynput.mouse.Button.left)
            except Exception as e:
                pass
            try:
                mouse.release(pynput.mouse.Button.middle)
            except Exception as e:
                pass

            # Release all keys on keyboard
            used_keys_list: [] = list(self.used_keys)
            for key in used_keys_list:
                try:
                    self.pynput_keyboard.release(key)
                except Exception as e:
                    pass

        logging.info("Agent released all virtual keys")

    def agent_create_new_world_if_done(self, seed: str = None, mode=0) -> bool:
        """
        Agent uses virtual inputs to navigate from the running game to main menu and creates a new world

        @return: if force stop
        """
        start_time = time.time()
        if mode==1:
            return self.arena_create_new_world()

        seed = str(seed) # validation

        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        logging.info("Agent creates world using virtual keyboard and mouse ...")
        self.release_all_keys()
        if is_inventory_open(agent_make_screenshot()):
            self.handle_close_inventory()

        self.pynput_keyboard.tap(pynput.keyboard.Key.esc)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True
        self.used_keys.add(pynput.keyboard.Key.esc)
        time.sleep(1)
        mouse.move(int(self.window_width/2 + 30), 750, absolute=True) # button main menu
        time.sleep(0.5)
        mouse.click()
        time.sleep(10)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        mouse.move(int(self.window_width / 2 + 30), 510, absolute=True) # button singleplayer
        time.sleep(0.5)
        mouse.click()
        time.sleep(5)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        mouse.move(1270, 870, absolute=True) # button create new world
        time.sleep(0.5)
        mouse.click()
        time.sleep(10)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        mouse.move(1300, 466, absolute=True) # button difficulty
        time.sleep(0.5)
        mouse.click()
        time.sleep(0.5)
        mouse.click()
        time.sleep(0.5)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        mouse.move(1300, 800, absolute=True)  # button world settings
        time.sleep(0.5)
        mouse.click()
        time.sleep(0.5)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        if seed is not None:
            mouse.move(1300, 300, absolute=True)  # button world type
            time.sleep(0.5)
            mouse.click()
            time.sleep(0.5)
            for char in seed:
                self.pynput_keyboard.tap(str(char))
                self.used_keys.add(str(char))
                time.sleep(0.1)
            time.sleep(0.5)

        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        mouse.move(1300, 466, absolute=True)  # button world type
        time.sleep(0.5)
        mouse.click()
        time.sleep(0.5)
        mouse.click()
        time.sleep(0.5)
        mouse.click()
        time.sleep(0.5)
        mouse.click()
        time.sleep(2)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        mouse.move(1300, 543, absolute=True)  # button customize
        time.sleep(0.5)
        mouse.click()
        time.sleep(2)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        mouse.move(1467, 237, absolute=True)  # scroll bar
        time.sleep(0.5)
        mouse.press()
        time.sleep(0.5)
        mouse.move(0, 150, absolute=False, duration=1)  # scroll bar
        time.sleep(0.5)
        mouse.release()
        time.sleep(0.5)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        mouse.move(613, 630, absolute=True)  # button forest
        time.sleep(0.5)
        mouse.click()
        time.sleep(0.5)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        mouse.move(630, 963, absolute=True)  # button done
        time.sleep(0.5)
        mouse.click()
        time.sleep(2)
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        mouse.move(630, 963, absolute=True)  # button create world
        time.sleep(0.5)
        mouse.click()
        time.sleep(0.5)
        self.release_all_keys()

        print("Time needed to create new world: " + str(time.time()-start_time) + "s")
        return False

    def exec_command(self, order="kill"):
        time.sleep(1)
        self.pynput_keyboard.tap('t')
        time.sleep(0.2)
        with self.pynput_keyboard.pressed(Key.shift):
            self.pynput_keyboard.tap('7')
        time.sleep(0.2)
        for char in order:
            self.pynput_keyboard.tap(str(char))
            time.sleep(0.1)
        time.sleep(0.2)
        self.pynput_keyboard.tap(Key.enter)
        time.sleep(0.2)
    def arena_create_new_world(self) -> bool:
        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True
        self.exec_command("kill @e[type=item]")
        time.sleep(2)
        self.exec_command("clear")
        time.sleep(2)
        self.exec_command("kill")
        time.sleep(2)
        self.click_on_respawn_button()
        time.sleep(3)
        self.exec_command("clone -3 -60 3 67 -56 -3 -30 60 3")  #("clone -5 94 10 19 100 33 50 94 10")
        time.sleep(2)
        self.exec_command("time set day")
        time.sleep(2)

        if force_stop_if_app_not_found(get_application_name()):
            self.release_all_keys()
            return True

        return False


