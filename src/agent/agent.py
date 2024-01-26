import json
import logging
import os
import time
import traceback
from collections import OrderedDict, deque
from threading import Thread

import cv2
import numpy as np
import pymsgbox
import torch
import wandb
from PIL import Image
from dotenv import load_dotenv
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from torchvision import transforms

from src.agent.actionExecution import ActionExecution
from config.train_config import get_train_config
from src.a3c.a3c_vit_gru import A3CMcRlNet
from src.a3c.architecture_factory import get_net_architecture
from src.agent.agent_http import http_get_training_config, http_post_gradients_file, \
    http_get_weights_file, http_post_end_of_epoch_data, http_get_global_agents_total_epochs, \
    http_post_reconstructed_map, http_get_model_info
from src.agent.observation.observation import Observation
from src.agent.observation.observe_death_screen import is_death_screen_shown
from src.agent.observation.observe_inventory.libs.observe_inventory_classify import load_image_classifier_model
from src.agent.observation.observe_inventory.observe_inventory import count_item
from src.agent.reward_function import get_inventory_reward, is_task_done
from src.common.action_space import get_action_dict_for_action_id, get_all_action_ids, \
    get_random_action_equal_distributed
from src.common.agent_map.agent_map import draw_map_for_agent
from src.common.countdown import countdown
from src.common.dummy_transitions import get_dummy_transition_seq
from src.common.env_utils.environment_info import get_application_name
from src.common.helpers.helpers import save_dict_as_json
from src.common.load_pretrained_vision_encoder import load_pretrained_vision_encoder_weights, check_for_pretrained_model
from src.common.observation_keys import health_key, hunger_key, experience_key, level_key, \
    POV_WIDTH, POV_HEIGHT, inventory_key, view_key
from src.common.persisted_memory import PersistedMemory
from src.common.resource_metrics import get_resource_metrics
from src.common.terminal_state import TerminalState
from src.common.transition import Transition
from src.dataloader.torch_mc_rl_data import AgentCustomDataset, DatasetForBatches
from src.dataloader.transform_functions import get_2D_image_of_last_3D_img_in_batch, get_concat_h
from src.trainers.a3c_functions import save_gradients
from src.trainers.a3c_trainer import A3CTrainer
from src.trainers.base_trainer import BaseTrainer
from src.agent.observation.agent_make_screenshot import agent_make_screenshot
from src.common.force_stop_if_app_not_found import force_stop_if_app_not_found

def str2bool(v):
  return str(v).lower() in ("yes", "true", "t", "1")

device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
load_dotenv()


class McRlAgent:

    def __init__(self, agent_id="0", generation_id: str = "0", session_id: str = "0", dry_run: bool = False,
                 local_filesystem_store_root_dir: str = "D:/McRlAgent",
                 mode="random_exploration", t_per_second=5, logging_with_wandb=True,
                 wandb_project_name: str = "mc_agent", world_seed: str = "3"):
        logging.info("initialize Agent...")
        self.train_config: dict = get_train_config()  # local config to get global net config
        self.local_global_net = self.train_config["local_global_net"]

        # Displays popups on top of the screen showing textual information such as choosen action.
        # Popups are not included in the screenshots during observation
        if "show_popups" in self.train_config:
            self.show_popups = self.train_config["show_popups"]
        else:
            self.show_popups = True
        self.dry_run: bool = dry_run  # in dry run mode no action is executed (for testing purpose only)
        self.mode: str = mode.lower()  # "train_a3c" | "eval_a3c"
        self.global_net_address = self.train_config["global_net_address"]
        if not self.local_global_net:
            self.get_train_config_from_global_net() # get same config as global net is using
        logging.info("using train_config of global net")
        logging.info(self.train_config)
        self.action_dim = self.train_config["action_dim"]
        self.action_execution = None


        if self.local_global_net:
            self.global_net_trainer = A3CTrainer(train_config=self.train_config)
            self.global_net_trainer.initialize_training()

        self.logging_with_wandb = logging_with_wandb
        if self.local_global_net:
            self.logging_with_wandb = False

        self.agent_id: str = str(agent_id)
        self.generation_id: str = str(generation_id)
        self.session_id: str = str(session_id)
        if "global_net_run_id" in self.train_config:
            self.session_id = self.train_config["global_net_run_id"]
        self.run_id = self.session_id + "_" + self.agent_id
        self.world_seed = world_seed
        self.world_created_automatically_current_epoch = False
        self.force_stop: bool = False  # force stop agent (emergency switch off)
        self.local_filesystem_store_root_dir = local_filesystem_store_root_dir

        self.using_a3c = mode == "train_a3c" or mode == "eval_a3c"
        self.metrics: dict = {}
        self.metrics_wandb_only: dict = {}
        self.done: bool = False  # agent epoch is done if terminal state.
        self.dead: bool = False  # agent is dead
        self.epoch = 0
        self.global_timestep = 0  # current timestep over multiple epochs
        self.epoch_timestep = 0
        self.discount_factor_gamma = 0.99  # gamma according to A3C paper Exp Setup
        self.pov_width = POV_WIDTH
        self.pov_height = POV_HEIGHT
        self.persisted_memory: PersistedMemory = PersistedMemory(img_shape=(self.pov_height, self.pov_width, 3),
                                                                 session_id=self.session_id,
                                                                 agent_id=self.agent_id,
                                                                 generation_id=self.generation_id,
                                                                 persist_to_local_filesystem=True,
                                                                 local_filesystem_store_root_dir=self.local_filesystem_store_root_dir)

        # most basic exploration method is epsilon-greedy, with a probability of 1 - epsilon exploitation is used
        self.epsilon_max = self.train_config["epsilon_max"]  # exploration rate at the beginning
        self.epsilon = self.epsilon_max
        self.epsilon_decay = self.train_config[
            "epsilon_decay"]  # decay rate that is applied after each experience replay
        self.epsilon_min = self.train_config["epsilon_min"]  # minimum exploration rate
        if self.mode == "eval_a3c":
            self.epsilon_max = 0
            self.epsilon = 0
            self.epsilon_min = 0
        self.output_dir = "./tmp/agent_net/" + self.train_config["net_architecture"] + "/"
        self.weights_file_path = os.path.join(self.output_dir + "global_net_weights.pth")
        self.gradients_file_path = os.path.join(self.output_dir + "agent" + self.agent_id + "_gradients_data.pth")

        self.timesteps_per_second: int = t_per_second
        self.timestep_length_sec: float = self.get_timestep_length_sec(self.timesteps_per_second)
        self.positive_reward = 0
        self.previous_inventory_reward = 0
        self.positive_reward_value = 1  # for each item
        self.next_state: dict = None
        self.max_steps_per_epoch: int = self.train_config["steps_per_epoch"]
        self.score = 0  # total reward of actual episode
        self.run = None  # wandb run
        self.model = None
        self.icon_classifier_model, self.icon_classifier_train_config, self.icon_classifier_int_2_label = \
            load_image_classifier_model(model_dir=self.train_config[
                "icon_classifier_model_dir"])  # makes one prediction to avoid lag on first use with GPU

        # dummy transitions for first transition sequences
        self.dummy_transition_seq = get_dummy_transition_seq(self.train_config["input_depth"],
                                                             self.timestep_length_sec,
                                                             screenshot=self.make_screenshot())

        self.gamma = self.train_config["gamma"]  # discount factor according to A3C paper Exp Setup
        self.entropy_coef = self.train_config["entropy_coef"]  # entropy regularization according to A3C paper Exp Setup
        self.batch_size = self.train_config["batch_size"]
        self.rewards = deque([], maxlen=self.batch_size)
        self.actions = deque([], maxlen=self.batch_size)
        self.states = deque([], maxlen=self.batch_size)
        self.create_new_world_after_epoch = self.train_config["create_new_world_after_epoch"]
        self.project_name = wandb_project_name
        ssl_head_args = BaseTrainer.get_ssl_head_args(self.train_config)
        self.model = get_net_architecture(net_architecture=self.train_config["net_architecture"],
                                          ssl_head_args=ssl_head_args,
                                          train_config=self.train_config,
                                          device=device)
        self.model.to(device)
        load_pretrained_vision_encoder_weights(check_for_pretrained_model(self.train_config["pretrained_vision_encoder_model_path"]), self.model,
                                               self.train_config["pretrained_vision_encoder_model_path"], device)
        self.model.train()
        if self.train_config["freeze_pretrained_vision_encoder_weights"] and self.train_config["unfreeze_pretrained_vision_encoder_weights"]:
            logging.info("Frozen layers will be unfrozen after " + str(self.train_config["finetuning_warmup_steps"]) + " steps")
        logging.info("NN model initialized")

        if logging_with_wandb:
            wandb_sweep_config = {
                'name': 'sweep',
                'method': 'grid',  # grid, random, bayes
                'metric': {'goal': 'minimize', 'name': 'loss'},
                'parameters': {
                    "seq_len": {"value": self.train_config["input_depth"]},
                    "epsilon_max": {"value": self.epsilon_max},
                    "epsilon_min": {"value": self.epsilon_min},
                    "epsilon_decay": {"value": self.epsilon_decay},
                    "max_steps_per_epoch": {"value": self.max_steps_per_epoch},
                    "pretrained_vision_encoder": {"value": check_for_pretrained_model(self.train_config["pretrained_vision_encoder_model_path"])},
                    "freeze_pretrained_vision_encoder_weights": {"value": self.train_config["freeze_pretrained_vision_encoder_weights"]},
                    "agent_id": {"value": str(self.agent_id)},
                    "gamma": {"value": self.gamma},
                    "entropy_coef": {"value": self.entropy_coef},
                    "lr_global_net": {"value": self.train_config["learning_rate"]}
                }
            }
            for key in self.train_config:
                wandb_sweep_config["parameters"][key] = {"values": [self.train_config[key]]}
            if self.logging_with_wandb:
                if not self.dry_run:
                    self.run = wandb.init(project=self.project_name, config=wandb_sweep_config, id=self.run_id,
                                          resume=True)
                else:
                    self.run = wandb.init(project=self.project_name, config=wandb_sweep_config)

            wandb.watch(self.model)

        self.config_filename = self.output_dir + "/train_config.json"
        self.latest_metrics_file_path = self.output_dir + "/latest_agent_metrics.json"
        self.latest_weights_file_path = self.output_dir + "/latest_agent_model_state_dict.pt"
        save_dict_as_json(self.train_config, None, self.config_filename)

        logging.info("Agent initialized")

    def remember(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.rewards = deque([], maxlen=self.batch_size)
        self.actions = deque([], maxlen=self.batch_size)
        self.states = deque([], maxlen=self.batch_size)

    def get_train_config_from_global_net(self):
        if not self.dry_run and self.mode == "train_a3c":
            self.train_config: dict = http_get_training_config(self.global_net_address)
        else:
            self.train_config: dict = get_train_config()  # dry run, use local config

    def initialize_agent_epoch(self, generation_id):
        try:
            self.epoch_start_time = time.time()
            self.action_execution: ActionExecution = ActionExecution()
            if not self.dry_run:
                self.create_new_world_for_next_epoch()
            self.epoch_timestep = 0  # timestep this epoch
            self.global_timestep = self.epoch * self.max_steps_per_epoch
            self.generation_id: str = generation_id
            self.persisted_memory.generation_id = generation_id
            self.next_state: dict = None
            self.done: bool = False  # agent epoch is done
            self.score = 0  # total reward of actual 0
            self.positive_reward = 0
            self.previous_inventory_reward = 0
            self.metrics = {}
            self.metrics_wandb_only: dict = {}
            self.metrics["deaths"] = 0
            self.metrics["reward"] = 0
            self.metrics["score"] = 0
            self.metrics_wandb_only["max_gradient"] = 0
            self.metrics_wandb_only["min_gradient"] = 0
            self.metrics_wandb_only["mean_gradient"] = 0
            self.clear_memory()
            
            # dummy transitions for first transition sequences
            self.dummy_transition_seq = get_dummy_transition_seq(self.train_config["input_depth"],
                                                                 self.timestep_length_sec,
                                                                 screenshot=self.make_screenshot())

            if self.mode == "train_a3c":
                if self.dry_run:
                    logging.warning("\nDRY RUN: no action execution and no connection to real global net!\n")
                if not self.dry_run:
                    try:
                        self.get_and_load_global_net_weights()
                        try:
                            self.model.print_info()
                        except Exception as e:
                            logging.error(e)
                            traceback.print_exc()
                    except Exception as e:
                        logging.error(e)
                        traceback.print_exc()
            elif self.mode == "eval_a3c":
                self.model_weights_dict: OrderedDict = torch.load(self.weights_file_path, map_location=device)
                self.model.load_state_dict(self.model_weights_dict)
                self.model.eval()

        except Exception as e:
            self.force_stop = True
            logging.error(e)
            traceback.print_exc()
        logging.info("Agent epoch initialized")

    def run_agent_epoch(self, generation_id: str) -> bool:
        """
        Training loop for agent epoch
        @param generation_id: id of generation (timestamp) using sync or async start of multiple agents
        @return: boolean: True if force stop was called
        """

        self.initialize_agent_epoch(generation_id=generation_id)

        if self.force_stop:
            logging.error("ERROR: Agent loop could not be started.")
            return True

        application_name = get_application_name()
        if self.dry_run:
            if self.action_execution is not None:
                self.action_execution.execution_disabled = True
        else:
            self.action_execution.execution_disabled = False
        try:
            self.model.print_info()
            logging.info(get_resource_metrics())
            logging.info("Agent mode: " + str(self.mode))
            logging.info("Epoch: " + str(self.epoch))
            logging.info("Continue with timestep: " + str(self.global_timestep))
            logging.info("timestep length: " + str(self.timestep_length_sec) + "s")
            logging.info("max epochs: " + str(self.max_steps_per_epoch))

            logging.info("Agent executes actions after countdown: ")
            logging.info("Game and mouse cursor must be on main screen using fullscreen mode!")
            if self.dry_run:
                self.show_popup("Script: RL Agent starts in 5 sec. Please focus " + application_name)
                countdown(5, optional_text="Agent starting. Please focus " + application_name)
            start_time = time.time()

            if is_death_screen_shown(agent_make_screenshot()):
                self.action_execution.click_on_respawn_button()
                time.sleep(10)
            self.world_created_automatically_current_epoch = False  # Reset
            try:
                while not self.done and self.epoch_timestep < self.max_steps_per_epoch and not self.force_stop:
                    # timestep length = timestep_length_sec
                    timestep_start_time: float = time.time()
                    logging.info(
                        "epoch " + str(self.epoch) +
                        " step " + str(self.epoch_timestep + 1) + "/" + str(
                            self.max_steps_per_epoch))
                    self.metrics["step"] = self.global_timestep
                    self.metrics["epoch"] = self.epoch

                    # For security check focused application
                    sec_start_time = time.time()
                    if not self.dry_run:
                        self.force_stop = force_stop_if_app_not_found(application_name)
                    self.metrics["check_application_time"] = time.time() - sec_start_time

                    if self.next_state is None:
                        state, _, _ = self.observe_env(self.force_stop, self.make_screenshot())
                    else:
                        state = self.next_state  # next state is previous state. Take previous state

                    input_start_time: float = time.time()
                    state_seq = self.get_current_state_sequence()
                    self.metrics["time_get_state_seq"] = time.time() - input_start_time

                    action_id = self.choose_action(state_seq)  # choose action depending on state
                    action_dict: dict = get_action_dict_for_action_id(action_id)  # get subactions for action

                    if self.force_stop:
                        break  # Fatal error or abort called by user. Do not execute action

                    logging.info("execute action " + str(action_id))
                    self.metrics["action_id"] = action_id
                    # Take a step in the environment
                    start_env_step = time.time()
                    self.force_stop, self.next_state, terminal_state, reward, truncated, terminated = self.env_step(
                        action_id, timestep_start_time)

                    self.metrics["reward"] = reward
                    logging.info("reward: " + str(reward))
                    self.remember(state_seq, action_id, reward)
                    self.score += reward

                    """
                    The score is the sum of the collected target items per epoch.
                    The agent can collect any number of blocks in a step. Recognition of the number of agents is not 
                    100% accurate while the collection of items is animated in the toolbar, 
                    because machine learning methods are in use.
                    Score is set to the current inventory reward if recognition of the rewards is slightly incorrect.         
                    """
                    self.score = min(self.score, self.metrics["current_inventory_reward"])
                    self.metrics["score"] = self.score
                    logging.info("score: " + str(self.score))
                    self.metrics["env_step_time"] = time.time() - start_env_step

                    self.done = terminated  # death = terminal state
                    if self.epoch_timestep == self.max_steps_per_epoch-1:  # after last step = terminal state
                        self.done = True
                    self.metrics["done"] = int(self.done)
                    self.metrics["truncated"] = int(truncated)

                    agent_status: dict = {
                        health_key: self.next_state[health_key],
                        hunger_key: self.next_state[hunger_key],
                        level_key: self.next_state[level_key],
                        experience_key: self.next_state[experience_key],
                    }
                    self.metrics.update(agent_status)
                    self.metrics["inv_all"] = count_item(self.next_state[inventory_key], None)
                    self.metrics["inv_state"] = self.next_state[inventory_key]
                    # Transition: SARS´
                    # next state S´ not saved, because of memory optimization. Next state is S of the next transition.
                    transition: Transition = Transition(t=self.global_timestep, state=state, action_id=action_id,
                                                        action=action_dict, reward=reward,
                                                        terminal_state=str(terminal_state),
                                                        timestamp=timestep_start_time)

                    if self.persisted_memory is not None:
                        self.persisted_memory.save_timestep_in_ram(transition=transition)

                    if self.using_a3c:
                        if (self.epoch_timestep > 0 and (
                                self.epoch_timestep + 1) % self.batch_size == 0) or self.done:
                            calc_start_time = time.time()
                            loss = self.calc_loss(self.done)
                            logging.info("loss: " + str(self.metrics["loss"]))
                            logging.info("backpropagation ...")
                            loss.backward()
                            logging.info(get_resource_metrics())
                            self.metrics["backward_calc_time"] = float(time.time() - calc_start_time)
                            self.calculate_and_send_gradients_to_global_net() # update global net

                    self.metrics.update(get_resource_metrics())

                    logging.info("--")
                    self.metrics["agent_epoch_step"] = int(self.epoch_timestep)
                    wandb_data: dict = self.metrics_wandb_only
                    wandb_data.update(self.metrics)
                    if self.epoch_timestep != self.max_steps_per_epoch-1 and not self.done:  # skip last to log at end of epoch
                        wandb.log(wandb_data)

                    self.global_timestep += 1
                    self.epoch_timestep += 1
                    # very last line of timestep
                    # end of step time regulation
                    time_needed = time.time() - timestep_start_time
                    self.metrics["step_time_no_sleep"] = float(time_needed)

                    sleep_time = max(self.timestep_length_sec - time_needed, 0)  # Waiting time for calculations
                    time.sleep(sleep_time)
                    self.metrics["sleep_time"] = sleep_time
                    logging.info("step_time: " + str(round(time.time() - timestep_start_time, 3)))
                    self.metrics["step_time"] = time.time() - timestep_start_time

                    if self.force_stop or self.done:
                        break
                    # end of step
                # end of epoch
            except KeyboardInterrupt:
                self.force_stop = True
                logging.warning("McRlAgent: KeyboardInterrupt. Kindly stopping agent...")
            except Exception as e:
                self.force_stop = True
                logging.error(e)
                traceback.print_exc()
            # cleanup epoch
            if self.action_execution is not None:
                self.action_execution.release_all_keys()
            agent_act_time = time.time() - start_time
            self.metrics["agent_total_epoch_act_time"] = agent_act_time
            logging.info("agent total time needed: " + str(agent_act_time))
            self.metrics["done"] = int(self.done)
            start_persisting_time = time.time()
            agent_map = None
            last_transition: Transition = self.persisted_memory.transitions[-1]
            last_image: np.ndarray = cv2.cvtColor(last_transition.state[view_key], cv2.COLOR_BGR2RGB)
            if self.persisted_memory is not None:
                if not self.dry_run:
                    agent_map = draw_map_for_agent(self.persisted_memory.transitions,
                                                       step_length=self.timestep_length_sec)
                if self.train_config["persist_transitions"] and not self.dry_run:
                    self.persisted_memory.save_from_ram_to_persisted(only_delete=False,
                                                                     generation_id=self.generation_id)
                else:
                    self.persisted_memory.save_from_ram_to_persisted(only_delete=True,
                                                                     generation_id=self.generation_id)
            self.metrics["persisting_time"] = time.time() - start_persisting_time
            self.metrics["epoch_time"] = time.time() - self.epoch_start_time

            try:
                if self.mode == "train_a3c":
                    if agent_map is not None and self.epoch > 0 and self.epoch % 10 == 0:
                        pil_image = Image.fromarray(agent_map)
                        self.metrics_wandb_only["agent_map"] = wandb.Image(pil_image)
                        # Sending the map takes place asynchronously in a thread to the global net for logging purpose

                        try:
                            thread2 = Thread(target=self.send_reconstructed_map_to_global_net, args=(agent_map,))
                            thread2.start()
                        except Exception as e:
                            logging.error(e)
                            traceback.print_exc()

                if self.epoch < 2:
                    self.metrics_wandb_only["agent_screenshot"] = wandb.Image(last_image)
                else:
                    self.metrics_wandb_only["agent_screenshot"] = None


            except Exception as e:
                logging.error(e)
                traceback.print_exc()

            wandb_data: dict = self.metrics_wandb_only
            wandb_data.update(self.metrics)

            wandb.log(wandb_data)  # log last step at the end of the entire epoch

            if not self.dry_run and self.mode == "train_a3c":
                data_dict: dict = {
                    "agent_id": self.agent_id,
                    "add_agent_epochs": 1,
                    "epoch": self.epoch}
                if not self.local_global_net:
                    try:
                        http_post_end_of_epoch_data(self.global_net_address, data_dict)
                    except Exception as e:
                        logging.error(e)
                        traceback.print_exc()
                else:
                    self.global_net_trainer.process_end_of_epoch_data_from_agent(data_dict)

            #self.save_model(self.model, self.metrics, self.latest_metrics_file_path, self.latest_weights_file_path)
            if self.force_stop:
                pymsgbox.alert('Agent force stop: ' + str(self.force_stop)
                               + " done: " + str(self.done) + ". Press OK to continue.", 'Message')
                return self.force_stop

            if not self.dry_run:
                if self.create_new_world_after_epoch:
                    logging.info("Agent creates and joins new world ...")
                    self.create_new_world_for_next_epoch()

        except Exception as e:
            self.force_stop = True
            logging.error(e)
            traceback.print_exc()
            if self.action_execution is not None:
                self.action_execution.release_all_keys()

        self.epoch += 1
        return self.force_stop

    def send_reconstructed_map_to_global_net(self, agent_map):
        http_post_reconstructed_map(self.global_net_address, agent_map, self.agent_id)
        return

    def calc_R(self, done):
        states = self.states

        _, v = self.predict(states)

        R = v[-1] * (1 - int(done))

        self.metrics["next_value"] = float(R.cpu().detach().numpy())
        batch_return = []
        rewards = list(self.rewards)[::-1]
        for reward in rewards:
            R = reward + self.gamma * R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return, dtype=torch.float).to(device)
        self.metrics["mean_discounted_reward"] = float(batch_return.mean().cpu().detach().numpy())

        return batch_return

    def calc_loss(self, done):
        states = self.states
        actions = torch.tensor(self.actions, dtype=torch.float).to(device)  # used actions

        returns = self.calc_R(done)

        pi, values = self.predict(states)
        values = values.squeeze(1)  # keep batch dimension when batch size 1
        critic_loss = (returns - values) ** 2  # squared error

        probs = torch.softmax(pi, dim=-1)
        dist: Categorical = Categorical(probs)
        log_probs = dist.log_prob(actions)

        entropy = dist.entropy()
        actor_loss = -log_probs * (returns - values) - self.entropy_coef * entropy

        total_loss = (actor_loss + critic_loss).mean()

        self.metrics["calc_log_prob"] = float(log_probs.mean().detach().cpu().numpy())
        self.metrics["value"] = float(values.mean().cpu().detach().numpy())
        self.metrics["entropy"] = float(entropy.mean().cpu().detach().numpy())
        self.metrics["loss"] = float(total_loss.cpu().detach().numpy())
        self.metrics["actor_loss"] = float(actor_loss.mean().cpu().detach().numpy())
        self.metrics["critic_loss"] = float(critic_loss.mean().cpu().detach().numpy())
        self.metrics["advantage"] = float((returns - values).mean().cpu().detach().numpy())
        logging.info(get_resource_metrics())
        return total_loss

    def choose_action(self, state_seq):
        epsilon = self.epsilon
        exploration = np.random.rand() > 1 - epsilon

        # choose action based on epsilon-greedy strategy
        if exploration:
            logging.info("Exploration. epsilon: " + str(self.epsilon))
            action_id = get_random_action_equal_distributed(self.action_dim)
            self.metrics["predicted_action_id"] = None
            self.metrics["prob_predicted_action"] = None
        else:
            prediction_start_time: float = time.time()
            pi, _ = self.predict([state_seq])
            self.metrics["time_for_prediction"] = time.time() - prediction_start_time
            probs = torch.softmax(pi, dim=1)
            dist: Categorical = Categorical(probs)
            action_id = int(dist.sample().detach().cpu()[0])  # multinomial distribution
            self.metrics["predicted_action_id"] = action_id
            self.metrics["prob_predicted_action"] = float(probs[0].detach().cpu().numpy()[action_id])

        self.show_popup("action: " + str(action_id) + " |exploration: " + str(exploration), show_action=True)

        return action_id

    def env_step(self, action_id, timestep_start_time):
        """
        env_step inspired by env.step of https://gymnasium.farama.org/
        @param action_id:
        @return: force_stop, next_state, terminal_state, reward, truncated, dead
        """
        force_stop: bool = False
        if self.action_execution is not None:
            start_action_exec = time.time()
            self.action_execution.execute_timestep(action_id=action_id, action_dict=None,
                                                   timestep_length_sec=self.timestep_length_sec)

            # Minecraft: Dropped items have a delay of 10 ticks (half a second) between appearing and being able to be picked up
            time_needed = time.time() - timestep_start_time
            #if time_needed < self.timestep_length_sec:
            #    sleep_time = max(3.5 - time_needed, 0)
            #    time.sleep(sleep_time)

            time_for_step_end = self.timestep_length_sec / 20
            sleep_time = max(self.timestep_length_sec - time_for_step_end - time_needed, 0)  # Waiting time to execute action
            time.sleep(sleep_time)
            self.metrics["sleep_time"] = sleep_time
            self.metrics["action_exec_time"] = time.time() - start_action_exec

            # Security
            if self.action_execution.force_stop:
                force_stop = True
                logging.warning("Agent force stop. Stop called by ActionExecution instance.")

        # state after action
        start_observe_env = time.time()
        next_state, terminal_state, terminated = self.observe_env(self.force_stop, self.make_screenshot())
        self.metrics["observe_env_time"] = time.time() - start_observe_env

        current_inventory_reward: float = get_inventory_reward(state=next_state,
                                                               task_item_key=self.train_config["task_item_key"])
        # only gets more positive reward if more target items are in the inv than last time checked
        self.positive_reward = max(current_inventory_reward - self.previous_inventory_reward, 0)

        self.metrics["current_inventory_reward"] = current_inventory_reward
        self.previous_inventory_reward = current_inventory_reward
        reward = self.positive_reward

        truncated = False
        return force_stop, next_state, terminal_state, reward, truncated, terminated

    def get_and_load_global_net_weights(self):
        if self.mode == "train_a3c":
            if not self.local_global_net:
                weights_start_time: float = time.time()
                response_data: dict = http_get_weights_file(self.global_net_address, self.weights_file_path)

                if response_data is not None and "trainer_stopped" in response_data:
                    logging.info("Training stopped by global net")
                    self.force_stop = True
                self.metrics["time_get_weights"] = time.time() - weights_start_time

                if "status_code" in response_data and response_data["status_code"] == 200:
                    self.model_weights_dict: OrderedDict = torch.load(self.weights_file_path, map_location=device)
                    apply_weights_start_time = time.time()
                    self.model.load_state_dict(self.model_weights_dict)
                    self.metrics["time_apply_weights"] = time.time() - apply_weights_start_time
                    logging.info("NN model updated with weights from global net")
                    try:
                        update_epsilon_time: float = time.time()
                        response_data: dict = http_get_global_agents_total_epochs(self.global_net_address)

                        if response_data is not None and "trainer_stopped" in response_data:
                            logging.info("Training stopped by global net")
                            self.force_stop = True

                        if "agents_total_epoch" in response_data:
                            # update exploration rate dependent on total epochs of all agents
                            agents_total_epoch = response_data["agents_total_epoch"]
                            self.update_epsilon(agents_total_epoch)
                        self.metrics["time_update_epsilon"] = time.time() - update_epsilon_time
                    except Exception as e:
                        logging.error(e)
                        traceback.print_exc()

                """
                Values of param.requires_grad are not saved together with the weights. Therefore, an extra query 
                by the agent is necessary, which value param.requires_grad has per layer
                """
                model_info: dict = http_get_model_info(self.global_net_address)
                if "number_cnn_frozen_layers" in model_info:
                    try:
                        if self.train_config["freeze_pretrained_vision_encoder_weights"] and self.train_config[
                            "unfreeze_pretrained_vision_encoder_weights"]:
                            if model_info["number_cnn_frozen_layers"] == 0:
                                success = self.model.try_unfreeze_all_cnn_layers()
                                if success:
                                    print("Unfrozen all layers at step " + str(self.global_timestep))
                    except Exception as e:
                        logging.error(e)
                        traceback.print_exc()

            else:
                self.model_weights_dict = self.global_net_trainer.model.state_dict()
                self.model.load_state_dict(self.model_weights_dict)
                self.update_epsilon(self.global_net_trainer.agents_total_epochs)

    def update_epsilon(self, agents_total_epoch):
        self.epsilon = max(self.epsilon_min,
                           self.epsilon_max - (agents_total_epoch
                                               * self.epsilon_decay))
        self.metrics["epsilon"] = self.epsilon

    def calculate_and_send_gradients_to_global_net(self):
        try:
            if self.mode == "train_a3c":
                logging.info("saving gradients and metrics to " + self.gradients_file_path)
                saved_model_gradients = save_gradients(self.model)

                max_gradients = []
                for param in saved_model_gradients:
                    if param is not None:
                        max_gradients.append(torch.max(param))
                if len(max_gradients) > 0:
                    self.metrics_wandb_only["max_gradient"] = float(max(max_gradients).cpu().detach().numpy())
                    self.metrics_wandb_only["min_gradient"] = float(min(max_gradients).cpu().detach().numpy())
                    self.metrics_wandb_only["mean_gradient"] = float(np.mean(max_gradients))
                else:
                    self.metrics_wandb_only["max_gradient"] = 0
                    self.metrics_wandb_only["min_gradient"] = 0
                    self.metrics_wandb_only["mean_gradient"] = 0

                json.dumps(self.metrics)  # test json data
                gradients_data: dict = {
                    "agent_id": self.agent_id,
                    "metrics": self.metrics,
                    "gradients": saved_model_gradients,
                    "add_agent_steps": self.epoch_timestep + 1,  # +1 because counting from 0
                }

                if not self.local_global_net:
                    torch.save(gradients_data, self.gradients_file_path)
                    logging.info("sending gradients and metrics to A3C global net...")
                    start_time_gradients = time.time()
                    if not self.dry_run:
                        response: dict = http_post_gradients_file(self.global_net_address, self.gradients_file_path)
                        self.metrics["send_gradients_time"] = time.time() - start_time_gradients
                        if response is not None:
                            if "trainer_stopped" in response:
                                logging.info("Training stopped by global net")
                                self.force_stop = True

                            logging.info("gradients and metrics sent successfully")
                else:
                    self.global_net_trainer.process_gradients_and_metrics(gradients_data)
                    self.global_net_trainer.try_update()
                    if self.global_net_trainer.early_stopping.early_stop:
                        logging.info("Training stopped by global net")
                        self.force_stop = True
        except Exception as e:
            self.force_stop = True
            logging.error(e)
            traceback.print_exc()

    def observe_env(self, force_stop, screenshot):
        # begin observation of changed env
        observation: Observation = Observation(self.icon_classifier_model, self.icon_classifier_train_config,
                                               self.icon_classifier_int_2_label)
        dead: bool = False
        terminal_state = TerminalState.NONE  # default
        observation.process_screenshot(screenshot)  # reads health bar etc.


        if is_death_screen_shown(screenshot):
            dead = True
            terminal_state = TerminalState.DEATH
            logging.info("Agent dead")
            self.action_execution.click_on_respawn_button()
            time.sleep(10)
            self.metrics["deaths"] += 1

        # inventory_open: bool = is_inventory_open(screenshot) # enable this when using the full inventory
        inventory_open = False
        observation.set_inventory_status(is_inventory_open=inventory_open)

        if not dead and not force_stop:
            read_inventory_start_time: float = time.time()
            observation.process_inventory_screenshot(screenshot, inventory_open)
            observe_inv_time = time.time() - read_inventory_start_time
            self.metrics["observe_inv_time"] = observe_inv_time
        state: dict = observation.get_actual_state()  # includes pov screenshot

        if dead:
            state[health_key] = 0

        terminated = dead
        return state, terminal_state, terminated

    def predict(self, batch: []):
        for i, d in enumerate(batch):
            batch[i] = d
        dataset = DatasetForBatches(batch)
        loader = DataLoader(
            dataset,
            batch_size=len(batch)
        )

        x_3d_image = None
        x_tensor_state_seq = None
        for b in loader:
            x_3d_image = b["tensor_image"].to(device)  # 2d image sequence as 3d image
            x_tensor_state_seq = b["tensor_state_seq"].to(device)

        x_3d_image = torch.squeeze(x_3d_image, 1)
        x_tensor_state_seq = torch.squeeze(x_tensor_state_seq, 1)

        logit, value = self.model(x_3d_image, x_tensor_state_seq)

        if self.global_timestep == self.train_config["input_depth"] + 1 and self.epoch < 2:
            # upload sample of agent´s vision
            concat_img = self.get_3D_image_as_2D_from_last_tensor_in_batch(x_3d_image)

            concat_img.save("tmp/img_input.png")
            self.metrics_wandb_only["img_input"] = wandb.Image(concat_img)
        else:
            self.metrics_wandb_only["img_input"] = None

        return logit, value

    def get_3D_image_as_2D_from_last_tensor_in_batch(self, tensor):
        if len(tensor.shape) == 4:
            return transforms.ToPILImage()(tensor[0])
        else:
            concat_img = None
            for i in range(self.train_config["input_depth"]):
                img = get_2D_image_of_last_3D_img_in_batch(tensor, image_index=i, squeeze=False)
                if concat_img is None:
                    concat_img = img
                else:
                    concat_img = get_concat_h(concat_img, img)
            return concat_img

    def get_current_state_sequence(self):
        last_n = (-self.train_config["input_depth"] - 1)
        if not self.enough_transitions_for_seq_available():
            transition_seq = self.persisted_memory.transitions[last_n:] + self.dummy_transition_seq
        else:
            transition_seq = self.persisted_memory.transitions[last_n:]  # transitions from last n steps
        transition_seq = transition_seq[last_n:]

        if self.train_config["pretrain_architecture"].lower() == "swin_vit":
            seq_to_3d_image = True
        else:
            seq_to_3d_image = False

        train_dataset: AgentCustomDataset = AgentCustomDataset(transition_seq=transition_seq,
                                                               x_depth=self.train_config["input_depth"],
                                                               width_2d=self.train_config["img_size"][1],
                                                               height_2d=self.train_config["img_size"][0],
                                                               cache=None,
                                                               state_dim=4,
                                                               seq_to_3d_image=seq_to_3d_image)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        batch = None
        for step, b in enumerate(train_loader):
            batch = b
            break

        batch["tensor_image"] = batch["tensor_image"].cpu()  # 2d image sequence as 3d image
        batch["tensor_state_seq"] = batch["tensor_state_seq"].cpu()

        return batch

    def enough_transitions_for_seq_available(self):
        return len(self.persisted_memory.transitions) > self.train_config["input_depth"]

    def create_new_world_for_next_epoch(self):
        self.show_popup("Script: create_new_world_for_next_epoch")
        if not self.force_stop and not self.world_created_automatically_current_epoch:
            self.force_stop = self.action_execution.agent_create_new_world_if_done(self.world_seed,
                                                                                   mode=self.train_config["world_creation_mode"])
            time.sleep(25)  # wait until env is loaded
        if not self.force_stop:
            self.world_created_automatically_current_epoch = True

    def make_screenshot(self) -> np.ndarray:
        return agent_make_screenshot()

    def show_popup(self, text, show_action: bool = False):
        if not self.show_popups:
            return

        window_base_name = "popup"
        window_name = window_base_name + str(self.epoch_timestep)

        if show_action:
            n = 10
            recent_actions = list(self.actions)[-n:]
            recent_actions.reverse()

            recent_actions_str = ""
            for action_id in recent_actions:
                recent_actions_str += str(action_id)
                recent_actions_str += " <- "
            text += (" |step:" + str(self.epoch_timestep) + "|time: " + str(round(self.time_needed_previous_step, 3))
                     + " |score:" + str(self.score) + " |prev: " + recent_actions_str)
        else:
            cv2.destroyAllWindows()
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img = cv2.resize(img, (1400, 20))
        cv2.namedWindow(window_name)  # Create a named window
        cv2.moveWindow(window_name, 300, -32)
        coordinates = (0, 15)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 255, 255)
        thickness = 1
        img = cv2.putText(img, text,
                          coordinates, font, fontScale, color,
                          thickness, cv2.LINE_AA)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(window_name, img)
        cv2.waitKey(1)
        try:
            if show_action:
                cv2.destroyWindow(window_base_name + str(
                    int(self.epoch_timestep - self.timestep_length_sec + 1)))  # close previous n window
            else:
                cv2.destroyAllWindows()
        except Exception as e:
            logging.error(e)

    @staticmethod
    def sec_to_timesteps(seconds, timestep_length_sec):
        return int(seconds / timestep_length_sec)

    @staticmethod
    def timesteps_to_sec(timesteps, timestep_length_sec):
        return int(timesteps * timestep_length_sec)

    @staticmethod
    def get_timestep_length_sec(timesteps_per_second) -> float:
        return round(1 / timesteps_per_second, 2)

    @staticmethod
    def save_model(model: A3CMcRlNet, metrics: dict, metrics_filepath, weights_filepath):
        save_dict_as_json(metrics, None, metrics_filepath)
        torch.save(model.state_dict(), weights_filepath)  # only encoder weights (model.swinViT)
        print("Model saved to dir:", weights_filepath)
