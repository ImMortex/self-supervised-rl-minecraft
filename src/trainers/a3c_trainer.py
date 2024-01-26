import logging
import math
import os
import time
import traceback
from collections import OrderedDict, deque
from datetime import datetime
from datetime import timedelta

import coloredlogs
import cv2
import numpy as np
import torch
import wandb
from PIL import Image
from dotenv import load_dotenv
from torch import nn
from torch.optim import Optimizer, Adam, RMSprop
from torch.optim.lr_scheduler import LinearLR

from src.a3c.a3c_vit_gru import A3CMcRlNet
from src.a3c.architecture_factory import get_net_architecture
from src.common.countdown import countdown
from src.common.early_stopping import EarlyStopping
from src.common.helpers.helpers import save_dict_as_json
from src.common.load_pretrained_vision_encoder import load_pretrained_vision_encoder_weights, check_for_pretrained_model
from src.common.resource_metrics import get_resource_metrics
from src.trainers.a3c_functions import save_gradients
from src.trainers.base_trainer import BaseTrainer
from src.trainers.shared_optimizer import SharedAdam, SharedRMSprop

coloredlogs.install(level='INFO')
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))

load_dotenv()
# CHECKPOINT_WAIT_TIME: not using value of training config here because to be able to change it during training
CHECKPOINT_WAIT_TIME = int(os.getenv("CHECKPOINT_WAIT_TIME"))
if CHECKPOINT_WAIT_TIME is None:
    CHECKPOINT_WAIT_TIME = 0  # not loading checkpoint by default


class A3CTrainer(BaseTrainer):
    """
    global A3C net trainer
    """

    def __init__(self, train_config: dict):
        super().__init__(train_config)
        logging.info("Initialize Global A3C net ...")
        self.train_config: dict = train_config
        logging.info("train config: ")
        logging.info(self.train_config)
        self.pretrained_vision_encoder_model_path = None
        self.use_pretrained_vision_encoder = check_for_pretrained_model(self.train_config["pretrained_vision_encoder_model_path"])
        if self.use_pretrained_vision_encoder:
            self.pretrained_vision_encoder_model_path = self.train_config["pretrained_vision_encoder_model_path"]

        self.learning_rate = self.train_config["learning_rate"]
        self.metrics: dict = {}
        self.metrics_wandb_only: dict = {}
        self.agent_data: dict = {}  # last metrics of each agent
        self.agent_reconstructed_maps: dict = {}  # last reconstructed map of each agent
        self.best_mean_score = 0
        self.best_score = 0

        self.start_time = time.time()
        self.run = None
        self.model_artifact = None  # is set later in code
        self.gradients_queue = deque()
        self.update_active = False  # Flag
        self.epoch: int = 0
        self.global_step_counter_T_max: int = self.train_config["global_step_counter_T_max"]
        self.step: int = 0
        self.agents_total_steps: int = 0
        self.agents_total_epochs: int = 0
        self.output_dir = "tmp/global_net/" + self.train_config["net_architecture"] + "/"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.checkpoint_dir = self.output_dir + "checkpoint"
        self.checkpoint_filename = "global_net_checkpoint.pt"
        self.checkpoint_path = self.checkpoint_dir + "/" + self.checkpoint_filename
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        self.output_dir_gradients = os.path.join(self.output_dir, "agents")
        if not os.path.exists(self.output_dir_gradients):
            os.makedirs(self.output_dir_gradients)

        # file get updated by update method
        self.best_metrics_file_path: str = os.path.join(self.output_dir, "best_metrics.json")
        self.best_weights_file_path: str = os.path.join(self.output_dir, "best_global_model_weights_dict.pth")
        self.latest_metrics_file_path: str = os.path.join(self.output_dir, "latest_metrics.json")
        self.latest_weights_file_path: str = os.path.join(self.output_dir,
                                                          "global_model_weights_dict_epoch_" + str(self.step) + '.pth')

        self.project_name = "global_A3CMcRlNet_exp"
        self.session_id: str = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S_%f"))
        self.run_id = self.session_id + "A3C"
        if self.use_pretrained_vision_encoder:
            self.run_id += "_PRE"
        self.start_time = time.time()

        self.model = get_net_architecture(net_architecture=self.train_config["net_architecture"],
                                          ssl_head_args=self.ssl_head_args,
                                          train_config=self.train_config,
                                          device=device)

        load_pretrained_vision_encoder_weights(self.use_pretrained_vision_encoder, self.model,
                                               self.pretrained_vision_encoder_model_path, device)

        self.model.to(device)
        self.model.train()

        self.early_stopping = EarlyStopping(tolerance=self.train_config["early_stopping_tolerance"],
                                            target_score=self.train_config["target_score"])

        self.optimizer: Optimizer = None
        if self.train_config["optimizer"] == "shared_adam":
            self.model.share_memory()
            self.optimizer: SharedAdam = SharedAdam(params=self.model.parameters(), lr=self.learning_rate)
        elif self.train_config["optimizer"] == "adam":
            self.optimizer: Adam = Adam(params=self.model.parameters(), lr=self.learning_rate)
        elif self.train_config["optimizer"] == "rmsprop":
            self.optimizer: RMSprop = RMSprop(params=self.model.parameters(), lr=self.learning_rate)
        elif self.train_config["optimizer"] == "shared_rmsprop":
            self.model.share_memory()
            self.optimizer: SharedRMSprop = SharedRMSprop(params=self.model.parameters(), lr=self.learning_rate)
            self.optimizer.share_memory()

        self.scheduler = None
        if "scheduler" in self.train_config and self.train_config["scheduler"] == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-10,
                                                                        last_epoch=-1)
        elif self.train_config["scheduler"] == "LinearLR":
            self.scheduler = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=40000)

        swin_unetr_encoder_args: dict = vars(self.ssl_head_args)
        self.wandb_sweep_config = {
            'name': 'sweep',
            'method': 'grid',  # grid, random, bayes
            'metric': {'goal': 'minimize', 'name': 'loss'},
            'parameters': {
                "seq_len": {"value": self.train_config["input_depth"]},
                "pretrained_vision_encoder": {"value": self.use_pretrained_vision_encoder},
                "freeze_pretrained_vision_encoder_weights": {
                    "value": self.train_config["freeze_pretrained_vision_encoder_weights"]},
                "init_lr": {"value": self.learning_rate},
                "dropout": {"value": self.train_config["dropout"]},
                "pretrain_mode": {"value": train_config["pretrain_mode"]}
            }
        }
        for key in self.train_config:
            self.wandb_sweep_config["parameters"][key] = {"values": [self.train_config[key]]}
        for key in swin_unetr_encoder_args:
            self.wandb_sweep_config["parameters"][key] = {"values": [swin_unetr_encoder_args[key]]}

        self.config_filename = os.path.join(self.output_dir, "train_config.json")
        save_dict_as_json(self.train_config, None, self.config_filename)

        self.start_time_tmp_artifact = time.time()

    def initialize_training(self):
        countdown(sec=CHECKPOINT_WAIT_TIME,
                  optional_text="Giving time to upload torch model checkpoint to "
                                + self.checkpoint_path + " e.g. using post request",
                  cancel_condition_function=self.checkpoint_exists)
        checkpoint_loaded: bool = self.load_checkpoint()
        self.save_weights()
        logging.info("Global A3C net initialized")

        self.run = wandb.init(project=self.project_name, config=self.wandb_sweep_config, resume=checkpoint_loaded,
                              id=self.run_id)
        wandb.watch(self.model)
        self.model_artifact = wandb.Artifact(self.run.project + '_' + self.session_id, type='model')

        self.start_time_tmp_artifact = time.time()

        try:
            if os.path.isfile(self.config_filename):
                self.model_artifact.add_file(self.config_filename)
            else:
                logging.error("Error: could not save to artifact: " + str(self.config_filename))
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

        self.training_running = True
        if self.train_config["freeze_pretrained_vision_encoder_weights"] and self.train_config["unfreeze_pretrained_vision_encoder_weights"]:
            logging.info("Frozen layers will be unfrozen after " + str(self.train_config["finetuning_warmup_steps"]) + " steps")
        self.handle_try_unfreeze_all_cnn_layers()
        try:
            self.model.print_info()
            logging.info(self.get_model_info())
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
        logging.info("global net initialized")

    def get_number_of_agents(self):
        try:
            return len(self.agent_data)
        except Exception as e:
            logging.error("get_number_of_agents")
            logging.error(e)
            traceback.print_exc()

        return 0

    def calculate_metrics(self):
        try:
            all_metrics = []
            all_keys: dict = {}
            mean: dict = {}
            agent_scores: dict = {}
            for agent_id in self.agent_data:
                if len(self.agent_data[agent_id]) > 0:
                    agent_data = self.agent_data[agent_id]
                    if "metrics" in agent_data:
                        metrics = agent_data["metrics"]
                        if "inv_state" in metrics:
                            for key in metrics["inv_state"]:
                                if "amount" in metrics["inv_state"][key]:
                                    metrics["inv_" + str(key)] = metrics["inv_state"][key]["amount"]
                        all_keys.update(metrics)
                        all_metrics.append(metrics)
                        if "score" in metrics:
                            agent_scores[agent_id] = metrics["score"]
                            self.best_score = max(self.best_score, metrics["score"])

            for metrics in all_metrics:
                for key in all_keys:
                    if isinstance(all_keys[key], float):
                        if key not in metrics:
                            metrics[key] = 0
                        mean[key] = 0.0
                    if isinstance(all_keys[key], int):
                        if key not in metrics:
                            metrics[key] = 0
                        mean[key] = 0

            for key in mean:
                for metrics in all_metrics:
                    if (isinstance(mean[key], float) or isinstance(mean[key], int)) and (
                            isinstance(metrics[key], float) or isinstance(metrics[key], int)):
                        mean[key] += metrics[key]
                if isinstance(mean[key], float) or isinstance(mean[key], int):
                    mean[key] = mean[key] / len(all_metrics)

            self.metrics.update(mean)
            self.metrics["global_agents_total_epochs"] = self.agents_total_epochs
            self.metrics["global_agents_total_steps"] = self.agents_total_steps
            self.metrics["num_agents"] = self.get_number_of_agents()

            if "score" in self.metrics:
                # early stopping watch score
                self.early_stopping(score=self.metrics["score"])

            agent_ids_sorted = list(agent_scores.keys())
            agent_ids_sorted.sort()
            scores = list(agent_scores.values())
            table = wandb.Table(data=[scores], columns=agent_ids_sorted)
            self.metrics_wandb_only["agent_scores"] = wandb.plot.histogram(table, "agent scores",
                                                                           title="Agents scores")
        except Exception as e:
            logging.error("calculate_metrics")
            logging.error(e)
            traceback.print_exc()

    def process_gradients_from_agent(self, file_path: str):
        data_dict: dict = {}
        try:
            data_dict = torch.load(file_path, map_location=device)
        except Exception as e:
            logging.error("process_gradients_from_agent()")
            logging.error(e)
            traceback.print_exc()

        return self.process_gradients_and_metrics(data_dict)

    def process_gradients_and_metrics(self, data_dict):
        response: dict = {}
        try:
            gradients: [] = []
            if "agent_id" in data_dict:
                agent_id = data_dict["agent_id"]
                response["agent_id"] = agent_id
                logging.info("process_data_from_agent" + agent_id)

                if "add_agent_steps" in data_dict:
                    self.agents_total_steps += int(data_dict["add_agent_steps"])
                    print("agent steps now: " + str(self.agents_total_steps))
                    self.handle_try_unfreeze_all_cnn_layers()

                if "gradients" in data_dict:
                    gradients: [] = data_dict.pop("gradients")

                self.agent_data[agent_id] = data_dict

                if len(gradients) > 0:
                    self.handle_update_request(gradients)
        except Exception as e:
            logging.error("process_gradients_and_metrics()")
            logging.error(e)
            traceback.print_exc()

        return response

    def process_end_of_epoch_data_from_agent(self, data_dict: dict):
        try:
            if "add_agent_epochs" in data_dict:
                added_epochs = int(data_dict["add_agent_epochs"])
                self.agents_total_epochs += added_epochs
        except Exception as e:
            logging.error("process_end_of_epoch_data_from_agent()")
            logging.error(e)
            traceback.print_exc()


    def get_model_info(self) -> dict:
        info_dict: dict = {}
        try:
            """
           Values of param.requires_grad are not saved together with the weights. Therefore, an extra query 
           by the agent is necessary, which value param.requires_grad has per layer.
           In this way, thawing of frozen gradients or weights between global net model and agent model is synchronized.
           """
            info_dict["number_cnn_frozen_layers"] = self.model.count_frozen_cnn_layers()

        except Exception as e:
            logging.error(e)

        return info_dict


    def stop(self):
        try:
            logging.info("trainer stop")
            self.stopped = True
            logging.info("trainer stopped")
            time.sleep(5)
            logging.info("saving checkpoint...")
        except Exception as e:
            logging.error("stop()")
            logging.error(e)
            traceback.print_exc()
        self.save_checkpoint()
        try:
            wandb.alert(
                title='Stop A3C Trainer ' + self.run_id,
                text=str("Stopped. global steps:" + str(self.step)),
                level=wandb.AlertLevel.INFO,
                wait_duration=timedelta(minutes=1)
            )
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

    def handle_update_request(self, gradients: []):
        try:
            self.gradients_queue.append(gradients)
        except Exception as e:
            logging.error("handle_update_request")
            logging.error(e)
            traceback.print_exc()

    def try_update(self):
        try:
            self.metrics_wandb_only["gradients_queue_len"] = len(self.gradients_queue)
            while len(self.gradients_queue) > 0:
                try:
                    logging.info("gradients_queue len " + str(len(self.gradients_queue)))
                    self.update_global_net(gradients=self.gradients_queue.popleft())
                except Exception as e:
                    logging.error("try_update() while loop")
                    logging.error(e)
                    traceback.print_exc()
                self.metrics_wandb_only["gradients_queue_len"] = len(self.gradients_queue)
                logging.info("gradients_queue len " + str(len(self.gradients_queue)))
        except Exception as e:
            logging.error("try_update()")
            logging.error(e)
            traceback.print_exc()

    def update_global_net(self, gradients: []):
        if self.update_active:
            return

        self.update_active = True
        if self.stopped or not self.training_running:
            self.update_active = False
            return

        try:
            # terminate (according to A3C paper)
            if self.global_step_counter_T_max > 0:
                if self.agents_total_steps > self.global_step_counter_T_max:
                    logging.info("max global steps reached")
                    self.stop()
                    self.update_active = False
                    try:
                        wandb.alert(
                            title='A3C Trainer ' + self.run_id,
                            text=str(
                                "Max global net steps reached. global steps:" + str(self.step)),
                            level=wandb.AlertLevel.INFO,
                            wait_duration=timedelta(minutes=1)
                        )
                    except Exception as e:
                        logging.error(e)
                        traceback.print_exc()
                    return

            if self.early_stopping.early_stop:  # stcngurs: added early stopping
                print("early stopping")
                self.stop()
                self.update_active = False
                try:
                    wandb.alert(
                        title='A3C Trainer ' + self.run_id,
                        text=str(
                            "Early Stopping. global steps:" + str(self.step)),
                        level=wandb.AlertLevel.INFO,
                        wait_duration=timedelta(minutes=1)
                    )
                except Exception as e:
                    logging.error(e)
                    traceback.print_exc()
                return

            # logging.info("try update global net")
            update_start_time = time.time()

            self.handle_try_unfreeze_all_cnn_layers()

            if gradients is not None:
                logging.info("global net step " + str(self.step))
                self.optimizer.zero_grad()
                self.apply_gradients(gradients)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # according https://arxiv.org/pdf/2106.10270.pdf

                saved_model_gradients = save_gradients(self.model)  # for logging only

                try:
                    max_gradients = []
                    for param in saved_model_gradients:
                        if param is not None:
                            max_gradients.append(torch.max(param))

                    if len(max_gradients) > 0:
                        self.metrics_wandb_only["max_gradient"] = float(max(max_gradients).cpu().detach().numpy())
                        self.metrics_wandb_only["min_gradient"] = float(min(max_gradients).cpu().detach().numpy())
                        self.metrics_wandb_only["mean_gradient"] = float(np.mean(max_gradients))
                    else:
                        try:
                            wandb.alert(
                                title='Global A3c net all gradients are None',
                                text='Global A3c net all gradients are None',
                                level=wandb.AlertLevel.WARN,
                                wait_duration=timedelta(minutes=15)
                            )
                        except Exception as e:
                            logging.error(e)
                        self.metrics_wandb_only["max_gradient"] = 0
                        self.metrics_wandb_only["min_gradient"] = 0
                        self.metrics_wandb_only["mean_gradient"] = 0
                except Exception as e:
                    logging.error(e)
                    traceback.print_exc()

                self.optimizer.step()
                try:
                    if self.step >= 10 and self.scheduler is not None:
                        self.scheduler.step()
                    after_lr = self.optimizer.param_groups[-1]['lr']
                    self.metrics["after_lr"] = float(after_lr)
                    if self.scheduler is not None:
                        self.metrics['after_lr'] = float(self.scheduler.get_lr()[-1])
                except Exception as e:
                    logging.error(e)
                    traceback.print_exc()
                self.save_weights()
                self.calculate_metrics()

                epoch_metrics = {"global_epoch": self.epoch, "weights_file": self.latest_weights_file_path,
                                 "global_step": self.step,
                                 "needed_time_for_update": float(time.time() - update_start_time),
                                 "global_best_mean_score": self.best_mean_score,
                                 "global_best_score": self.best_score
                                 }
                # resources metrics
                try:
                    logging.info(get_resource_metrics())
                    self.metrics.update(get_resource_metrics())
                except Exception as e:
                    logging.error(e)
                    traceback.print_exc()

                self.metrics.update(epoch_metrics)
                wandb_data: dict = self.metrics_wandb_only
                wandb_data.update(self.metrics)
                if self.run is not None:
                    self.run.log(wandb_data)

                self.step += 1

                if self.step >= 0 and self.agents_total_epochs % 100 == 0:
                    self.save_checkpoint()

        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            logging.info("Saving checkpoint because of exception")
            try:
                wandb.alert(
                    title="ERROR " + str(self.run_id),
                    text='ERROR Global A3C update_global_net() ' + str(e),
                    level=wandb.AlertLevel.ERROR,
                    wait_duration=timedelta(minutes=1)
                )
            except Exception as e:
                logging.error(e)
            self.save_checkpoint()

        self.update_active = False

    def handle_try_unfreeze_all_cnn_layers(self):
        try:
            if self.train_config["freeze_pretrained_vision_encoder_weights"] and self.train_config[
                "unfreeze_pretrained_vision_encoder_weights"]:
                if self.agents_total_steps >= (self.train_config["finetuning_warmup_steps"]):
                    success = self.model.try_unfreeze_all_cnn_layers()
                    if success:
                        print("Unfrozen all layers at step " + str(self.agents_total_steps))
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            self.save_checkpoint()

    def save_weights(self):
        prefix = "global_model_weights_dict_epoch_"
        try:
            # delete outdated weights files
            for file_name in os.listdir(self.output_dir):
                if file_name.startswith(prefix) and file_name != self.latest_weights_file_path:
                    os.remove(os.path.join(self.output_dir, file_name))
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

        self.latest_weights_file_path = os.path.join(self.output_dir,
                                                     prefix + str(self.step) + '.pth')

        self.save_model(self.metrics, self.latest_metrics_file_path, self.latest_weights_file_path)

        if (("score" in self.metrics and self.metrics["score"] >= self.best_score) or
                not os.path.isfile(self.best_metrics_file_path)):
            self.best_metrics = self.metrics
            self.save_model(self.best_metrics, self.best_metrics_file_path, self.best_weights_file_path)

    def save_model(self, metrics: dict, metrics_filepath, weights_filepath):
        model_weights: OrderedDict = self.get_weights()
        save_dict_as_json(metrics, None, metrics_filepath)
        torch.save(model_weights, weights_filepath)  # only encoder weights (model.swinViT)
        print("Model saved to dir:", weights_filepath)

    def make_backup_wandb_artifact(self):
        try:
            self.start_time_tmp_artifact = time.time()
            logging.info("save checkpoint artifact")
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            tmp_checkpoint_artifact = wandb.Artifact(
                "checkpoint_" + self.run.project + '_' + self.run_id + '_' + str(self.step),
                type='model')
            if os.path.isfile(self.checkpoint_path):
                tmp_checkpoint_artifact.add_file(self.checkpoint_path)
            if os.path.isfile(self.best_metrics_file_path):
                tmp_checkpoint_artifact.add_file(self.best_metrics_file_path)
            if os.path.isfile(self.best_weights_file_path):
                tmp_checkpoint_artifact.add_file(self.best_weights_file_path)
            if os.path.isfile(self.latest_metrics_file_path):
                tmp_checkpoint_artifact.add_file(self.latest_metrics_file_path)
            if os.path.isfile(self.latest_weights_file_path):
                tmp_checkpoint_artifact.add_file(self.latest_weights_file_path)
            if os.path.isfile(self.config_filename):
                tmp_checkpoint_artifact.add_file(self.config_filename)
            self.run.log_artifact(tmp_checkpoint_artifact)
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

            try:
                wandb.alert(
                    title="ERROR " + str(self.run_id),
                    text='ERROR Global A3C make_backup_wandb_artifact() ' + str(e),
                    level=wandb.AlertLevel.ERROR,
                    wait_duration=timedelta(minutes=1)
                )
            except Exception as e:
                logging.error(e)

    def save_checkpoint(self) -> []:
        try:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            save_obj: dict = {
                'epoch': self.epoch,
                'step': self.step,
                'agents_total_steps': self.agents_total_steps,
                'agents_total_epochs': self.agents_total_epochs,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': self.metrics,
                'train_config': self.train_config,
                'run_id': self.run_id,
                'agent_data': self.agent_data,
                'gradients_queue': self.gradients_queue,
                'best_score': self.best_score,
                'best_mean_score': self.best_mean_score,
                'best_metrics': self.best_metrics,
                'early_stopping': self.early_stopping,
            }

            if self.scheduler is not None:
                save_obj['scheduler_state_dict'] = self.scheduler.state_dict()

            torch.save(save_obj, self.checkpoint_path)
            self.make_backup_wandb_artifact()
            logging.info("Checkpoint saved")
            return [self.checkpoint_path, self.latest_metrics_file_path, self.best_metrics_file_path,
                    self.latest_weights_file_path, self.best_weights_file_path, self.config_filename]

        except Exception as e:
            logging.error(e)
            traceback.print_exc()
            return "error " + str(e)

    def load_checkpoint(self) -> bool:
        try:
            if os.path.isfile(self.checkpoint_path):
                checkpoint = torch.load(self.checkpoint_path, map_location=device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.epoch = checkpoint['epoch']
                self.step = checkpoint['step']
                if 'agents_total_steps' in checkpoint:
                    self.agents_total_steps = checkpoint['agents_total_steps']
                if 'agents_total_epochs' in checkpoint:
                    self.agents_total_epochs = checkpoint['agents_total_epochs']
                if "metrics" in checkpoint:
                    self.metrics.update(checkpoint['metrics'])
                if "train_config" in checkpoint:
                    self.train_config.update(checkpoint['train_config'])
                if "run_id" in checkpoint:
                    self.run_id = checkpoint['run_id']
                if "agent_data" in checkpoint:
                    self.agent_data = checkpoint["agent_data"]
                if "gradients_queue" in checkpoint:
                    try:
                        self.gradients_queue = checkpoint["gradients_queue"]
                    except Exception as e:
                        logging.error(e)
                        traceback.print_exc()
                if 'best_score' in checkpoint:
                    self.best_score = checkpoint['best_score']
                if 'best_mean_score' in checkpoint:
                    self.best_mean_score = checkpoint['best_mean_score']
                if 'best_metrics' in checkpoint:
                    self.best_metrics = checkpoint['best_metrics']
                if 'early_stopping' in checkpoint:
                    self.early_stopping = checkpoint["early_stopping"]

                if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


                self.model.train()
                logging.info("Checkpoint loaded")
                return True
            else:
                logging.info("OK. Checkpoint could not be loaded. No file existing. Stop training manually if this was not expected and try uploading checkpoint again")
                return False
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
        return False

    def add_reconstructed_map(self, data_dict: dict):
        """
        This function adds the map of an agent and then overlays the maps of all agents,
        each of which reconstructs the agent's movement pattern. The result is logged on Weights and Biases
        @param data_dict: 
        @return: 
        """
        try:
            if "agent_id" in data_dict:
                agent_id = data_dict["agent_id"]
                self.agent_reconstructed_maps[agent_id] = data_dict["img"]

                # overlay last image of every agent
                img: np.ndarray = None
                for agent_id in self.agent_reconstructed_maps:
                    if img is None:
                        img = self.agent_reconstructed_maps[agent_id]
                    else:
                        img = np.minimum(img, self.agent_reconstructed_maps[agent_id])

                if img is not None:
                    img: np.ndarray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img)
                    self.metrics_wandb_only["overlaid_maps_of_all_agents"] = wandb.Image(pil_image)
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
