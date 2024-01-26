import logging
import os
from collections import OrderedDict

import coloredlogs
import torch
import torch.optim as optim

from src.trainers.a3c_functions import get_weights, apply_gradients
from src.trainers.swin_unetr_default_args import get_default_args_parser_edited

coloredlogs.install(level='INFO')
device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


class BaseTrainer:
    """
    Abstract super class for trainers
    """

    def __init__(self, train_config: dict):
        self.run_id: str = "None"
        self.train_config = train_config

        ssl_head_args = BaseTrainer.get_ssl_head_args(train_config)

        self.ssl_head_args = ssl_head_args

        self.losses_from_agents: [] = []
        self.gradients_from_agents: [] = []  # Input queue of gradients_each_layer from multiple agents
        self.model = None
        self.optimizer = None
        self.saved_gradients_each_layer: [] = []

        self.best_metrics_file_path: str = None
        self.best_weights_file_path: str = None
        self.latest_metrics_file_path: str = None
        self.latest_weights_file_path: str = None

        self.metrics: dict = {}
        self.best_metrics: dict = {}

        self.training_run_counter: int = 0
        self.training_running: bool = False
        self.stopped: bool = False

        self.agent_data: dict = {}
        self.output_dir = "tmp/trainer/"
        self.output_dir_gradients = self.output_dir
        self.checkpoint_path = "tmp/trainer/checkpoint/checkpoint.pt"
        self.agents_total_epochs = 0

    def get_latest_model_weights_file_path(self):
        return self.latest_weights_file_path

    def add_gradients_to_queue(self, gradients_each_layer: []):
        self.gradients_from_agents.append(gradients_each_layer)
        # apply_weights(self.model, gradients_each_layer)

    def get_next_gradients(self) -> []:
        if len(self.gradients_from_agents) > 0:
            return self.gradients_from_agents.pop(0)
        else:
            return None

    def apply_gradients(self, gradients_each_layer: []):
        apply_gradients(self.model, gradients_each_layer)

    def get_weights(self) -> OrderedDict:
        return get_weights(self.model)

    def stop(self):
        logging.info("trainer stop")
        self.stopped = True

    def resume(self):
        self.stopped = False

    @staticmethod
    def get_ssl_head_args(train_config):
        parser = get_default_args_parser_edited()
        ssl_head_args = parser.parse_args()
        ssl_head_args.in_channels = train_config["in_channels"]
        ssl_head_args.batch_size = train_config["batch_size"]
        ssl_head_args.sw_batch_size = train_config["sw_batch_size"]
        ssl_head_args.amp = False  # changed
        ssl_head_args.epochs = train_config["ssl_pretrain_epochs"]
        torch.backends.cudnn.benchmark = True
        torch.autograd.set_detect_anomaly(True)
        ssl_head_args.distributed = False
        if "WORLD_SIZE" in os.environ:
            ssl_head_args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        ssl_head_args.device = device
        ssl_head_args.world_size = 1
        ssl_head_args.rank = 0
        if ssl_head_args.distributed:
            ssl_head_args.device = "cuda:%d" % ssl_head_args.local_rank
            torch.device.set_device(ssl_head_args.local_rank)
            torch.distributed.init_process_group(backend="nccl", init_method=ssl_head_args.dist_url)
            ssl_head_args.world_size = torch.distributed.get_world_size()
            ssl_head_args.rank = torch.distributed.get_rank()
            print(
                "Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d."
                % (ssl_head_args.rank, ssl_head_args.world_size)
            )
        else:
            print("Training with a single process on 1 GPUs.")
        assert ssl_head_args.rank >= 0
        return ssl_head_args

    @staticmethod
    def get_optimizer(args, model, learning_rate):

        # default
        optimizer = optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=args.decay)

        if args.opt == "adamw":
            optimizer = optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=args.decay)

        elif args.opt == "sgd":
            optimizer = optim.SGD(params=model.parameters(), lr=learning_rate, momentum=args.momentum,
                                  weight_decay=args.decay)
        return optimizer

    def initialize_training(self):
        pass

    def process_gradients_from_agent(self, out_file_path: str):
        pass

    def process_end_of_epoch_data_from_agent(self, data: dict):
        pass

    def update_global_net(self, gradients: []):
        pass

    def save_checkpoint(self) ->[]:
        pass

    def load_checkpoint(self) -> bool:
        pass

    def checkpoint_exists(self) -> bool:
        return os.path.exists(self.checkpoint_path)

    def run_training(self, train_dataset=None, val_dataset=None):
        pass

    def calculate_metrics(self):
        pass

    def try_update(self):
        pass

    def get_model_info(self):
        model_info: dict = {}
        return model_info

    def get_run_id(self):
        return self.run_id
