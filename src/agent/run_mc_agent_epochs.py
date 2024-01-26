import logging
import os
import socket
import traceback
from datetime import datetime

import coloredlogs
import wandb
from dotenv import load_dotenv

from config.train_config import get_train_config
from src.agent.agent import McRlAgent
from src.common.persisted_memory import PersistedMemory

coloredlogs.install(level='INFO')
load_dotenv()

WANDB_KEY = os.getenv("WANDB_KEY")
use_global_a3c_model_weights_path = "tmp/agent_net/eval/global_net_weights.pth"  # only for mode "eval_a3c"


def run_mc_agent():
    coloredlogs.install(level='INFO')
    train_config: dict = get_train_config()  # initial config. Will be overwritten by global net config after first update
    dry_run = train_config["dry_run"]  # no action execution and using initial A3C actor net weights for debugging
    logging.info("login to wandb... ")
    if WANDB_KEY is not None:
        try:
            wandb.login(key=str(WANDB_KEY))
            logging.info("login to wandb done")
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

    hostname = socket.gethostname()
    agent_id = str(hostname)
    session_id: str = PersistedMemory.get_session_id_today()
    local_filesystem_store_root_dir: str = train_config["agent_persisted_memory_out_dir"] + train_config[
        "mode"] + "McRlAgent/"
    epoch_id: str = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
    agent: McRlAgent = McRlAgent(dry_run=dry_run, agent_id=agent_id, generation_id=epoch_id,
                                 session_id=session_id,
                                 local_filesystem_store_root_dir=local_filesystem_store_root_dir,
                                 mode=train_config["mode"],
                                 t_per_second=train_config["t_per_second"],
                                 wandb_project_name="mc_agent",
                                 world_seed=train_config["world_seed"])

    epoch: int = 0
    force_stop: bool = False
    while not force_stop:
        try:
            epoch_id: str = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
            logging.info("next epoch: " + epoch_id)
            if train_config["mode"] == "eval_a3c":
                agent.weights_file_path = use_global_a3c_model_weights_path
            force_stop = agent.run_agent_epoch(epoch_id)
            if force_stop:
                break
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
        epoch += 1
    logging.info("\nForce stop")
