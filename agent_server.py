
import logging

import os
import signal
import threading
import time

import warnings

import coloredlogs

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from starlette.middleware.cors import CORSMiddleware

from config.train_config import get_train_config
from src.agent.run_cartpole_agent_epochs import run_cartpole_agent

from src.trainers.base_trainer import BaseTrainer


warnings.simplefilter(action='ignore', category=FutureWarning)
coloredlogs.install(level='INFO')

load_dotenv()
WANDB_KEY = os.getenv("WANDB_KEY")  # get secret from kubernetes cluster
AGENT_PORT = os.getenv("AGENT_PORT")
start_time = time.time()

server_name: str = "agent"

trainer_name: str = None
trainer_instance: BaseTrainer = None

agent_active: bool = True

train_config: dict = get_train_config()

thread = None

"""
server config
"""
app = FastAPI()
# constraints for rest api
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])
# only change the prefixes if you know what you are doing
api_prefix: str = '/api'  # url prefix for http requests
# ws_prefix: str = '/ws'  # url prefix for websocket
uvicorn_port: int = int(AGENT_PORT)  # server port


@app.get("/")
async def get_status():
    logging.info("get_status")
    status_dict: dict = {"server": "uvicorn", "port": uvicorn_port, "name": server_name, "train_config": train_config,
                         "agent_active": agent_active}
    return status_dict


def agent_process():
    global agent_active
    run_cartpole_agent()
    agent_active = False
    print("agent inactive")

def main(server_instance_config: dict = None):
    """
    This is method starts the main thread of the server.
    The main thread is responsible for the rest api. All other processes must be executed as separate threads.
    """
    global server_name

    if server_instance_config is not None:
        if "trainer_name" in server_instance_config:
            trainer_name = server_instance_config["trainer_name"]

    logging.info("running python server " + str(server_name) + " port: " + str(uvicorn_port))
    logging.info("\nSwagger UI is listening: http://127.0.0.1:" + str(uvicorn_port) + "/docs")

    start_agent_process()

    uvicorn.run(app, host="0.0.0.0", port=uvicorn_port)  # main thread loop forever


def start_agent_process():
    # start trainer process
    thread = threading.Thread(target=agent_process)
    thread.daemon = True
    thread.start()


# this entry point is called by docker image using env variables from .env file or from kubernetes secrets
if __name__ == "__main__":
    main()  # calls the above defined main function

