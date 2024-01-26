import json
import logging
import math
import os
import shutil
import socket
import threading
import time
import traceback
import warnings
import zipfile
from io import BytesIO

import aiofiles
import coloredlogs
import numpy as np
import urllib3
import uvicorn
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile, Body
from fastapi.responses import StreamingResponse, FileResponse
from fastapi_utils.tasks import repeat_every
from sklearn.model_selection import train_test_split
from starlette.middleware.cors import CORSMiddleware
from typing_extensions import Annotated

import wandb
from config.train_config import get_train_config
from src.common.minio_fncts.minio_helpers import get_s3_transition_rel_paths_grouped_by_agent, minio_check_bucket, \
    get_minio_client_secure_no_cert
from src.dataloader.torch_mc_rl_data import S3MinioCustomDataset
from src.trainers.a3c_trainer import A3CTrainer
from src.trainers.base_trainer import BaseTrainer
from src.trainers.multitask_pretrain_trainer import MultiTaskPretrainTrainer
from src.trainers.pretrain_trainer import PretrainTrainer
from src.trainers.simCLR_pretrain_trainer import SimClrPretrainTrainer

warnings.simplefilter(action='ignore', category=FutureWarning)
coloredlogs.install(level='INFO')

load_dotenv()
WANDB_KEY = os.getenv("WANDB_KEY")  # get secret from kubernetes cluster
TRAINER = os.getenv("TRAINER")

TRAINER_PORT = os.getenv("TRAINER_PORT")
start_time = time.time()

bucket_name = "testbucket"
server_name: str = "trainer"

trainer_name: str = None
trainer_instance: BaseTrainer = None

loop_active: bool = True

old_config_file = "./used-config/train_config.json"
if os.path.exists(old_config_file):
    os.remove(old_config_file)
train_config: dict = get_train_config()

minio_bucket_name = train_config["minio_bucket_name"]

thread = None
hostname = socket.gethostname()

def create_datasets_for_pretraining(train_config, test_split=0.2):
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.warning("Disabled minio warnings")
    minio_client = get_minio_client_secure_no_cert()
    bucket_name = minio_bucket_name
    minio_check_bucket(minio_client, bucket_name)
    s3_transition_rel_paths_grouped_by_agent = get_s3_transition_rel_paths_grouped_by_agent(minio_client,
                                                                                            bucket_name)
    logging.info("transitions from " + str(len(s3_transition_rel_paths_grouped_by_agent)) + " agents")

    if test_split > 0:
        train_paths_g_by_agent, val_paths_g_by_agent, _, _ = train_test_split(s3_transition_rel_paths_grouped_by_agent,
                                                                              s3_transition_rel_paths_grouped_by_agent,
                                                                              test_size=0.2,
                                                                              shuffle=True,
                                                                              random_state=29)
    else:
        train_paths_g_by_agent = s3_transition_rel_paths_grouped_by_agent


    if train_config["pretrain_architecture"].lower() == "swin_vit":
        seq_to_3d_image = True
    else:
        seq_to_3d_image = False

    logging.info("Creating train and val datasets ...")
    train_dataset: S3MinioCustomDataset = S3MinioCustomDataset(train_paths_g_by_agent,
                                                               x_depth=train_config["input_depth"],
                                                               width_2d=train_config["img_size"][1],
                                                               height_2d=train_config["img_size"][0],
                                                               minio_client=minio_client, bucket_name=bucket_name,
                                                               seq_to_3d_image=seq_to_3d_image)

    if test_split > 0:
        val_dataset: S3MinioCustomDataset = S3MinioCustomDataset(val_paths_g_by_agent,
                                                                 x_depth=train_config["input_depth"],
                                                                 width_2d=train_config["img_size"][1],
                                                                 height_2d=train_config["img_size"][0],
                                                                 minio_client=minio_client, bucket_name=bucket_name,
                                                                 seq_to_3d_image=seq_to_3d_image)
    else:
        val_dataset = None
    return train_dataset, val_dataset


def initialize_trainer(train_dataset, val_dataset):
    global server_name
    global trainer_name
    global trainer_instance
    global train_config

    logging.info("login to wandb... ")
    if WANDB_KEY is not None:
        try:
            wandb.login(key=str(WANDB_KEY))
            logging.info("login to wandb done")
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

    try:
        if trainer_name is None:
            trainer_name = TRAINER

        logging.info("trainer_name: " + str(trainer_name))
        if trainer_name is not None:
            if str(trainer_name).lower() == "pretrain":
                trainer_instance = PretrainTrainer(train_config)
                trainer_instance.run_training(train_dataset, val_dataset)
            elif str(trainer_name).lower() == "simclr_pretrain":
                trainer_instance = SimClrPretrainTrainer(train_config)
                trainer_instance.run_training(train_dataset, val_dataset)
            elif str(trainer_name).lower() == "multitask_pretrain":
                trainer_instance = MultiTaskPretrainTrainer(train_config)
                trainer_instance.run_training(train_dataset, val_dataset)
            elif str(trainer_name).lower().startswith("a3c"):
                # train global A3C net pretrained or not pretrained (specified by train_config)
                trainer_instance = A3CTrainer(train_config)
                trainer_instance.initialize_training()
    except Exception as e:
        logging.error(e)
        traceback.print_exc()


"""
server config
"""
app = FastAPI()
# constraints for rest api
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])
# only change the prefixes if you know what you are doing
api_prefix: str = '/api'  # url prefix for http requests
# ws_prefix: str = '/ws'  # url prefix for websocket
uvicorn_port: int = int(TRAINER_PORT)  # server port


def files_as_zip(file_list) -> StreamingResponse:
    io = BytesIO()
    zip_file_name = "zip"
    zip_filename = "%s.zip" % zip_file_name
    with zipfile.ZipFile(io, mode='w', compression=zipfile.ZIP_DEFLATED) as zip:
        for fpath in file_list:
            zip.write(fpath)
        # close zip
        zip.close()
    return StreamingResponse(
        iter([io.getvalue()]),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": "attachment;filename=" + zip_filename}
    )


@app.get("/")
async def get_status():
    logging.info("get_status")
    status_dict: dict = {"hostname": hostname, "server": "uvicorn", "port": uvicorn_port, "name": server_name, "train_config": get_train_config()}
    try:
        trainer_initialized: bool = False
        if trainer_instance is not None:
            trainer_initialized = True

        status_dict.update({"trainer_initialized": trainer_initialized, "trainer_name": trainer_name})

    except Exception as e:
        logging.error(e)
        traceback.print_exc()
    return status_dict


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


@app.get(api_prefix + "/getMetrics")
async def get_metrics():
    logging.info("/getMetrics")
    status_dict: dict = {"server": "uvicorn", "port": uvicorn_port, "name": server_name}
    try:
        trainer_initialized: bool = False
        if trainer_instance is not None:
            trainer_initialized = True

        status_dict.update({"trainer_initialized": trainer_initialized, "trainer_name": trainer_name})

        if trainer_instance is not None and trainer_instance.metrics is not None:
            trainer_instance.calculate_metrics()
            status_dict["metrics"] = trainer_instance.metrics
            status_dict["stopped"] = trainer_instance.stopped
            # print(status_dict)
            for key in status_dict["metrics"]:
                try:
                    # on error case
                    if not is_jsonable(status_dict["metrics"][key]):
                        status_dict["metrics"][key] = 0
                    if str(status_dict["metrics"][key]).lower() == "nan":
                        status_dict["metrics"][key] = 0
                    if not isinstance(status_dict["metrics"][key], str):
                        if math.isnan(status_dict["metrics"][key]):
                            status_dict["metrics"][key] = 0
                except Exception as e:
                    logging.error(e)

            # logging.info(status_dict)
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
    return status_dict


@app.get(api_prefix + "/getGlobalAgentsTotalEpochs")
async def get_global_agents_total_epochs():
    logging.info("getGlobalAgentsTotalEpochs")
    response_dict: dict = {}
    try:

        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")

        trainer_initialized: bool = False
        if trainer_instance is not None:
            trainer_initialized = True

        response_dict.update({"trainer_initialized": trainer_initialized, "trainer_name": trainer_name})

        if trainer_instance is not None and trainer_instance.metrics is not None:
            agents_total_epochs = trainer_instance.agents_total_epochs
            response_dict["agents_total_epoch"] = agents_total_epochs

    except Exception as e:
        logging.error(e)
        traceback.print_exc()
    return response_dict


@app.get(api_prefix + "/getAgentMetrics/{agent_id}")
async def get_agent_metrics(agent_id: str = None):
    logging.info("get_agent_metrics")
    response_dict: dict = {}
    try:

        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")

        if trainer_instance is not None and trainer_instance.metrics is not None:
            response_dict = trainer_instance.agent_data[agent_id]
            for key in response_dict:
                if not is_jsonable(response_dict[key]):
                    response_dict[key] = 0
                if str(response_dict[key]).lower() == "nan" or math.isnan(response_dict[key]):
                    response_dict[key] = 0
            # logging.info(response_dict)
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
    return response_dict


@app.get(api_prefix + "/getWeightsFile")
async def get_weights_file() -> FileResponse:
    logging.info(api_prefix + "/getWeightsFile")
    try:

        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")

        if trainer_instance.training_running:
            latest_model_weights_file_path: str = trainer_instance.get_latest_model_weights_file_path()
            return FileResponse(latest_model_weights_file_path)
        else:
            raise HTTPException(status_code=500, detail="Trainer not ready yet")

    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        raise HTTPException(status_code=404, detail="File not found")


@app.get(api_prefix + "/getTrainingConfig")
async def get_training_config():
    logging.info(api_prefix + "/getTrainingConfig")
    try:
        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")

        if trainer_instance.training_running:
            train_config["global_net_run_id"] = trainer_instance.get_run_id()
            return train_config
        else:
            raise HTTPException(status_code=500, detail="Trainer not ready yet")
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get(api_prefix + "/getModelInfo")
async def get_training_config():
    logging.info(api_prefix + "/getModelInfo")
    try:
        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")

        if trainer_instance.training_running:
            return trainer_instance.get_model_info()
        else:
            raise HTTPException(status_code=500, detail="Trainer not ready yet")
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")





@app.post(api_prefix + "/postGradientsFile")
async def post_gradients(file: UploadFile = File(...)):
    logging.info(api_prefix + "/postGradientsFile")
    try:
        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")
        if not trainer_instance.training_running:
            raise HTTPException(status_code=500, detail="Trainer is not ready yet")

        if trainer_instance.training_running:
            try:
                file_name = os.path.basename(file.filename)
                out_file_path = os.path.join(trainer_instance.output_dir_gradients, file_name)
                logging.info(out_file_path)
                time_start_write = time.time()
                async with aiofiles.open(out_file_path, 'wb+') as out_file:
                    while content := await file.read():  # async read chunk
                        await out_file.write(content)  # async write chunk
                logging.info("Wrote gradients file in " + str(time.time() - time_start_write))
            except Exception as e:
                logging.error(e)
                traceback.print_exc()
                raise HTTPException(status_code=500, detail="Error while writing file " + out_file_path)

            return trainer_instance.process_gradients_from_agent(out_file_path)
        else:
            raise HTTPException(status_code=500, detail="Trainer not ready yet")

    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(api_prefix + "/postEndOfEpochData")
async def post_end_of_epoch_data(data_dict=Body(...)):
    logging.info(api_prefix + "/postEndOfEpochData")
    logging.info(data_dict)
    try:
        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")

        if trainer_instance.training_running:
            return trainer_instance.process_end_of_epoch_data_from_agent(data_dict)
        else:
            raise HTTPException(status_code=500, detail="Trainer not ready yet")

    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(api_prefix + "/postStop")
async def post_stop():
    logging.info(api_prefix + "/postStop")
    try:
        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")

        trainer_instance.stop()
        trainer_instance.save_checkpoint()
        return {"stopped": trainer_instance.stopped}
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(api_prefix + "/runTraining")
async def run_training():
    logging.info(api_prefix + "/runTraining")
    try:
        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")

        trainer_instance.run_training()
        return {"stopped": trainer_instance.stopped}
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(api_prefix + "/saveCheckpoint")
async def post_save_checkpoint() -> StreamingResponse:
    logging.info(api_prefix + "/saveCheckpoint")
    try:
        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")

        zip_file = None
        try:
            paths = trainer_instance.save_checkpoint()
            zip_file = files_as_zip(paths)
        except Exception as e:
            logging.error(e)
            traceback.print_exc()

        return zip_file

    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(api_prefix + "/uploadCheckpointFile")
async def post_upload_checkpoint_file(file: UploadFile = File(...)):
    logging.info(api_prefix + "/uploadCheckpointFile")
    response: dict = {"success": False}
    try:
        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")
        try:
            tmp_upload_file_name = trainer_instance.checkpoint_path + "tmp"
            out_file_path = tmp_upload_file_name
            async with aiofiles.open(out_file_path, 'wb+') as out_file:
                while content := await file.read():  # async read chunk
                    await out_file.write(content)  # async write chunk

            shutil.copyfile(out_file_path, trainer_instance.checkpoint_path)

            logging.info("uploaded checkpoint")
            response: dict = {"path": out_file_path, "success": True}
            logging.info(response)
            return response
        except Exception as e:
            logging.error(e)
            traceback.print_exc()
        return response

    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(api_prefix + "/loadCheckpoint")
async def post_load_checkpoint() -> dict:
    logging.info(api_prefix + "/loadCheckpoint")
    response: dict = {}
    try:
        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")

        if trainer_instance.training_running:
            path = trainer_instance.load_checkpoint()
            response["load_checkpoint_response"] = path
            return response
        else:
            raise HTTPException(status_code=500, detail="Trainer not ready yet")
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post(api_prefix + "/postReconstructedMap")
async def post_reconstructed_map(
        file: Annotated[UploadFile, File()]):
    logging.info(api_prefix + "/postReconstructedMap")
    response: dict = {"success": True}
    try:
        if trainer_instance is None:
            raise HTTPException(status_code=500, detail="Trainer not initialized yet")
        if trainer_instance.stopped:
            raise HTTPException(status_code=423, detail="LOCKED. Trainer stopped training")

        im = Image.open(file.file)
        if im.mode in ("RGBA", "P"):
            im = im.convert("RGB")
        agent_id = os.path.basename(file.filename.split(".")[0].split("_")[1])
        data = {}
        data["agent_id"] = agent_id
        data["img"] = np.array(im)
        trainer_instance.add_reconstructed_map(data)
        return response
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()
        im.close()


@app.on_event("startup")
@repeat_every(seconds=0.5)
def try_update() -> None:
    try:
        if trainer_instance is not None and trainer_instance.training_running:
            trainer_instance.try_update()
    except Exception as e:
        logging.error(e)
        traceback.print_exc()


def trainer_process(train_dataset, val_dataset):
    try:
        time.sleep(1)
        initialize_trainer(train_dataset, val_dataset)
        while True:
            time.sleep(1)
    except Exception as e:
        logging.error(e)
        traceback.print_exc()
    logging.warning("trainer thread process stopped")


def main(server_instance_config: dict = None):
    """
    This is method starts the main thread of the server.
    The main thread is responsible for the rest api. All other processes must be executed as separate threads.
    """
    global server_name
    global trainer_name
    global trainer_instance
    global uvicorn_port

    if server_instance_config is not None:
        if "trainer_name" in server_instance_config:
            trainer_name = server_instance_config["trainer_name"]
        if "port" in server_instance_config:
            uvicorn_port = server_instance_config["port"]

    logging.info("running python server " + str(server_name) + " port: " + str(uvicorn_port))
    logging.info("\nSwagger UI is listening: http://127.0.0.1:" + str(uvicorn_port) + "/docs")

    train_config = get_train_config()

    if trainer_name is None:
        trainer_name = TRAINER

    pretrain_train_dataset = None
    pretrain_val_dataset = None
    if str(trainer_name).lower() == "pretrain" or str(trainer_name).lower() == "swin-unetr" or str(
            trainer_name).lower() == "simclr_pretrain" or str(trainer_name).lower() == "multitask_pretrain":
        pretrain_train_dataset, pretrain_val_dataset = create_datasets_for_pretraining(train_config)

    start_trainer_process(pretrain_train_dataset, pretrain_val_dataset)

    uvicorn.run(app, host="0.0.0.0", port=uvicorn_port)  # main thread loop forever


def start_trainer_process(train_dataset, val_dataset):
    # start trainer process
    thread = threading.Thread(target=trainer_process, args=(train_dataset, val_dataset,))
    thread.daemon = True
    thread.start()


# this entry point is called by docker image using env variables from .env file or from kubernetes secrets
if __name__ == "__main__":
    main()  # calls the above defined main function
