import json
import logging
import os
import time
import traceback

import coloredlogs
import cv2
import numpy as np
import urllib3
from PIL import Image
from dotenv import load_dotenv

from src.common.helpers.helpers import save_dict_as_json
from src.common.minio_fncts.minio_api_calls import download_json, download_image
from src.common.minio_fncts.minio_helpers import get_minio_client_secure_no_cert, \
    get_s3_transition_rel_paths_grouped_by_agent, minio_check_bucket

from src.common.persisted_memory import PersistedMemory
from src.common.transition import Transition

coloredlogs.install(level='INFO')
load_dotenv()

MINIO_BUCKET_NAME = os.getenv("MINIO_BUCKET_NAME")

if __name__ == "__main__":
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    logging.warning("Disabled minio warnings")

    minio_client = get_minio_client_secure_no_cert()
    bucket_name = MINIO_BUCKET_NAME
    minio_check_bucket(minio_client, bucket_name)
    s3_transition_rel_paths_grouped_by_agent = get_s3_transition_rel_paths_grouped_by_agent(minio_client, bucket_name)

    for agent_paths in s3_transition_rel_paths_grouped_by_agent:

        for path in agent_paths:
            transition_file, view_file = PersistedMemory.get_transition_file_paths(path)
            transition: Transition = None
            image: Image = None
            img: np.ndarray = None
            needed_retries: int = 0

            # retry if connection to server is lost
            while True:
                error = False
                try:
                    downlaod_dir = "./tmp/downloads/"
                    destination_view_path = downlaod_dir  + view_file
                    destination_transition_path = downlaod_dir  + transition_file

                    if os.path.isfile(destination_view_path) and os.path.isfile(destination_transition_path):
                        break
                    start_download_time = time.time()

                    img: np.ndarray = download_image(minio_client, bucket_name, view_file)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image: Image = Image.fromarray(img)

                    transition_dict: dict = json.loads(
                        download_json(minio_client, bucket_name, transition_file))
                    transition: Transition = PersistedMemory.deserialize_transition(transition_dict=transition_dict, image=None)
                    logging.info("Transition downloaded in " + str(time.time() - start_download_time))

                    dir_name = os.path.dirname(destination_view_path)
                    if not os.path.exists(dir_name):
                        os.makedirs(dir_name)
                        logging.info(dir_name)
                    image.save(destination_view_path)
                    save_dict_as_json(transition.__dict__, None, destination_transition_path)
                except Exception as e:
                    logging.error("Transition download failed")
                    error = True
                    traceback.print_exc()

                if transition is not None and image is not None and not error:
                    break  # everything went good
                needed_retries += 1
                time.sleep(5)
                logging.error("Transition download error. Retries: " + str(needed_retries))



    logging.info("Download finished")
