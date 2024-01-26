import json
import logging
import os
import time
import traceback

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()
api_prefix: str = '/api'
auth_prefix = "Bearer "
HTTP_BEARER_TOKEN = os.getenv("HTTP_BEARER_TOKEN")
redirects = True
verify = False


def target_server(address) -> str:
    server_address = str(address).replace("//localhost", "127.0.0.1")
    return server_address


def http_get_global_agents_total_epochs(address) -> dict:
    url: str = address + api_prefix + "/getGlobalAgentsTotalEpochs"
    response_data: dict = {}
    headers = {
        "cache-control": "no-cache",
        "Authorization": auth_prefix + HTTP_BEARER_TOKEN
    }
    print(url)
    status_code = 0
    retry: int = 0
    while True:
        error = False
        try:
            response = httpx.get(url, timeout=30, headers=headers, follow_redirects=redirects, verify=verify)
            status_code = response.status_code
            response_data: dict = json.loads(response.text)
        except Exception as e:
            print(url + " failed")
            error = True
            traceback.print_exc()

        if status_code == 423:
            return {"trainer_stopped": True}

        if status_code == 200 and not error:
            break
        retry += 1
        time.sleep(1)
        logging.error("status_code: " + str(status_code) + " retry: " + str(retry))

    return response_data

def http_get_training_config(address) -> dict:
    logging.info("downloading training config ...")
    url: str = address + api_prefix + "/getTrainingConfig"
    response_data: dict = {}
    headers = {
        "cache-control": "no-cache",
        "Authorization": auth_prefix + HTTP_BEARER_TOKEN
    }
    print(url)
    status_code = 0
    retry: int = 0
    while True:
        error = False
        try:
            response = httpx.get(url, timeout=30, headers=headers, follow_redirects=redirects, verify=verify)
            status_code = response.status_code
            response_data: dict = json.loads(response.text)
        except Exception as e:
            print(url + " failed")
            error = True
            if retry == 0:
                traceback.print_exc()

        if status_code == 423:
            return {"trainer_stopped": True}

        if status_code == 200 and not error:
            break
        retry += 1
        time.sleep(1)
        logging.error("status_code: " + str(status_code) + " retry: " + str(retry))
    return response_data

def http_get_model_info(address) -> dict:
    logging.info("downloading model info ...")
    url: str = address + api_prefix + "/getModelInfo"
    response_data: dict = {}
    headers = {
        "cache-control": "no-cache",
        "Authorization": auth_prefix + HTTP_BEARER_TOKEN
    }
    print(url)
    status_code = 0
    retry: int = 0
    while True:
        error = False
        try:
            response = httpx.get(url, timeout=30, headers=headers, follow_redirects=redirects, verify=verify)
            status_code = response.status_code
            response_data: dict = json.loads(response.text)
        except Exception as e:
            print(url + " failed")
            error = True
            if retry == 0:
                traceback.print_exc()

        if status_code == 423:
            return {"trainer_stopped": True}

        if status_code == 200 and not error:
            break
        retry += 1
        time.sleep(1)
        logging.error("status_code: " + str(status_code) + " retry: " + str(retry))
    return response_data





def http_get_weights_file(address, download_file_path) -> dict:
    logging.info("downloading weights ...")
    url: str = address + api_prefix + "/getWeightsFile"
    print(url)
    headers = {
        "cache-control": "no-cache",
        "Authorization": auth_prefix + HTTP_BEARER_TOKEN
    }
    response_data: dict = {}
    status_code = 0
    retry: int = 0
    while True:
        error = False
        try:
            response = httpx.get(url, headers=headers, timeout=60, follow_redirects=redirects, verify=verify)
            status_code = response.status_code
            if status_code == 200:
                logging.info("saving downloaded weights ...")
                file_content = response.content
                with open(download_file_path, "wb") as f:
                    f.write(file_content)
                    f.close()
                response_data["status_code"] = status_code

        except Exception as e:
            print(url + " failed")
            error = True
            if retry == 0:
                traceback.print_exc()

        if status_code == 423:
            return {"trainer_stopped": True}

        if status_code == 200 and not error:
            break
        retry += 1
        time.sleep(1)
        logging.error("status_code: " + str(status_code) + " retry: " + str(retry))
    return response_data


def http_post_gradients_file(address, file_path: str) -> dict:
    files = None
    try:
        with open(file_path, "rb") as file:
            file_data = file.read()

        # Prepare the data for the POST request
        files = {"file": (file_path, file_data)}

    except FileNotFoundError:
        print("http_post_gradients_file file not found: " + str(file_path))

    url: str = address + api_prefix + "/postGradientsFile"
    print(url)
    headers = {
        "cache-control": "no-cache",
        "Authorization": auth_prefix + HTTP_BEARER_TOKEN
    }
    response_data: dict = {}
    status_code = 0
    retry: int = 0
    while True:
        error = False
        try:
            response = httpx.post(url, files=files, headers=headers, timeout=60, follow_redirects=redirects,
                                  verify=verify)
            status_code = response.status_code
            response_data: dict = json.loads(response.text)

        except Exception as e:
            print(url + " failed")
            error = True
            if retry == 0:
                traceback.print_exc()

        if status_code == 423:
            return {"trainer_stopped": True}

        if status_code == 200 and not error:
            break
        retry += 1
        time.sleep(1)
        logging.error("status_code: " + str(status_code) + " retry: " + str(retry))
    return response_data

def http_post_reconstructed_map(address, img: np.ndarray, agent_id: str) -> dict:
    file_path = "tmp/reconstructedMap_" + str(agent_id) + ".png"
    form_data = None
    try:
        cv2.imwrite(file_path, img)
        with open(file_path, "rb") as file:
            file_data = file.read()

        # Prepare the data for the POST request
        form_data = {"file": (file_path, file_data)}

    except FileNotFoundError:
        print("http_post_gradients_file file not found: " + str(file_path))

    url: str = address + api_prefix + "/postReconstructedMap"
    print(url)
    headers = {
        "cache-control": "no-cache",
        "Authorization": auth_prefix + HTTP_BEARER_TOKEN
    }

    response = httpx.post(url, files=form_data, headers=headers, timeout=60, follow_redirects=redirects,
                          verify=verify)
    status_code = response.status_code
    if status_code == 423:
        return {"trainer_stopped": True}

    response_data: dict = json.loads(response.text)
    return response_data

def http_post_end_of_epoch_data(address, data_dict: dict) -> dict:
    url: str = address + api_prefix + "/postEndOfEpochData"
    print(url)
    headers = {
        "cache-control": "no-cache",
        "Content-Type": "application/json",
        "Authorization": auth_prefix + HTTP_BEARER_TOKEN
    }
    response_data: dict = {}
    status_code = 0
    retry: int = 0
    while True:
        error = False
        try:
            response = httpx.post(url, json=data_dict, headers=headers, timeout=60, follow_redirects=redirects,
                                  verify=verify)
            status_code = response.status_code
            response_data: dict = json.loads(response.text)

        except Exception as e:
            print(url + " failed")
            error = True
            if retry == 0:
                traceback.print_exc()

        if status_code == 423:
            return {"trainer_stopped": True}

        if status_code == 200 and not error:
            break
        retry += 1
        time.sleep(1)
        logging.error("status_code: " + str(status_code) + " retry: " + str(retry))
    return response_data