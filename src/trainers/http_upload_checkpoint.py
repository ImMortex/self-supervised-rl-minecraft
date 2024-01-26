import json
import logging
import os
import time
import traceback

import httpx
from dotenv import load_dotenv

load_dotenv()
api_prefix: str = '/api'
auth_prefix = "Bearer "
HTTP_BEARER_TOKEN = os.getenv("HTTP_BEARER_TOKEN")
redirects = True
verify = False



def http_post_upload_checkpoint_file(address, file_path: str) -> dict:
    files = None
    try:
        with open(file_path, "rb") as file:
            file_data = file.read()

        # Prepare the data for the POST request
        files = {"file": (file_path, file_data)}

    except FileNotFoundError:
        print("http_post_gradients_file file not found: " + str(file_path))

    url: str = address + api_prefix + "/uploadCheckpointFile"
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
            response = httpx.post(url, files=files, headers=headers, timeout=3600, follow_redirects=redirects,
                                  verify=verify)
            status_code = response.status_code
            response_data: dict = json.loads(response.text)

        except Exception as e:
            print(url + " failed")
            error = True
            if retry == 0:
                traceback.print_exc()
        if status_code == 200 and not error:
            break
        retry += 1
        time.sleep(5)
        logging.error("status_code: " + str(status_code) + " retry: " + str(retry))
    return response_data
