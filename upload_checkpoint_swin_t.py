import os
import time

from dotenv import load_dotenv

from src.trainers.http_upload_checkpoint import http_post_upload_checkpoint_file

if __name__ == '__main__':
    start_time = time.time()
    load_dotenv()
    api_prefix: str = '/api'
    auth_prefix = "Bearer "
    PRETRAIN_NET_ADDRESS = os.getenv("PRETRAIN_NET_ADDRESS")

    path = "tmp/upload_checkpoints/swin-t_ssl_checkpoint.pt"
    response = http_post_upload_checkpoint_file(address=PRETRAIN_NET_ADDRESS, file_path=path)
    print(response)
    print("needed time: " + str(time.time() - start_time))
