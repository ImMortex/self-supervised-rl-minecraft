import os
import time

from dotenv import load_dotenv

from src.trainers.http_upload_checkpoint import http_post_upload_checkpoint_file

if __name__ == '__main__':
    start_time = time.time()
    load_dotenv()
    api_prefix: str = '/api'
    auth_prefix = "Bearer "
    PRETRAIN_NET_ADDRESS_SIMCLR = "https://rancher.hs-anhalt.de/k8s/clusters/c-m-wpkf52vh/api/v1/namespaces/stable-gym/services/http:trainer-service-pretrain2:8080/proxy"

    path = "tmp/simclr-pretrain/checkpoint/2/simclr-pretrain_checkpoint.pt"
    response = http_post_upload_checkpoint_file(address=PRETRAIN_NET_ADDRESS_SIMCLR, file_path=path)
    print(response)
    print("needed time: " + str(time.time() - start_time))
