import logging
import os

import urllib3
from minio import Minio
from dotenv import load_dotenv

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logging.warning("Disabled minio warnings")
load_dotenv()

MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_ADDRESS = os.getenv("MINIO_ADDRESS")
MINIO_PORT = os.getenv("MINIO_PORT")

client = Minio(
    MINIO_ADDRESS + ":" + MINIO_PORT,
    secure=True,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    cert_check=False
)

for bucket in client.list_buckets():
    print(bucket.name, bucket.creation_date)

    paths = []
    count = 0
    for item in client.list_objects(bucket.name, recursive=True):
        path = item.object_name
        paths.append(path)
        count += 1

    for p in paths:
        print(p)
    print(len(paths))
