FROM nvidia/cuda:11.6.2-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3 python3-pip



# all below copied files will be inside dir /app
WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# install python requirements inside docker image
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

ARG CACHEBUST="$(date)"

# copy all needed src files, configs and documentation to the docker image (no test files)
COPY ./net_trainer_server.py ./net_trainer_server.py
COPY ./README.md ./README.md
COPY ./.env-example ./.env-example
COPY ./config ./config
COPY ./src/common ./src/common
COPY ./src/dataloader ./src/dataloader
COPY ./src/a3c ./src/a3c
COPY ./src/swin_unetr ./src/swin_unetr
COPY ./src/trainers ./src/trainers
COPY ./pretrainedModels/icon_classifier ./pretrainedModels/icon_classifier
COPY ./pretrainedModels/pretrained_naturecnn_out512_seq1/model ./pretrainedModels/pretrained_naturecnn_out512_seq1/model


ARG CACHEBUST

# publish inner port
EXPOSE 8080

CMD ["python3", "-u", "net_trainer_server.py"]