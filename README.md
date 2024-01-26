# self-supervised-rl-minecraft

- Masterthesis of Christian Gurski

## Experiment Results
The results of the experiments are described in detail in the master's thesis. 
The ``results`` folder contains GIFs of the first and last epochs of the Minecraft experiment.

## Initial Project Setup to be able to recreate the Experiment
- see chapter ``Setup python env that supports cuda``
- see chapter ``Setup device``
- see chapter ``Setup wandb, global A3C net trainer server and minio server``
- see chapter ``Setup env variables (train config)``
- set settings in Minecraft (see chapter ``Minecraft settings``)
- see chapter ``Setup config files``
- see chapter ``Setup minecraft icons``
- run unit tests (see chapter ``Unit Tests and Integration Tests``)
- build docker image ``Docker image``

## Run
- Agent and global net are started with the same config. During initialization, the agent will request the config of the global net via rest api and use it.
- If done ``Initial project setup`` you can run the agent using:
````shell
python agent_train_a3c_async.py
````
- The global actor critic can be run locally using:
````shell
python run_net_trainer_server_a3c.py
````
- Alternatively, the global net can be started on another computer using the same env variables as for the agent with TRAINER=a3c.
- The Dockerfile in the project root can be used for this purpose.

- Set env variable MINIO_UPLOAD_FROM inside your .env file
- Upload persisted transitions to minio for pretraining:
````shell
python minio_upload_sessions_to_s3_storage.py
````

- The simCLR pretraining can be run locally using:
````shell
python run_net_trainer_server_simCLR_pretrain.py
````
- Alternatively, the pretraining can be started on another computer using the same env variables described in ``.env-example-pretraining-simCLR``  with TRAINER=simclr_pretrain.
- The Dockerfile in the project root can be used for this purpose.
- Note that the hyperparameters and architecture of the vision encoder in the pretraining must later be adjusted to match 
the hyperparameters and architecture of the vision encoder for actor critic RL.

- after training save a pretrained model as .pt file e.g. from wandb artifacts
- copy the pretrained model to dir ``pretrainedModels``
- set the path of pretrained model in env variables


## Setup python env that supports cuda
- https://pytorch.org/get-started/locally/
- then install requirements using pip
- Example: Read README_setup_pc_pool.md

## Setup config files
- read and follow ``agentObservationTestInput/README.md``

````shell
python run_agent_observation_test.py
````

## Setup Device:
- Set entire screen to Resolution: 16:9 Screen 1920x1080@60
- smaller screens than 1920x1080 are not supported

## Setup .env variables (train config):
- There is a config that ensures the reproducibility of experiments and serves to control the training, e.g. Hyper parameters: ``train_config.py`` 
- A copy of the generated config is stored as a JSON file in the used-config folder and this is used. 
- To change the config or to generate a new config, the JSON file in used-config must be deleted manually
- Create your ``.env`` file in the project root, to set parameters of ``train_config.py`` 
- ``.env-example`` shows all needed variables as example
- You need to generate your own secrets for wandb, global A3C net trainer server and minio server
- The HTTP connection is needed for global A3C net trainer server and optional pretraining server to upload
checkpoints e.g. ``using upload_checkpoint_a3c.py``
- HTTPS using BEARER_TOKEN

## Setup wandb, global A3C net trainer server and minio server
- You need a wandb account and the access key for it
- Run global A3C net trainer server as Kubernetes pod using ``trainer-pod-a3c.yml`` or ``trainer-pod-a3c-pretrained.yml``
- Setup a minio server with or without replicas

## Setup Minecraft settings:

- use Minecraft (Java) version 1.19.1
- Set Standard settings except:
- Language: English (UK)
- Resolution: 16:9 Screen 1920x1080@60
- Fullscreen: Off, Maximize Window
- Render and Animation Distance: 6 chunks
- GUI Scale: Auto
- Controls -> Auto Jump: Off
- Controls -> Mouse Settings -> Raw Input Off
- FOV: normal (70)


## Docker Image
- there is a Dockerfile defined in the project root
- the same docker Image is used to train global actor critic net and for pretraining
- the desired training is configured using the env variables
- Minecraft RL agents cannot run using docker
- TRAINER=a3c to start a global actor critic net 
- TRAINER=simclr_pretrain to start a pretraining using simCLR

## Unit Tests and Integration Tests

- Directory: tests
- run all tests:

````shell
#run shell command in project root
# This way all test functions (with prefix "test") are found and executed from project root.
python -m unittest discover ./tests -v
````

## (optional) Setup minecraft icons
- You need to do only this step if dir ``./tmp/minecraft_icons`` is empty or you are using a newer version of Minecraft

- The minecraft icons are provided once at the beginning of the project:

### Step 1 Download:

- Download minecraft icons from https://mc.nerothe.com/ for your Minecraft Version and save them to ``./tmp/minecraft_icons`` dir

### Step 2 Rename and copy to directory:

- The images must be named like the items in the minecraft-data lib are called.
- The files will have the pattern ``id_name.png`` with underscores instead of spaces.
- To do so run:

````shell
python generate_minecraft_item_icons.py
````

- After that you have to train the icon classifier again:
````shell
python train_simple_icon_classifier.py
````

- You find the trained model files in the ``./tmp/icon_classifier_trainer_output/`` dir in a subdir with the timestamp
- Copy the subdir to the dir ``./pretrainedModels/icon_classifier``. You can delete the old one
- Change the ``PRETRAINED_ICON_CLASSIFIER`` variable in your ``.env`` file