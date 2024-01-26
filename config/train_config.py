import logging
import os

from dotenv import load_dotenv

from src.common.helpers.helpers import save_dict_as_json, load_from_json_file

load_dotenv()

def str2bool(v):
  return str(v).lower() in ("yes", "true", "t", "1")

def set_train_config():
    config_path = "./used-config/train_config.json"

    if os.path.isfile(config_path):
        return
    # Train config is equal on agent and global net side or optional for pretraining trainer. Ensures reproduction of experiments
    train_config: dict = {
        # load checkpoints before or during training
        "global_net_address": os.getenv("GLOBAL_A3C_NET_ADDRESS"),  # agent must know the global net address for rest api
        "checkpoint_wait_time": int(os.getenv("CHECKPOINT_WAIT_TIME") or -1),  # wait time in seconds to upload checkpoints using e.g. see upload_checkpoint_a3c.py

        # self supervised pretraining 3D or 2D
        "in_channels": 3, # RGB, Hardcoded
        "pretrain_batch_size": int(os.getenv("PRETRAIN_BATCH_SIZE") or -1),  # batch size for pretraining
        "pretrain_lr": float(os.getenv("PRETRAIN_LEARNING_RATE") or -1),  # initial learning rate for pretraining
        "sw_batch_size": 1,  # Hardcoded, swin VIT number of sliding window batch size for dataset of SwinUNETR by monai
        "input_depth": int(os.getenv("INPUT_DEPTH") or -1),  # depth of 3D image is equal to len state sequence. Default: 1 for simple state
        "img_size": [int(os.getenv("IMG_SIZE_H") or -1), int(os.getenv("IMG_SIZE_W") or -1)],
        "ssl_pretrain_epochs": int(os.getenv("PRETRAIN_EPOCHS") or -1),  # max epochs for pretraining SSL Head by Nvidia
        "ssl_pretrain_steps": int(os.getenv("PRETRAIN_STEPS") or -1),  # max steps for simCLR or multitask pretraining
        "ram_gb_limit": int(os.getenv("RAM_GB_LIMIT") or -1),  # ram limit needed because caching is used
        "vision_encoder_out_dim": int(os.getenv("VISION_ENCODER_OUT_DIM") or -1),  # out dimension of resnet using simCLR pretraining of a resnet
        "pretrain_architecture": str(os.getenv("PRETRAIN_ARCHITECTURE")), # type of pretrained vision encoder
        "minio_bucket_name": str(os.getenv("MINIO_BUCKET_NAME")),  # source of hierarchical data (imgs and jsons) for pretraining
        "pretrain_max_train_batches": int(os.getenv("PRETRAIN_MAX_TRAIN_BATCHES") or -1),
        "pretrain_max_val_batches": int(os.getenv("PRETRAIN_MAX_VAL_BATCHES") or -1),

        # global A3C:
        "batch_size": int(os.getenv("BATCH_SIZE") or -1),  # batch size == size of agent replay memory
        "action_dim": int(os.getenv("ACTION_DIM") or -1),  # number of actions for RL agent. Depends on env
        "no_vision_state_dim": 4,  # number of features if no vision encoder is used. Hardcoded for Cartpole env POC
        "pretrained_vision_encoder_model_path": str(os.getenv("PRETRAINED_VISION_ENCODER_PATH", "")),
        "freeze_pretrained_vision_encoder_weights": str2bool(os.getenv("FREEZE_PRETRAINED_VISION_ENCODER_WEIGHTS")),
        "unfreeze_pretrained_vision_encoder_weights": str2bool(os.getenv("FREEZE_PRETRAINED_VISION_ENCODER_WEIGHTS")), # unfreeze all vision encoder layers after steps == total_steps/10
        "finetuning_warmup_steps": int(os.getenv("FINETUNING_WARMUP_STEPS") or -1),
        "global_step_counter_T_max": int(os.getenv("MAX_GLOBAL_NET_STEPS") or -1),  # max steps disabled for this experiment
        "learning_rate":  float(os.getenv("LEARNING_RATE") or -1),  # initial lr
        "optimizer":  str(os.getenv("OPTIMIZER", "")).lower(),  # optimizer name: shared_adam | shared_rmsprop
        "dense_hidden_dim": 256,  # if model architecture has hidden dense layers
        "dropout": float(os.getenv("DROPOUT") or 0.0),  # dropouts for dense layers. Default: 0.

        "icon_classifier_model_dir": os.getenv("PRETRAINED_ICON_CLASSIFIER"),  # Minecraft agent needs item classifier to read reward
        "net_architecture": str(os.getenv("NET_ARCHITECTURE","")).lower(),  # same model architecture used for global net and agents
        "target_score": int(os.getenv("TARGET_SCORE") or -1),  # target score for early stopping
        "early_stopping_tolerance": int(os.getenv("EARLY_STOPPING_TOLERANCE")  or -1),  # number of global net epochs having mean score of all agents > "target_score"
        "local_global_net": str2bool(os.getenv("LOCAL_GLOBAL_NET")),  # Enable this if using only 1 agent, build in global net is used on agents side so no rest api is needed
        "scheduler": str(os.getenv("SCHEDULER")),  # Name of learning rate scheduler (See hardcoded scheduler parameters in code)

        # agent
        "mode": str(os.getenv("MODE", "train_a3c")).lower(),  # train mode (train weights) or test mode (just use weights)
        "dry_run": str2bool(os.getenv("DRY_RUN")),  # debug mode
        "show_popups": str2bool(os.getenv("SHOW_POPUPS")),  # for Minecraft env. Popup shows chosen action outside of screenshot area
        "t_per_second": float(os.getenv("AGENT_STEPS_PER_SECOND") or -1),  # must be the same for each agent. Depends on task
        "create_new_world_after_epoch": str2bool(os.getenv("AGENT_CREATE_NEW_WORLD_AFTER_EPOCH")),  # for Minecraft env
        "world_seed": str(os.getenv("AGENT_WORLD_SEED", "29")),  # world generation seed for Minecraft env
        "steps_per_epoch": int(os.getenv("AGENT_STEPS_PER_EPOCH") or -1),  # depends on task. Is equal to max batch size
        "epsilon_max": float(os.getenv("AGENT_EPSILON_MAX") or -1),  # for greedy exploration (not in original A3C paper)
        "epsilon_min": float(os.getenv("AGENT_EPSILON_MIN") or -1),    # for greedy exploration
        "epsilon_decay": float(os.getenv("AGENT_EPSILON_DECAY") or -1),  # per global step (=agent epoch)
        "entropy_coef": float(os.getenv("AGENT_ENTROPY_COEF") or -1),  # factor for using entropy loss in loss calculation
        "gamma": float(os.getenv("AGENT_GAMMA") or -1),    # for reinforcement learning loss
        "task_item_key": str(os.getenv("TASK_ITEM_KEY", "oak_wood")),   # for Minecraft env. Name of item
        "persist_transitions": str2bool(os.getenv("PERSIST_TRANSITIONS")),  # if transitions should be saved on hard disk after epoch
        "agent_persisted_memory_out_dir": os.getenv("AGENT_PERSISTENT_MEMORY_OUT_DIR", "./tmp"),  # Output path if "persist_transitions"
        "pretrain_augmentation": str(os.getenv("PRETRAIN_AUGMENTATIONS", "")).replace(" ", "").split(","),
        "world_creation_mode": int(os.getenv("WORLD_CREATION_MODE", 0) or 0),  # default 0, Choose set of hard coded commands to recreate agents environment
    }
    # add additional parameters depending on other parameters
    pretrain_mode = "no pretraining"
    pretrained = False
    unfrozen = False
    if 'pretrained_vision_encoder_model_path' in train_config:
        if train_config['pretrained_vision_encoder_model_path'] is not None and \
                train_config['pretrained_vision_encoder_model_path'] != "None" and train_config['pretrained_vision_encoder_model_path'] != "":
            pretrained = True
    if 'unfreeze_pretrained_vision_encoder_weights' in train_config:
        if train_config['unfreeze_pretrained_vision_encoder_weights']:
            unfrozen = True
    if pretrained and unfrozen:
        pretrain_mode = "finetuning"

    if pretrained and not unfrozen:
        pretrain_mode = "linear probing"
    train_config["pretrain_mode"] = pretrain_mode  # for wandb logging

    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")
    if not os.path.exists("./wandb_tmp"):
        os.makedirs("./wandb_tmp")
    if not os.path.exists("./used-config"):
        os.makedirs("./used-config")

    save_dict_as_json(train_config, None, config_path, sort_keys=False)


    if not os.path.isfile(config_path):
        raise Exception("Train config was not created.")

    logging.info("train_config")
    logging.info(train_config)

def get_train_config():
    set_train_config()
    return load_from_json_file("./used-config/train_config.json")