import copy
import json
import logging
import os
import time
import traceback

import cv2
import numpy as np
import torch
from PIL import Image
from dotenv import load_dotenv
from expiringdict import ExpiringDict
from minio import Minio
from torch.utils.data import Dataset

from src.common.helpers.helpers import load_from_json_file
from src.common.minio_fncts.minio_api_calls import download_image, download_json
from src.common.observation_keys import health_key, hunger_key, level_key, experience_key, view_key
from src.common.persisted_memory import PersistedMemory, IMAGE_ENDING, TRANSITION_ENDING
from src.common.resource_metrics import get_resource_metrics
from src.common.transition import Transition
from src.dataloader.transform_functions import get_img_to_tensor_transform

load_dotenv()

RAM_GB_LIMIT = os.getenv("RAM_GB_LIMIT")
MINIO_DATASET_TENSOR_CACHE_SIZE = os.getenv("MINIO_DATASET_TENSOR_CACHE_SIZE")
MINIO_DATASET_TENSOR_CACHE_MAX_AGE = os.getenv("MINIO_DATASET_TENSOR_CACHE_MAX_AGE")
minio_dataset_tensor_cache_size: int = 100
minio_dataset_tensor_cache_max_age: int = 7200
if MINIO_DATASET_TENSOR_CACHE_SIZE is not None:
    minio_dataset_tensor_cache_size = int(MINIO_DATASET_TENSOR_CACHE_SIZE)

if MINIO_DATASET_TENSOR_CACHE_MAX_AGE is not None:
    minio_dataset_tensor_cache_max_age = int(MINIO_DATASET_TENSOR_CACHE_MAX_AGE)


def append_features(t: Transition, state_seq: [], cache: ExpiringDict, state_dim):
    key = "Tr" + str(t.t)
    if cache is not None and key in cache:
        state = cache[key]
    else:
        hp = float(t.state[health_key])
        hunger = float(t.state[hunger_key])
        exp_progress = float(t.state[experience_key])
        exp_level = float(t.state[level_key])

        state = []
        state.append(hp)  # state[0]
        state.append(hunger)  # state[1]
        state.append(exp_progress)  # state[2]
        state.append(exp_level)  # state[3]

        if cache is not None and enough_ram_available():
            cache[key] = state

    state_seq.append(state)


def enough_ram_available():
    return get_resource_metrics()["ram_gb_used_main_thread"] < float(RAM_GB_LIMIT) * 0.8


def append_resized_img_as_tensor(transition_id, img: np.ndarray, image_tensors: [], width_2d, height_2d,
                                 cache: ExpiringDict, idx, debug: str):
    key = "I" + str(transition_id)
    if cache is not None and key in cache:
        tensor = cache[key]
    else:
        img = cv2.resize(img, (width_2d, height_2d))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img: Image = Image.fromarray(img)
        try:
            if idx == 0:
                img.save("tmp/dataloader_image_" + debug + ".png")  # for debug
        except Exception as e:
            logging.error(e)
        img_to_tensor_transform = get_img_to_tensor_transform()
        tensor = img_to_tensor_transform(img)

        if cache is not None and enough_ram_available():
            cache[key] = tensor
    image_tensors.append(tensor)


def append_resized_img(transition_id, img: np.ndarray, images: [], width_2d, height_2d,
                       cache: ExpiringDict, idx, debug: str):
    key = "I" + str(transition_id)
    if cache is not None and key in cache:
        img = cache[key]
    else:
        img = cv2.resize(img, (width_2d, height_2d))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img: Image = Image.fromarray(img)

        if cache is not None and enough_ram_available():
            cache[key] = img
    images.append(img)


def get_data_sequence_paths(idx, first_transition_abs_paths, x_depth):
    first_path_of_seq = first_transition_abs_paths[idx]
    first_id = PersistedMemory.get_transition_id_from_path(first_path_of_seq)
    paths = []
    for i in range(x_depth):
        path: str = PersistedMemory.get_transition_neighbour_path(first_path_of_seq, first_id + i)
        paths.append(path)
    return paths


def create_tensor_dict_3d_images(x_depth, image_tensors, state_seq, state_dim):
    # https://pytorch.org/docs/stable/generated/torch.stack.html, add new dimension at position -1
    # that matches the dimensions as described in paper https://arxiv.org/abs/2111.14791
    tensor_3d_image: torch.Tensor = torch.stack(image_tensors, dim=-1)
    tensor_3d_image = tensor_3d_image / 255  # Normalize to torch.float32 in the range [0, 1]
    tensor_3d_image = tensor_3d_image.to(torch.float)

    # 3 img Channels, Height, Width, Depth:  C x H x W x D

    # sequences of length D as tensor
    tensor = torch.from_numpy(np.array(state_seq))
    if len(state_seq) >= x_depth:
        tensor_state_seq: torch.Tensor = tensor.view(x_depth, state_dim).to(torch.float)
    else:
        tensor_state_seq = None  # no tensor available
    dictionary: dict = {
        "tensor_image": tensor_3d_image,  # vision data
    }

    if tensor_state_seq is not None:
        dictionary["tensor_state_seq"] = tensor_state_seq  # no vision data

    return dictionary


def create_tensor_dict_2d_images(x_depth, image_tensor, state_seq, state_dim):
    # https://pytorch.org/docs/stable/generated/torch.stack.html, add new dimension at position -1
    # that matches the dimensions as described in paper https://arxiv.org/abs/2111.14791
    tensor_image = image_tensor / 255  # Normalize to torch.float32 in the range [0, 1]
    tensor_image = tensor_image.to(torch.float)

    # 3 img Channels, Height, Width, Depth:  C x H x W x D

    # sequences of length D as tensor
    tensor = torch.from_numpy(np.array(state_seq))
    if len(state_seq) >= x_depth:
        tensor_state_seq: torch.Tensor = tensor.view(x_depth, state_dim).to(torch.float)
    else:
        tensor_state_seq = None  # no tensor available
    dictionary: dict = {
        "tensor_image": tensor_image,  # vision data
    }

    if tensor_state_seq is not None:
        dictionary["tensor_state_seq"] = tensor_state_seq  # no vision data

    return dictionary


def get_img_seq(images: []):
    seq_new_obs = None
    for img in images:
        if seq_new_obs is None:
            seq_new_obs = img
        else:
            seq_new_obs = np.concatenate((seq_new_obs, img), axis=1)

    return seq_new_obs


class S3MinioCustomDataset(Dataset):
    """
    The transitions are downloaded from the addressed minio bucket the first time they are used
    and cached if enough free RAM exists.
    """

    def __init__(self, s3_transition_paths_grouped_by_agent: [], x_depth: int, width_2d: int, height_2d: int,
                 minio_client: Minio, bucket_name: str, state_dim=4, only_use_images=True, seq_to_3d_image=True):
        self.seq_to_3d_image = seq_to_3d_image
        self.only_use_images = only_use_images
        self.state_dim = state_dim
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.first_transition_abs_paths = []  # nach Validierung wird jeweils erster path von x gespeichert (Rest ergibt sich)
        self.x_depth = x_depth  # D
        self.width_2d = width_2d  # W
        self.height_2d = height_2d  # H

        # caching: https://www.pluralsight.com/guides/explore-python-libraries:-in-memory-caching-using-expiring-dict
        self.cache: ExpiringDict = ExpiringDict(max_len=max(100 * self.x_depth, 100),
                                                max_age_seconds=1.2 * self.x_depth,
                                                items={})

        self.tensor_cache: ExpiringDict = ExpiringDict(max_len=minio_dataset_tensor_cache_size,
                                                       max_age_seconds=minio_dataset_tensor_cache_max_age,
                                                       items={})

        self.length = 1
        if s3_transition_paths_grouped_by_agent is not None and len(s3_transition_paths_grouped_by_agent) > 0:
            self.validate_paths(s3_transition_paths_grouped_by_agent)
        else:
            raise Exception("S3MinioCustomDataset cannot be created.")

        self.previous_tensor_dict = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        try:
            tensor_dict = self.get_tensor_dict(idx)
            if tensor_dict is not None:  # if no error happened
                self.previous_tensor_dict = copy.deepcopy(tensor_dict)
                return self.get_tensor_dict(idx)
        except Exception as e:
            print(e)
            logging.error(" self.get_tensor_dict minio failed. Fatal error")
            traceback.print_exc()
        return self.previous_tensor_dict  # on error return previous data example to avoid training run crash

    def validate_paths(self, transition_paths_grouped_by_agent):
        """
        Valid sequences of transitions are generated.
        It is ensured that in each case a number x_depth of transitions is located in correct order without gaps.
        Transitions that do not fit into any sequence are ignored and not counted in the data set.

        This way self.length is set and is the number of valid sequences.
        @param transition_paths_grouped_by_agent: base paths for existing files for each transition
        """

        for transition_paths_one_agent in transition_paths_grouped_by_agent:
            for index, abs_path in enumerate(transition_paths_one_agent):
                abs_paths = transition_paths_one_agent[index:index + self.x_depth]
                if len(abs_paths) == self.x_depth:
                    transition_ids = [PersistedMemory.get_transition_id_from_path(p) for p in abs_paths]
                    prev_timestep: int = None
                    valid = True
                    for t_id in transition_ids:
                        if prev_timestep is None:
                            prev_timestep = t_id
                        else:
                            if t_id == prev_timestep + 1:
                                prev_timestep = t_id
                            else:
                                valid = False
                                break
                    if valid:
                        self.first_transition_abs_paths.append(abs_paths[0])
        self.length = len(self.first_transition_abs_paths)
        if self.length == 0:
            raise Exception("S3MinioCustomDataset has no data after validation.")

    def get_tensor_dict(self, idx) -> dict:
        try:
            if str(idx) in self.tensor_cache:
                return self.tensor_cache.get(str(idx))
            # stack of 2D images with length D
            image_tensors: [] = []

            images: [] = []

            # sequences of length D
            state_seq: [] = []  # state except img

            paths = get_data_sequence_paths(idx, self.first_transition_abs_paths, self.x_depth)

            transition: Transition = None
            img: np.ndarray = None
            for path in paths:
                transition_file, view_file = PersistedMemory.get_transition_file_paths(path)

                needed_retries: int = 0
                transition_id = 0
                if path in self.cache:
                    transition, img = self.cache.get(path)
                else:
                    # retry if connection to server is lost
                    while True:
                        error = False
                        try:
                            start_download_time = time.time()
                            logging.info(
                                "Downloading Transtion " + str(path))
                            transition_id = int(os.path.splitext(os.path.basename(view_file))[0].split("_")[-1])

                            img = download_image(self.minio_client, self.bucket_name, view_file)
                            if not self.only_use_images:
                                transition_dict: dict = json.loads(
                                    download_json(self.minio_client, self.bucket_name, transition_file))
                                transition = PersistedMemory.deserialize_transition(transition_dict=transition_dict,
                                                                                    image=None)
                            logging.info(
                                "Transition downloaded from minio in " + str(time.time() - start_download_time))
                            logging.info("minio data cached: " + str(len(self.cache.items())) +
                                         " tensors cached: " + str(len(self.tensor_cache.items())))

                        except Exception as e:
                            logging.error("Transition download from minio failed")
                            logging.error(str(view_file))
                            logging.error(e)
                            error = True
                            traceback.print_exc()

                        # transition json is optional
                        if img is not None and not error:
                            break  # everything went good
                        needed_retries += 1
                        time.sleep(5)
                        logging.error("Transition download from minio error. Retries: " + str(needed_retries))

                if self.seq_to_3d_image:
                    append_resized_img_as_tensor(transition_id, img, image_tensors,
                                                 self.width_2d, self.height_2d, self.cache, idx, "minio")
                else:
                    append_resized_img(transition_id, img, images,
                                       self.width_2d, self.height_2d, self.cache, idx, "minio")

                if transition is not None:
                    append_features(transition, state_seq, self.cache, self.state_dim)
                self.cache[path] = (transition, img)

            if self.seq_to_3d_image:
                dictionary = create_tensor_dict_3d_images(self.x_depth, image_tensors, state_seq,
                                                          self.state_dim)
            else:
                img_seq = get_img_seq(images)
                if img_seq.__class__.__name__ != "Image":
                    img_seq: Image = Image.fromarray(img_seq)
                try:
                    if idx == 0:
                        img_seq.save("tmp/dataloader_image_minio.png")  # for debug
                except Exception as e:
                    logging.error(e)
                img_to_tensor_transform = get_img_to_tensor_transform()
                image_tensor = img_to_tensor_transform(img_seq)
                dictionary = create_tensor_dict_2d_images(self.x_depth, image_tensor, state_seq,
                                                          self.state_dim)

            if enough_ram_available():
                self.tensor_cache[str(idx)] = dictionary
            return dictionary

        except Exception as e:
            print(e)
            logging.error("Transition download from minio failed. Fatal error")
            traceback.print_exc()

        return None


class AgentCustomDataset(Dataset):
    """
    Dataset to get actual state sequence during agent training
    """

    def __init__(self, transition_seq: [], x_depth: int, width_2d: int, height_2d: int, cache: ExpiringDict = None,
                 state_dim=4, seq_to_3d_image=True):

        self.seq_to_3d_image = seq_to_3d_image
        self.state_dim = state_dim
        self.transition_seq = transition_seq
        self.x_depth = x_depth  # D alias state sequence length
        self.width_2d = width_2d  # W
        self.height_2d = height_2d  # H
        self.length = 1
        self.cache = cache

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.get_tensor_dict(idx)

    def get_tensor_dict(self, idx) -> dict:
        # stack of 2D images with length D
        image_tensors: [] = []
        images: [] = []

        # sequences of length D
        state_seq: [] = []  # state except img

        if self.transition_seq is not None and len(self.transition_seq) >= self.x_depth:
            transition_seq = self.transition_seq[:self.x_depth]
            for transition in transition_seq:
                if self.seq_to_3d_image:
                    append_resized_img_as_tensor(transition.t, transition.state[view_key], image_tensors,
                                                 self.width_2d, self.height_2d, self.cache, idx, "agent")
                else:
                    append_resized_img(transition.t, transition.state[view_key], images,
                                       self.width_2d, self.height_2d, self.cache, idx, "minio")
                append_features(transition, state_seq, self.cache, self.state_dim)
        else:
            raise Exception("AgentCustomDataset get_tensor_dict() not enough data")

        if self.seq_to_3d_image:
            dictionary = create_tensor_dict_3d_images(self.x_depth, image_tensors, state_seq, self.state_dim)
        else:
            img_seq = get_img_seq(images)
            if img_seq.__class__.__name__ != "Image":
                img_seq: Image = Image.fromarray(img_seq)
            try:
                if idx == 0:
                    img_seq.save("tmp/dataloader_image_minio.png")  # for debug
            except Exception as e:
                logging.error(e)
            img_to_tensor_transform = get_img_to_tensor_transform()
            image_tensor = img_to_tensor_transform(img_seq)
            dictionary = create_tensor_dict_2d_images(self.x_depth, image_tensor, state_seq,
                                                      self.state_dim)
        return dictionary


class HardDiskCustomDataset(Dataset):
    """
    DEPRECATED
    """

    def __init__(self, transition_abs_paths_grouped: [], x_depth: int, width_2d: int, height_2d: int,
                 state_dim=4):
        self.state_dim = state_dim
        self.first_transition_abs_paths = []  # after validation the first path of x is stored

        self.x_depth = x_depth  # D
        self.width_2d = width_2d  # W
        self.height_2d = height_2d  # H
        self.cache: ExpiringDict = ExpiringDict(max_len=max(10 * self.x_depth, 100), max_age_seconds=1.2 * self.x_depth,
                                                items={})

        self.length = 1
        if transition_abs_paths_grouped is not None and len(transition_abs_paths_grouped) > 0:
            self.validate_paths(transition_abs_paths_grouped)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.get_tensor_dict(idx)

    def validate_paths(self, transition_abs_paths_grouped):
        # Pfade mit nicht lesbaren Daten entfernen
        # Pfade, die keine Knn=1 Nachbarn für x_depth + 1 mit korrekter id haben, werden entfernt
        # basierend auf x_depth wird self.length berechnet

        """
        STEP 1: Filter out invalid paths
        """
        filtered_transition_abs_paths_grouped = []
        for transition_abs_paths in transition_abs_paths_grouped:
            transition_abs_paths = [p for p in transition_abs_paths if PersistedMemory.is_transition_valid(p)]
            filtered_transition_abs_paths_grouped.append(transition_abs_paths)

        transition_abs_paths_grouped = filtered_transition_abs_paths_grouped

        """
        STEP 2: Save first path of valid sequences
        """
        for transition_abs_paths in transition_abs_paths_grouped:
            for index, abs_path in enumerate(transition_abs_paths):
                abs_paths = transition_abs_paths[index:index + self.x_depth]
                if len(abs_paths) == self.x_depth:
                    transition_ids = [PersistedMemory.get_transition_id_from_path(p) for p in abs_paths]
                    prev_timestep: int = None
                    valid = True
                    for t_id in transition_ids:
                        if prev_timestep is None:
                            prev_timestep = t_id
                        else:
                            if t_id == prev_timestep + 1:
                                prev_timestep = t_id
                            else:
                                valid = False
                                break
                    if valid:
                        self.first_transition_abs_paths.append(abs_paths[0])
        self.length = len(self.first_transition_abs_paths)
        if self.length == 0:
            raise Exception("HardDiskCustomDataset has no data after validation.")

    def get_tensor_dict(self, idx) -> dict:
        # stack of 2D images with length D
        image_tensors: [] = []

        # sequences of length D
        state_seq: [] = []  # state except img

        # data from file paths
        if len(self.first_transition_abs_paths) > idx:
            paths = get_data_sequence_paths(idx, self.first_transition_abs_paths, self.x_depth)
            for path in paths:
                transition_dict: dict = load_from_json_file(path + TRANSITION_ENDING)
                transition: Transition = PersistedMemory.deserialize_transition(transition_dict, None)

                image: np.ndarray = cv2.imread(path + IMAGE_ENDING)
                append_resized_img_as_tensor(transition.t, image, image_tensors, self.width_2d, self.height_2d,
                                             self.cache, idx,
                                             debug="harddisk")

                append_features(transition, state_seq, self.cache, self.state_dim)
        else:
            raise Exception("HardDiskCustomDataset get_tensor_dict() not enough data")

        dictionary = create_tensor_dict_3d_images(self.x_depth, image_tensors, state_seq, self.state_dim)

        return dictionary


class DatasetForBatches(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        x = self.data[index]
        return x

    def __len__(self):
        return len(self.data)
