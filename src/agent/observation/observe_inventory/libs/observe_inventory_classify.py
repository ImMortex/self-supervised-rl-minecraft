import logging
from collections import OrderedDict

import cv2
import numpy as np
import torch
from torch import nn

from src.agent.observation.observe_inventory.libs.classify_minecraft_item_icon import classify_minecraft_item_icon, \
    model_predict
from src.agent.observation.observe_inventory.libs.inventory_img_utils import adjust_inv_item_background
from src.agent.observation.observe_inventory.libs.observe_inventory_lib import to_gray
from src.agent.observation.trainers.simple_icon_classifier_utils import get_model
from src.common.helpers.helpers import load_from_json_file

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def load_image_classifier_model(model_dir: str):
    # Recreate the exact same model, including its weights and the optimizer
    train_config: dict = load_from_json_file(model_dir + "config.json")
    class_labels_data: dict = load_from_json_file(model_dir + "class_labels.json")
    int_2_label = class_labels_data["int_2_label"]
    class_labels = list(int_2_label.values())
    model: nn.Module = get_model(len(class_labels), train_config["model"], in_channels=train_config["channels"])
    trained_model_state_dict: OrderedDict = torch.load(model_dir + "model_state_dict.pt",
                                                       map_location=torch.device('cpu'))
    model.load_state_dict(trained_model_state_dict)
    logging.info('Icon classifier model loaded: ' + train_config["model"] + " from " + model_dir)
    logging.info('Using device for prediction: ' + str(device))

    model.eval()
    model = model.to(device)

    logging.info("testing prediction with device ...")
    test_img: np.ndarray = np.zeros((60, 60, 3), np.uint8)
    model_predict(test_img, model, config=train_config, int_2_label=int_2_label)
    logging.info("testing prediction finished")

    return model, train_config, int_2_label


def classify_item_in_slot(slot_img: np.ndarray, icon_classifier_model, icon_classifier_train_config: dict,
                          icon_classifier_int_2_label) -> str:
    tresh = cv2.threshold(to_gray(slot_img), 127, 255, cv2.THRESH_BINARY)[1]
    count_black_pixels = np.sum(tresh == 0)

    if count_black_pixels < 100:
        return "None"

    slot_img = adjust_inv_item_background(slot_img)

    classified_item = classify_minecraft_item_icon(slot_img, icon_classifier_model, icon_classifier_train_config,
                                                   icon_classifier_int_2_label)
    if "air" in classified_item.lower():
        classified_item = "None"

    return classified_item
