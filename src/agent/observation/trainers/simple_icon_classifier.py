"""
Using only one sample for each class because the icons in minecraft are unique and always the same
"""
import copy
import logging
import math
# used tutorial: https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/
import os
import socket
from datetime import datetime
from os.path import isfile, join

import cv2
# set the matplotlib backend so figures can be saved in the background
import matplotlib
import wandb
from PIL import Image
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from src.agent.observation.observe_inventory.libs.inventory_img_utils import adjust_inv_item_background
from src.agent.observation.observe_inventory.libs.observe_inventory_classify import classify_item_in_slot
from src.agent.observation.observe_inventory.libs.observe_inventory_lib import adjust_img_color_format, \
    get_cropped_slot_img
from src.agent.observation.observe_inventory.libs.observe_inventory_recipe_book import is_inventory_recipe_book_open
from src.agent.observation.observe_inventory.libs.observe_inventory_slots import get_slot_positions
from src.agent.observation.trainers.simple_icon_classifier_utils import custom_preprocess_data_x, custom_augment_data_x, \
    cv_img_list_to_torch_tensor, predict, resize_data_x, get_model, one_hots_to_labels

matplotlib.use("Agg")
# import the necessary packages

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
import time

from src.common.helpers.helpers import save_dict_as_json, load_from_json_file
from src.common.env_utils.minecraft_filename import icon_filename_to_display_name

inv_conf: dict = load_from_json_file("./config/inventoryScreenshotNoFullscreenConf.json")
custom_path = "./config/inventoryScreenshotNoFullscreenConfCustom.json"
if os.path.isfile(custom_path):
    inv_conf: dict = load_from_json_file(custom_path)

hostname = socket.gethostname()




device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))


def get_acc_test_data_1(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label):
    screenshot_path = "tests/observation/img/test_icon_classifier2.png"
    screenshot: np.ndarray = np.array(Image.open(screenshot_path))
    screenshot = adjust_img_color_format(screenshot)
    true_slots: dict = {'0': 'coal_ore', '1': 'deepslate_coal_ore', '2': 'iron_ore', '3': 'deepslate_iron_ore',
                        '4': 'copper_ore',
                        '5': 'deepslate_copper_ore', '6': 'gold_ore', '7': 'deepslate_gold_ore',
                        '8': 'redstone_ore',
                        '9': 'deepslate_redstone_ore', '10': 'emerald_ore', '11': 'deepslate_emerald_ore',
                        '12': 'lapis_ore',
                        '13': 'deepslate_lapis_ore', '14': 'diamond_ore', '15': 'deepslate_diamond_ore',
                        '16': 'raw_iron_block', '17': 'raw_copper_block', '18': 'raw_gold_block',
                        '19': 'polished_granite',
                        '20': 'diorite', '21': 'polished_diorite', '22': 'andesite', '23': 'polished_andesite',
                        '24': 'deepslate', '25': 'cobbled_deepslate', '26': 'dark_oak_sapling',
                        '27': 'oak_sapling', '28': 'spruce_sapling', '29': 'birch_sapling', '30': 'jungle_sapling',
                        '31': 'acacia_sapling', '32': 'wheat_seeds', '33': 'apple',
                        '34': 'poppy', '35': 'dandelion'}
    classified_dict = get_classified_inventory_dict(screenshot, icon_classifier_model,
                                                    icon_classifier_train_config,
                                                    icon_classifier_int_2_label)
    acc = inventory_accuracy(classified_dict, true_slots)
    print("test_icon_classifier2 unseen test data acc: " + str(acc))
    return acc


def get_acc_test_data_2(icon_classifier_model, icon_classifier_train_config, icon_classifier_int_2_label):
    screenshot_path = "tests/observation/img/test_icon_classifier3.png"
    screenshot: np.ndarray = np.array(Image.open(screenshot_path))
    screenshot = adjust_img_color_format(screenshot)
    true_slots: dict = {'0': 'crafting_table', '1': 'furnace', '2': 'chest', '3': 'enchanting_table',
                        '4': 'composter',
                        '5': 'barrel', '6': 'smoker', '7': 'blast_furnace',
                        '8': 'white_bed',
                        '9': 'gray_bed', '10': 'black_bed', '11': 'light_gray_bed',
                        '12': 'wooden_sword',
                        '13': 'stone_sword', '14': 'golden_sword', '15': 'iron_sword',
                        '16': 'diamond_sword', '17': 'iron_helmet', '18': 'iron_chestplate',
                        '19': 'iron_boots',
                        '20': 'iron_leggings', '21': 'diamond_helmet', '22': 'diamond_chestplate',
                        '23': 'diamond_leggings',
                        '24': 'diamond_boots', '25': 'leather_cap', '26': 'leather_tunic',
                        '27': 'leather_pants', '28': 'leather_boots', '29': 'golden_helmet',
                        '30': 'golden_chestplate',
                        '31': 'golden_leggings', '32': 'golden_boots', '33': 'flint_and_steel',
                        '34': 'bow', '35': 'arrow'}
    classified_dict = get_classified_inventory_dict(screenshot, icon_classifier_model,
                                                    icon_classifier_train_config,
                                                    icon_classifier_int_2_label)
    acc = inventory_accuracy(classified_dict, true_slots)
    print("test_icon_classifier3 unseen test data acc: " + str(acc))
    return acc


def get_classified_inventory_dict(screenshot, icon_classifier_model, icon_classifier_train_config,
                                  icon_classifier_int_2_label):
    sp = get_slot_positions(is_inventory_recipe_book_open(screenshot))
    classified_dict: dict = {}
    for slot_id in range(len(sp)):
        classified_item = classify_item_in_slot(get_cropped_slot_img(screenshot, slot_id, sp),
                                                icon_classifier_model, icon_classifier_train_config,
                                                icon_classifier_int_2_label)
        classified_dict[str(slot_id)] = classified_item
    return classified_dict


def inventory_accuracy(classified_dict, true_slots):
    correct_counter: int = 0
    for true_key, key in zip(true_slots, classified_dict):
        if true_slots[true_key] == classified_dict[true_key]:
            correct_counter += 1
    acc = correct_counter / len(true_slots)
    return acc


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class IconClassifierTrainer:

    def __init__(self, config_dict):
        self.config_dict: dict = config_dict
        self.best_model = None
        self.best_model_history: dict = {}
        self.best_model_filename = ""
        self.metrics: dict = {}

    def training(self, trainings: int = 1):
        print("loading data...")
        config_dict = self.config_dict
        BATCH_SIZE = config_dict['batch_size']

        training_data_dir = "agent_assets/icon_classifier_data/minecraft_icons"

        base_train_x = np.array([cv2.imread(f.path) for f in os.scandir(training_data_dir)])
        icon_file_names = [f for f in os.listdir(training_data_dir) if
                           isfile(join(training_data_dir, f)) and f.endswith(".png")]

        class_labels = []
        for icon_file_name in icon_file_names:
            label = icon_filename_to_display_name(icon_file_name)
            class_labels.append(label)

        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(np.array(class_labels))
        # onehot encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        one_hot_encodings = onehot_encoder.fit_transform(integer_encoded)
        onehot_encoder.categories = onehot_encoder.categories_
        classes_count: int = len(class_labels)
        # encoder = LabelBinarizer()
        # one_hot_encodings = encoder.fit_transform(class_labels)

        label_2_int: dict = {}
        int_2_label: dict = {}
        for label, integer in zip(class_labels, list(integer_encoded)):
            integer = integer[0]
            label_2_int[label] = integer
            int_2_label[str(integer)] = label

        extra_val_x, extra_val_y_one_hot = self.get_extra_val_data(label_2_int, onehot_encoder)
        extra_x, extra_y_one_hot = self.get_extra_train_data(label_2_int, onehot_encoder)

        print("preprocess data...")
        base_train_x = resize_data_x(base_train_x, config_dict)
        extra_val_x = resize_data_x(extra_val_x, config_dict)
        # multiply amount of training data using augmentation
        train_x_1 = custom_augment_data_x(base_train_x.copy(), config_dict=config_dict)
        train_x_2 = custom_augment_data_x(base_train_x.copy(), inv_conf["slot_width"], inv_conf["slot_height"],
                                          config_dict=config_dict)
        train_x_3 = custom_augment_data_x(extra_val_x, resize_width=inv_conf["slot_width"],
                                          resize_height=inv_conf["slot_height"], config_dict=config_dict)
        train_y_3 = extra_val_y_one_hot
        train_x_4 = custom_augment_data_x(base_train_x.copy(), inv_conf["slot_width"] * 2, inv_conf["slot_height"] * 2,
                                          config_dict=config_dict)
        train_x_5 = custom_augment_data_x(base_train_x.copy(), inv_conf["slot_width"], inv_conf["slot_height"] + 1,
                                          config_dict=config_dict)
        train_x_6 = custom_augment_data_x(base_train_x.copy(), inv_conf["slot_width"], inv_conf["slot_height"] + 2,
                                          config_dict=config_dict)
        train_x_7 = custom_augment_data_x(base_train_x.copy(), inv_conf["slot_width"], inv_conf["slot_height"] + 3,
                                          config_dict=config_dict)
        train_x_8 = custom_augment_data_x(base_train_x.copy(), inv_conf["slot_width"] + 1, inv_conf["slot_height"],
                                          config_dict=config_dict)
        train_x_9 = custom_augment_data_x(base_train_x.copy(), inv_conf["slot_width"] + 2, inv_conf["slot_height"],
                                          config_dict=config_dict)
        train_x_10 = custom_augment_data_x(base_train_x.copy(), inv_conf["slot_width"] + 3, inv_conf["slot_height"],
                                           config_dict=config_dict)
        train_x_11 = custom_augment_data_x(base_train_x.copy(), int(inv_conf["slot_width"] * 1.1),
                                           inv_conf["slot_height"], config_dict=config_dict)
        train_x_12 = custom_augment_data_x(base_train_x.copy(), inv_conf["slot_width"],
                                           int(inv_conf["slot_height"] * 1.1), config_dict=config_dict)

        train_x_13 = custom_augment_data_x(extra_x, resize_width=inv_conf["slot_width"],
                                           resize_height=inv_conf["slot_height"], config_dict=config_dict)
        train_y_13 = extra_y_one_hot

        train_x = np.concatenate((train_x_1,
                                  train_x_2,
                                  train_x_3,
                                  train_x_4,
                                  train_x_5,
                                  train_x_6,
                                  train_x_7,
                                  train_x_8,
                                  train_x_9,
                                  train_x_10,
                                  train_x_11,
                                  train_x_12,
                                  train_x_13
                                  ), axis=0)

        train_x = custom_preprocess_data_x(train_x, config_dict=config_dict)
        train_y = np.concatenate((one_hot_encodings,
                                  one_hot_encodings,
                                  train_y_3,
                                  one_hot_encodings,
                                  one_hot_encodings,
                                  one_hot_encodings,
                                  one_hot_encodings,
                                  one_hot_encodings,
                                  one_hot_encodings,
                                  one_hot_encodings,
                                  one_hot_encodings,
                                  one_hot_encodings,
                                  train_y_13,
                                  ), axis=0)

        val_x = custom_augment_data_x(extra_val_x, resize_width=inv_conf["slot_width"],
                                      resize_height=inv_conf["slot_height"], config_dict=config_dict)
        val_x = custom_preprocess_data_x(val_x, config_dict=config_dict)
        val_y = extra_val_y_one_hot

        # debug
        debug_dir = "tmp/train_icon_classifier"
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        count = 0
        for x, y in zip(train_x, train_y):
            y = torch.Tensor(y)
            y = y.detach().cpu().numpy()
            index = y.argmax(axis=None)
            prediction = int_2_label[str(index)]
            if prediction == "oak_log":
                # cv2.imshow("image", x)
                # cv2.waitKey(0)
                cv2.imwrite(os.path.join(debug_dir, "train_" + str(count) + "_" + str(prediction) + ".png"), x * 255)
                # print(prediction)
            count += 1

        # transform to torch tensor
        train_tensor_x = cv_img_list_to_torch_tensor(train_x)
        train_tensor_y = torch.Tensor(np.array(train_y))
        val_tensor_x = cv_img_list_to_torch_tensor(val_x)
        val_tensor_y = torch.Tensor(np.array(val_y))

        train_dataset = TensorDataset(train_tensor_x, train_tensor_y)
        val_dataset = TensorDataset(val_tensor_x, val_tensor_y)

        # initialize the train, validation, and test data loaders
        trainDataLoader = DataLoader(train_dataset, shuffle=True,
                                     batch_size=BATCH_SIZE)
        valDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        testDataLoader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

        for _ in range(trainings):
            self.train_model(classes_count, int_2_label, testDataLoader, trainDataLoader, valDataLoader)

    def get_extra_val_data(self, label_2_int, onehot_encoder):
        screenshot_path = "agent_assets/icon_classifier_data/validation_icon_classifier.png"
        inventory_screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        inventory_screenshot = adjust_img_color_format(inventory_screenshot)
        true_slots: dict = {'0': 'dirt', '1': 'coarse_dirt', '2': 'podzol', '3': 'rooted_dirt', '4': 'wooden_axe',
                            '5': 'stone_axe', '6': 'wooden_pickaxe', '7': 'stone_pickaxe', '8': 'wooden_shovel',
                            '9': 'oak_fence', '10': 'spruce_fence', '11': 'birch_fence', '12': 'jungle_fence',
                            '13': 'acacia_fence', '14': 'dark_oak_fence', '15': 'mangrove_fence',
                            '16': 'crimson_fence', '17': 'warped_fence', '18': 'oak_log', '19': 'spruce_log',
                            '20': 'birch_log', '21': 'jungle_log', '22': 'acacia_log', '23': 'dark_oak_log',
                            '24': 'mangrove_log', '25': 'stripped_oak_log', '26': 'stripped_spruce_log',
                            '27': 'stone_shovel', '28': 'iron_pickaxe', '29': 'cobblestone', '30': 'mossy_cobblestone',
                            '31': 'stick', '32': 'coal', '33': 'raw_iron',
                            '34': 'iron_ingot', '35': 'diamond'}
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(inventory_screenshot))
        extra_val_x = []
        extra_val_y = []
        for slot_id in range(len(slot_positions)):
            slot_img = get_cropped_slot_img(inventory_screenshot, slot_id, slot_positions)
            slot_img = adjust_inv_item_background(slot_img)
            slot_img = cv2.cvtColor(slot_img, cv2.COLOR_BGR2RGB)
            true_label = true_slots[str(slot_id)]
            extra_val_x.append(slot_img)
            extra_val_y.append(true_label)
        extra_val_y_int = []
        for y in extra_val_y:
            y_int = label_2_int[y]
            extra_val_y_int.append([y_int])
        extra_val_y_one_hot = onehot_encoder.fit_transform(extra_val_y_int)
        return extra_val_x, extra_val_y_one_hot

    def get_extra_train_data(self, label_2_int, onehot_encoder):
        screenshot_path = "agent_assets/icon_classifier_data/train_icon_classifier.png"
        inventory_screenshot: np.ndarray = np.array(Image.open(screenshot_path))
        inventory_screenshot = adjust_img_color_format(inventory_screenshot)
        true_slots: dict = {}
        for i in range(36):
            true_slots[str(i)] = 'oak_log'
        slot_positions = get_slot_positions(is_inventory_recipe_book_open(inventory_screenshot))
        extra_x = []
        extra_y = []
        for slot_id in range(len(slot_positions)):
            slot_img = get_cropped_slot_img(inventory_screenshot, slot_id, slot_positions)
            slot_img = adjust_inv_item_background(slot_img)
            slot_img = cv2.cvtColor(slot_img, cv2.COLOR_BGR2RGB)
            true_label = true_slots[str(slot_id)]
            extra_x.append(slot_img)
            extra_y.append(true_label)
        extra_y_int = []
        for y in extra_y:
            y_int = label_2_int[y]
            extra_y_int.append([y_int])
        extra_y_one_hot = onehot_encoder.fit_transform(extra_y_int)
        return extra_x, extra_y_one_hot

    def train_model(self, classes_count, int_2_label, testDataLoader, trainDataLoader, valDataLoader):

        INIT_LR = self.config_dict['init_lr']
        EPOCHS = self.config_dict['epochs']
        BATCH_SIZE = self.config_dict['batch_size']



        # calculate steps per epoch for training and validation set
        trainSteps = len(trainDataLoader.dataset) // BATCH_SIZE
        valSteps = len(valDataLoader.dataset) // BATCH_SIZE
        print("training...")
        session_id: str = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        print("training session", session_id)
        model_dir = "tmp/icon_classifier_trainer_output/" + session_id

        self.best_model_filename = model_dir + "/model_state_dict.pt"

        class_labels_filename = os.path.join(model_dir, "class_labels.json")
        save_dict_as_json({"int_2_label": int_2_label}, None, class_labels_filename)
        config_filename = os.path.join(model_dir, "config.json")
        save_dict_as_json(self.config_dict, None, config_filename)

        print('Using device for training:', device)
        model = get_model(classes_count, self.config_dict["model"], in_channels=self.config_dict["channels"])
        model = model.to(device)
        # initialize our optimizer and loss function
        opt = Adam(model.parameters(), lr=INIT_LR)
        lossFn = nn.CrossEntropyLoss()
        early_stopper = EarlyStopper(patience=3, min_delta=10)

        # setup wandb run
        session_id: str = str(datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))

        wandb_parameters: dict = self.config_dict
        wandb_sweep_config = {
            'name': 'sweep',
            'method': 'grid',  # grid, random, bayes
            'metric': {'goal': 'maximize', 'name': 'val_acc'},
            'parameters': {
                "train_data_len": {"value": len(trainDataLoader.dataset)},
                "val_data_len": {"value": len(valDataLoader.dataset)}
            }
        }
        for key in wandb_parameters:
            wandb_sweep_config["parameters"][key] = {"values": [wandb_parameters[key]]}
        run = wandb.init(project="icon-classifier", id=session_id)
        wandb.watch(model)
        model_artifact = wandb.Artifact(run.project + '_' + session_id, type='model')

        model_artifact.add_file(class_labels_filename)
        model_artifact.add_file(config_filename)

        # initialize a dictionary to store training history
        H: dict = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "best_acc": 0
        }
        self.best_model_history = copy.deepcopy(H)
        startTime = time.time()
        # loop over our epochs
        step = 0
        epoch = 0
        for epoch in range(0, EPOCHS):

            # set the model in training mode
            model.train()
            # initialize the total training and validation loss
            totalTrainLoss = 0
            totalValLoss = 0
            # initialize the number of correct predictions in the training
            # and validation step
            trainCorrect = 0
            valCorrect = 0
            # loop over the training set
            step = 0
            for (x, y) in trainDataLoader:
                model.train()
                # send the input to the device
                (x, y) = (x.to(device), y.to(device))
                # perform a forward pass and calculate the training loss
                predictions_tensor = model(x)
                loss = lossFn(predictions_tensor, y)
                # zero out the gradients, perform the backpropagation step,
                # and update the weights
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far and
                # calculate the number of correct predictions
                totalTrainLoss += loss

                predictions = predictions_tensor.detach().cpu().numpy()
                predicted_labels = one_hots_to_labels(int_2_label, predictions)
                labels_y = y.detach().cpu().numpy()
                true_labels = one_hots_to_labels(int_2_label, labels_y)
                for a, b in zip(predicted_labels, true_labels):
                    if a == b:
                        trainCorrect += 1
                step += 1

            # switch off autograd for evaluation
            with torch.no_grad():
                # set the model in evaluation mode
                model.eval()
                # loop over the validation set
                for (x, y) in valDataLoader:
                    # send the input to the device
                    (x, y) = (x.to(device), y.to(device))
                    # make the predictions and calculate the validation loss
                    predictions_tensor = model(x)
                    totalValLoss += lossFn(predictions_tensor, y)
                    # calculate the number of correct predictions
                    predictions = predictions_tensor.detach().cpu().numpy()
                    predicted_labels = one_hots_to_labels(int_2_label, predictions)
                    labels_y = y.detach().cpu().numpy()
                    true_labels = one_hots_to_labels(int_2_label, labels_y)
                    for a, b in zip(predicted_labels, true_labels):
                        if a == b:
                            valCorrect += 1
            # calculate the average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgValLoss = totalValLoss / valSteps
            # calculate the training and validation accuracy
            trainCorrect = trainCorrect / len(trainDataLoader.dataset)
            valCorrect = valCorrect / len(valDataLoader.dataset)
            # update our training history
            H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            H["train_acc"].append(trainCorrect)
            avg_val_loss = avgValLoss.cpu().detach().numpy()
            H["val_loss"].append(avg_val_loss)
            H["val_acc"].append(valCorrect)

            self.save_model(model, model_dir, H, epoch, model_artifact)

            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(epoch + 1, EPOCHS))
            print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
                avgTrainLoss, trainCorrect))
            print("Val loss: {:.6f}, Val accuracy: {:.4f}".format(
                avgValLoss, valCorrect))
            print("best val acc: " + str(H["best_acc"]))
            print("\n")

            metrics: dict = {"session_id": session_id,
                             "val_loss": avgValLoss, "train_loss": avgTrainLoss, "train_acc": trainCorrect,
                             "val_acc": valCorrect,
                             "epoch": epoch, "step": step}
            self.metrics.update(metrics)

            wandb.log(self.metrics)

            if early_stopper.early_stop(avg_val_loss):
                logging.warning("Early stopping")
                break

        # finish measuring how long training took
        endTime = time.time()
        print("[INFO] total time taken to train the model: {:.2f}s".format(
            endTime - startTime))
        # we can now evaluate the network on the test set
        print("[INFO] evaluating network...")
        test_x = []
        for (x, y) in testDataLoader:
            test_x.append(x)
        # initialize a list to store our predictions

        model.eval()
        start_predict_time = time.time()
        predicted_labels = predict(int_2_label, model, test_x)
        self.metrics["predict_time"] = time.time() - start_predict_time

        acc_test_1 = get_acc_test_data_1(model, self.config_dict, int_2_label)
        acc_test_2 = get_acc_test_data_2(model, self.config_dict, int_2_label)

        test_acc = float(np.mean([acc_test_1, acc_test_2]))
        print("test acc: " + str(test_acc))
        self.metrics["test_acc"] = test_acc
        wandb.log(self.metrics)
        print(predicted_labels)

        # plot the training loss and accuracy
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(H["train_loss"], label="train_loss")
        plt.plot(H["val_loss"], label="val_loss")
        plt.plot(H["train_acc"], label="train_acc")
        plt.plot(H["val_acc"], label="val_acc")
        plt.title("Training Loss and Accuracy on Dataset")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        # serialize the model to disk
        plt.savefig(model_dir + "/plot.png")

        if os.path.isfile(self.best_model_filename):
            model_artifact.add_file(self.best_model_filename)
        run.log_artifact(model_artifact)
        wandb.finish()

    def save_model(self, model, model_dir: str, history, epoch_num: int, model_artifact):
        model.eval()
        if not os.path.exists(model_dir):
            # Create a new directory because it does not exist
            os.makedirs(model_dir)

        if epoch_num is not None and epoch_num > int(self.config_dict["epochs"]/2) and epoch_num % 50 == 0:
            best_model_metrics = self.get_best_model_metrics()
            save_dict_as_json(best_model_metrics, model_dir,
                              'epoch_' + str(epoch_num) + "metrics.json")

            model_filename = model_dir + "/" + 'epoch_' + str(epoch_num) + "model_state_dict.pt"
            torch.save(self.best_model.state_dict(), model_filename)
            model_artifact.add_file(model_filename)

        if epoch_num is not None and epoch_num > 0:
            if history["val_acc"][-1] >= history["best_acc"] or math.isclose(history["val_acc"][-1],
                                                                             history["best_acc"]):
                history["best_acc"] = history["val_acc"][-1]

                self.best_model = copy.deepcopy(model).to('cpu')
                self.best_model_history = copy.deepcopy(history)

            best_model_metrics = self.get_best_model_metrics()
            save_dict_as_json(best_model_metrics, model_dir, "metrics.json")
            torch.save(self.best_model.state_dict(), self.best_model_filename)
            print("Model saved to dir:", self.best_model_filename)

    def get_best_model_metrics(self):
        best_model_metrics: dict = {
            "best_val_acc": float(self.best_model_history["best_acc"]),
            "val_acc": float(self.best_model_history["val_acc"][-1]),
            "train_acc": float(self.best_model_history["train_acc"][-1]),
            "train_loss": float(self.best_model_history["train_loss"][-1]),
            "val_loss": float(self.best_model_history["val_loss"][-1])
        }
        return best_model_metrics
