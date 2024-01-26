import logging


import torch.nn.functional as F
from torch import nn
from torchvision import models

from src.a3c.nature_cnn import get_nature_cnn_arch
from src.common.load_pretrained_vision_encoder import check_for_pretrained_model


class A3CWithVisionEncoder(nn.Module):
    """
    A3C implementation for Reinforcement Learning in the Environment Minecraft.
    This implementation is used as global net instance and multiple actor net instances.
    Neural network that approximate the policy and value function.
    
    Simple example of A3C: https://dilithjay.com/blog/actor-critic-methods-a-quick-introduction-with-code/
    """

    def __init__(self,  train_config: dict, device, model_name):
        super(A3CWithVisionEncoder, self).__init__()
        self.device = device
        self.train_config = train_config
        output_kernel_size = [1, 1, 1]  # default output layer kernel size
        if self.train_config["input_depth"] < 32:
            output_kernel_size[2] = int(32/self.train_config["input_depth"])

        self.vision_encoder_out_dim = self.train_config["vision_encoder_out_dim"]

        self.frozen_layers = 0
        self.unfreeze_layer_id = 0
        if model_name.startswith("res"):
            self.cnn = models.get_model(model_name, num_classes=self.vision_encoder_out_dim)
        elif model_name.lower() == "naturecnn":
            self.cnn = get_nature_cnn_arch(self.train_config, self.vision_encoder_out_dim)

        logging.info("check_for_pretrained_model")
        logging.info(check_for_pretrained_model(self.train_config["pretrained_vision_encoder_model_path"]))
        logging.info("freeze_pretrained_vision_encoder_weights")
        logging.info(train_config["freeze_pretrained_vision_encoder_weights"])

        if train_config["freeze_pretrained_vision_encoder_weights"] and check_for_pretrained_model(self.train_config["pretrained_vision_encoder_model_path"]):
            logging.info("freezing cnn")
            for child in self.cnn.children():
                logging.info("freeze " + str(child))
                self.frozen_layers += 1
                self.unfreeze_layer_id = self.frozen_layers-1
                for param in child.parameters():
                    param.requires_grad = False

            #for p in self.resnet.parameters():
            #    p.requires_grad = False

        action_dim = self.train_config["action_dim"]
        self.dropout_rate = self.train_config["dropout"]

        # classification head pi
        self.fc_pi2 = nn.Linear(self.vision_encoder_out_dim, action_dim)  # actor output layer (before softmax)

        # classification head v
        self.fc_v2 = nn.Linear(self.vision_encoder_out_dim, 1)  # critic output layer

        self.print_info()

    def print_info(self):
        logging.info(str(self.get_total_parameters()) + " total parameters")
        logging.info(str(self.get_total_trainable_parameters()) + " trainable parameters")
        self.count_frozen_cnn_layers()

    def forward(self, x_2d_image, x_tensor_state_seq):

        # features from image sequence
        x = F.dropout(F.relu(self.cnn(x_2d_image)), self.dropout_rate)

        pi_logits = self.fc_pi2(x)  # actor output without softmax activation
        value = self.fc_v2(x)  # critic output, a.k.a. reward value for the state (approx. value function output)
        return pi_logits, value

    def get_total_parameters(self):
        return sum(self.get_parameters())

    def get_parameters(self):
        return [param.nelement() for param in self.parameters()]

    def get_total_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def try_unfreeze_all_cnn_layers(self):
        self.print_info()
        self.count_frozen_cnn_layers()
        if self.train_config["freeze_pretrained_vision_encoder_weights"] and self.train_config[
            "unfreeze_pretrained_vision_encoder_weights"] and self.frozen_layers > 0:
            for child in self.cnn.children():
                print("unfreeze " + str(child))
                for param in child.parameters():
                    param.requires_grad = True
            self.count_frozen_cnn_layers()
            return True
        return False

    def count_frozen_cnn_layers(self):
        self.frozen_layers = 0
        for child in self.cnn.children():
            frozen_layer_found = False
            for param in child.parameters():
                if not param.requires_grad:
                    frozen_layer_found = True
            if frozen_layer_found:
                self.frozen_layers += 1
        print(str(self.frozen_layers) + " cnn layers are frozen")
        return self.frozen_layers

    def try_unfreeze_next_layer(self):
        """
        Unfreeze next layer of the resnet beginning from behind at the output side of the net
        """

        if self.frozen_layers <= 0:
            return
        i = 0
        for child in self.cnn.children():
            if i == self.unfreeze_layer_id:
                print("unfreeze " + str(child))
                for param in child.parameters():
                    param.requires_grad = True
                self.unfreeze_layer_id -= 1
                self.frozen_layers -= 1
            i += 1

        self.print_info()

        return self.frozen_layers
