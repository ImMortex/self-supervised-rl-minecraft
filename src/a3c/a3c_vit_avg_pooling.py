import torch
import torch.nn.functional as F
from torch import nn

from src.swin_unetr.models.ssl_head import SSLHead


class A3CViTAvgPooling(nn.Module):
    """
    A3C implementation for Reinforcement Learning in the Environment Minecraft.
    This implementation is used as global net instance and multiple actor net instances.
    Neural network that approximate the policy and value function.
    
    Simple example of A3C: https://dilithjay.com/blog/actor-critic-methods-a-quick-introduction-with-code/
    """

    def __init__(self, ssl_head_args, train_config: dict, device):
        super(A3CViTAvgPooling, self).__init__()
        self.device = device
        self.train_config = train_config
        output_kernel_size = [1, 1, 1]  # default output layer kernel size
        if self.train_config["input_depth"] < 32:
            output_kernel_size[2] = int(32/self.train_config["input_depth"])
        self.image_encoder = SSLHead(
            ssl_head_args, output_kernel_size=tuple(output_kernel_size)).swinViT  # only ViT

        if train_config["freeze_pretrained_vision_encoder_weights"] and check_for_pretrained_model(self.train_config["pretrained_vision_encoder_model_path"]):
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.H = train_config["img_size"][0]
        self.W = train_config["img_size"][1]
        self.D = self.train_config["input_depth"]
        vit_features = 768  # constant from paper https://arxiv.org/abs/2111.14791

        action_dim = self.train_config["action_dim"]

        self.hidden_dim = self.train_config["dense_hidden_dim"]
        self.dropout_rate = self.train_config["dropout"]
        self.state_dim = 768

        # classification head pi
        self.fc_pi = nn.Linear(self.state_dim, self.hidden_dim)  # initialized in first forward call
        self.fc_pi2 = nn.Linear(self.hidden_dim, action_dim)  # actor output layer (before softmax)

        # classification head v
        self.fc_v = nn.Linear(self.state_dim, self.hidden_dim)  # initialized in first forward call
        self.fc_v2 = nn.Linear(self.hidden_dim, 1)  # critic output layer

        self.a_avg_pooling = nn.AdaptiveAvgPool3d(1)

    def forward(self, x_3d_image, x_tensor_state_seq):

        # features from image sequence
        x_image_out = self.image_encoder(x_3d_image)[-1]  # image encoder (optional pretrained)

        x = self.a_avg_pooling(x_image_out)
        x = F.dropout(F.relu(torch.flatten(x, start_dim=1, end_dim=- 1)), self.dropout_rate)  # flatten, keep batch dimension
        x_pi = F.dropout(F.relu(self.fc_pi(x)), self.dropout_rate)
        x_v = F.dropout(F.relu(self.fc_v(x)), self.dropout_rate)

        pi_logits = self.fc_pi2(x_pi)  # actor output without softmax activation
        value = self.fc_v2(x_v)  # critic output, a.k.a. reward value for the state (approx. value function output)
        return pi_logits, value

    def get_total_parameters(self):
        return sum(self.get_parameters())

    def get_parameters(self):
        return [param.nelement() for param in self.parameters()]
