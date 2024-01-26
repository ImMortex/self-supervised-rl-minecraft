import torch
import torch.nn.functional as F
from torch import nn

from src.a3c.gru_network import GruNet
from src.swin_unetr.models.ssl_head import SSLHead


class A3CDense(nn.Module):
    """
    A3C implementation for Reinforcement Learning in the Environment Minecraft.
    This implementation is used as global net instance and multiple actor net instances.
    Neural network that approximate the policy and value function.
    
    Simple example of A3C: https://dilithjay.com/blog/actor-critic-methods-a-quick-introduction-with-code/
    """

    def __init__(self, ssl_head_args, train_config: dict, device):
        super(A3CDense, self).__init__()
        self.device = device
        self.train_config = train_config


        action_dim = self.train_config["action_dim"]

        feature_dim = self.train_config["no_vision_state_dim"]
        self.dropout_rate = self.train_config["dropout"]


        self.state_dim = int(feature_dim * self.train_config["input_depth"])

        self.hidden_dim = self.train_config["dense_hidden_dim"]

        # classification head pi
        self.fc_pi = nn.Linear(self.state_dim, self.hidden_dim)  # initialized in first forward call
        self.fc_pi2 = nn.Linear(self.hidden_dim, action_dim)  # actor output layer (before softmax)

        # classification head v
        self.fc_v = nn.Linear(self.state_dim, self.hidden_dim)  # initialized in first forward call
        self.fc_v2 = nn.Linear(self.hidden_dim, 1)  # critic output layer

    def forward(self, x_3d_image, x_tensor_state_seq):
        x = torch.flatten(x_tensor_state_seq, start_dim=1, end_dim=- 1)  # flatten, keep batch dimension
        x_pi = F.dropout(F.relu(self.fc_pi(x)), self.dropout_rate)
        x_v = F.dropout(F.relu(self.fc_v(x)), self.dropout_rate)

        pi_logits = self.fc_pi2(x_pi)  # actor output without softmax activation
        value = self.fc_v2(x_v)  # critic output, a.k.a. reward value for the state (approx. value function output)
        return pi_logits, value
    
    def get_total_parameters(self):
        return sum(self.get_parameters())

    def get_parameters(self):
        return [param.nelement() for param in self.parameters()]
