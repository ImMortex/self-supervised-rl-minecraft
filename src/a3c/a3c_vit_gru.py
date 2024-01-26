import torch
import torch.nn.functional as F
from torch import nn

from src.common.load_pretrained_vision_encoder import check_for_pretrained_model
from src.swin_unetr.models.ssl_head import SSLHead


class A3CMcRlNet(nn.Module):
    """
    A3C implementation for Reinforcement Learning in the Environment Minecraft.
    This implementation is used as global net instance and multiple actor net instances.
    Neural network that approximate the policy and value function.
    
    Simple example of A3C: https://dilithjay.com/blog/actor-critic-methods-a-quick-introduction-with-code/
    """

    def __init__(self, ssl_head_args, train_config: dict, device):
        super(A3CMcRlNet, self).__init__()
        self.device = device
        self.train_config = train_config
        output_kernel_size = [1, 1, 1]  # default output layer kernel size
        if self.train_config["input_depth"] < 32:
            output_kernel_size[2] = int(32/self.train_config["input_depth"])
        self.image_encoder = SSLHead(
            ssl_head_args, output_kernel_size=tuple(output_kernel_size)).swinViT  # only encoder of swin_unetr SSLHead, optional pretrained subnet

        if train_config["freeze_pretrained_vision_encoder_weights"] and check_for_pretrained_model(self.train_config["pretrained_vision_encoder_model_path"]):
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.H = train_config["img_size"][0]
        self.W = train_config["img_size"][1]
        self.D = self.train_config["input_depth"]
        vit_features = 768  # constant from paper https://arxiv.org/abs/2111.14791

        # dimensions last encoder layer from paper https://arxiv.org/abs/2111.14791
        img_out_flatten_dim = max(self.H / 32, 1) * max(self.W / 32, 1) * max(self.D / 32, 1) * vit_features

        action_dim = self.train_config["action_dim"]
        gru_hidden_dim = 10
        gru_feature_dim = self.train_config["no_vision_state_dim"]
        gru_dropout = 0.2
        gru_n_layers = 1
        self.state_gru: GruNet = GruNet(input_dim=gru_feature_dim, hidden_dim=gru_hidden_dim,
                                        output_dim=gru_feature_dim, n_layers=gru_n_layers, drop_prob=gru_dropout)

        self.state_dim = int(img_out_flatten_dim + gru_feature_dim)

        self.hidden_dim = self.train_config["dense_hidden_dim"]
        self.dropout_rate = self.train_config["dropout"]

        # classification head pi
        self.fc_pi = nn.Linear(self.state_dim, self.hidden_dim)  # initialized in first forward call
        self.fc_pi2 = nn.Linear(self.hidden_dim, action_dim)  # actor output layer (before softmax)

        # classification head v
        self.fc_v = nn.Linear(self.state_dim, self.hidden_dim)  # initialized in first forward call
        self.fc_v2 = nn.Linear(self.hidden_dim, 1)  # critic output layer

    def forward(self, x_3d_image, x_tensor_state_seq):
        # vision encoder feature + additional features

        # features from image sequence
        x_image_out = self.image_encoder(x_3d_image)[-1]  # image encoder (optional pretrained)

        # Extract the remaining features of the state. Using GRUs many to one
        x_state_output, hx = self.state_gru(x_tensor_state_seq, None)
        x_image_out = torch.flatten(x_image_out, start_dim=1, end_dim=- 1)  # flatten, keep batch dimension

        x = torch.cat((
            x_image_out,
            x_state_output
        ), 1)



        x_pi = F.dropout(F.relu(self.fc_pi(x)), self.dropout_rate)
        x_v = F.dropout(F.relu(self.fc_v(x)), self.dropout_rate)

        pi_logits = self.fc_pi2(x_pi)  # actor output without softmax activation
        value = self.fc_v2(x_v)  # critic output, a.k.a. reward value for the state (approx. value function output)
        return pi_logits, value
    
    def get_total_parameters(self):
        return sum(self.get_parameters())

    def get_parameters(self):
        return [param.nelement() for param in self.parameters()]
