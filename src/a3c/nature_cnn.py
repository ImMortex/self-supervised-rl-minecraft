import torch
from torch import nn


def get_nature_cnn_arch(train_config, vision_encoder_out_dim):
    cnn = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(),
        nn.Flatten(),
    )
    # Compute shape by doing one forward pass
    with torch.no_grad():
        sample = torch.randn(8, 3, train_config["img_size"][0], train_config["img_size"][1] * train_config["input_depth"])
        n_flatten = cnn(sample.float()).shape[1]
    cnn.append(nn.Sequential(nn.Linear(n_flatten, vision_encoder_out_dim), nn.ReLU()))
    return cnn
