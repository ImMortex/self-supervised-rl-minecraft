import logging

import torch.nn as nn
import torchvision.models as models

from src.a3c.nature_cnn import get_nature_cnn_arch
from src.trainers.exepctions import InvalidBackboneError
import torchvision.transforms as T
import torch.nn.functional as F

class ResNetSimCLR(nn.Module):

    def __init__(self, base_model: str, out_dim: int, train_config):
        super(ResNetSimCLR, self).__init__()

        logging.info("vision encoder  architecture: " + base_model)
        self.resnet_dict: dict = {}
        if base_model.lower().startswith("res"):
            self.resnet_dict: dict = {base_model: models.get_model(base_model, num_classes=out_dim)}
        elif base_model.lower() == "naturecnn":
            self.resnet_dict: dict = {base_model: get_nature_cnn_arch(train_config, out_dim)}

        self.backbone = self._get_basemodel(base_model)
        #dim_mlp = self.backbone.fc.in_features

        out_features = out_dim #self.backbone.fc.out_features

        # add mlp projection head

        self.fc1 = nn.Linear(out_features, out_features)
        self.fc_out = nn.Linear(out_features, out_features)
        #self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        self.dropout_rate = 0.2

    def _get_basemodel(self, model_name: str):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file")
        else:
            return model

    def forward(self, x):
        feat = self.backbone(x)
        x_contrast = F.dropout(F.relu(self.fc1(feat)), self.dropout_rate)
        x_contrast = self.fc_out(x_contrast)

        return x_contrast