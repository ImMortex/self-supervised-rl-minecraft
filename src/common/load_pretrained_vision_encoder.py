import logging
import os
from collections import OrderedDict

import torch

def check_for_pretrained_model(pretrained_model_path):

    if pretrained_model_path is None or pretrained_model_path == "":
        return False

    else:
        if not isinstance(pretrained_model_path, str) and os.path.isfile(pretrained_model_path):
            raise Exception("No model found at " + str(pretrained_model_path))
        return True




def load_pretrained_vision_encoder_weights(use_pretrained_vision_encoder, model,
                                           pretrained_vision_encoder_model_path, device):
    if use_pretrained_vision_encoder:
        # Load pretrained weights of self-supervised learned image encoder
        logging.info("Loading pretrained vision encoder weights " + pretrained_vision_encoder_model_path)
        pretrained_model_state_dict: OrderedDict = torch.load(pretrained_vision_encoder_model_path,
                                                              map_location=device)
        try:
            if 'model_state_dict' in pretrained_model_state_dict:
                model_state_dict = pretrained_model_state_dict['model_state_dict']
            else:
                model_state_dict = pretrained_model_state_dict
        except TypeError:
            # Maybe it is the entire model so we can get state_dict() from it
            model_state_dict = pretrained_model_state_dict.state_dict()

        success_text: str = "Pretrained vision encoder weights loaded"
        if hasattr(model, 'image_encoder') and model.image_encoder is not None:
            model.image_encoder.load_state_dict(model_state_dict)
            logging.info(success_text)
        elif hasattr(model, 'resnet') and model.cnn is not None:
            model.cnn.load_state_dict(model_state_dict)
            logging.info(success_text)
        elif hasattr(model, 'vision_encoder') and model.vision_encoder is not None:
            model.vision_encoder.load_state_dict(model_state_dict)
            logging.info(success_text)
        elif hasattr(model, 'cnn') and model.cnn is not None:
            model.cnn.load_state_dict(model_state_dict)
            logging.info(success_text)
    else:
        logging.info("No pretrained vision encoder weights loaded")
