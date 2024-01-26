import logging
from collections import OrderedDict
from datetime import timedelta

import numpy as np
import torch
import wandb


def gradients_torch_tensor_to_list(gradients_each_layer: []) -> []:
    new_list: [] = []
    for g in gradients_each_layer:
        if g is None:
            new_list.append(g)
        else:
            new_list.append(g.cpu().detach().numpy().tolist())
    return new_list

def gradients_list_to_torch_tensor(gradients_each_layer: [], device) -> []:
    new_list: [] = []
    for g in gradients_each_layer:
        if g is None:
            new_list.append(g)
        else:
            new_list.append(torch.tensor(np.array(g)).float().to(device))
    return new_list


def save_gradients(model) -> []:
    """
    Returns Gradients (Tensor) of each nn layer (list of Tensors)
    A3C https://discuss.pytorch.org/t/continuous-action-a3c/1033 (11.08.2023)
    """
    not_none_count: int = 0
    gradients_each_layer = []

    # save gradients
    for idx, param in enumerate(model.parameters()):
        parameter = param.grad
        if param.grad is not None:
            not_none_count += 1
            parameter = parameter.cpu()
        gradients_each_layer.append(parameter)
    if not_none_count == 0:
        logging.error("save_gradients saved only None values")

    return gradients_each_layer


def apply_gradients(model, gradients_each_layer: []):
    """
    Overwrite existing gradients of the model. Then do not use optimizer.step()!!!
    A3C https://discuss.pytorch.org/t/continuous-action-a3c/1033 (11.08.2023)
    """
    # apply gradients
    for idx, param in enumerate(model.parameters()):
        if gradients_each_layer[idx] is not None:
            grad = gradients_each_layer[idx].to(param.device)
            param.grad = grad
        else:
            param.grad = gradients_each_layer[idx]


def get_weights(model) -> OrderedDict:
    """
    Get weights of the model.
    A3C https://discuss.pytorch.org/t/continuous-action-a3c/1033 (11.08.2023)
    """
    return model.state_dict()


def apply_weights(model, trained_model_state_dict: OrderedDict):
    """
    Overwrite existing weights of the model.
    A3C https://discuss.pytorch.org/t/continuous-action-a3c/1033 (11.08.2023)
    """
    # load weights from stored model file
    model.load_state_dict(trained_model_state_dict)
